"""
swing.py — SwingStrategy: data fetching and per-fold simulation for swing walk-forward.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from scripts.walkforward.gates import (
    FoldResult, compute_profit_factor, compute_calmar, compute_k_ratio, fold_years,
)

logger = logging.getLogger(__name__)


class SwingStrategy:
    """Encapsulates everything swing-specific: data download, fold simulation."""

    def __init__(
        self,
        model,
        version: int,
        symbols: List[str],
        atr_stop_mult: float = 0.5,
        atr_target_mult: float = 1.5,
        meta_model=None,
        pm_abstention_vix: float = 0.0,
        pm_abstention_spy_ma_days: int = 0,
        pm_abstention_spy_5d: bool = False,
        transaction_cost_pct: float = 0.0005,
        use_opportunity_score: bool = False,
        no_prefilters: bool = False,
        earnings_blackout: Optional[Dict[str, set]] = None,
        feature_cache_workers: int = 0,
        feature_cache_executor: str = "process",
        feature_cache_disable: bool = False,
        sim_scan_interval_days: int = 1,
    ):
        self.model = model
        self.version = version
        self.symbols = symbols
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.meta_model = meta_model
        self.pm_abstention_vix = pm_abstention_vix
        self.pm_abstention_spy_ma_days = pm_abstention_spy_ma_days
        self.pm_abstention_spy_5d = pm_abstention_spy_5d
        self.transaction_cost_pct = transaction_cost_pct
        self.use_opportunity_score = use_opportunity_score
        self.no_prefilters = no_prefilters
        self.earnings_blackout = earnings_blackout
        self.feature_cache_workers = feature_cache_workers
        self.feature_cache_executor = feature_cache_executor
        self.feature_cache_disable = feature_cache_disable
        self.sim_scan_interval_days = sim_scan_interval_days
        self.symbols_data: Dict[str, pd.DataFrame] = {}
        self.spy_prices: Optional[pd.Series] = None

    def fetch_data(self, start: datetime, end: datetime) -> None:
        """Download daily bars for all symbols in the strategy universe."""
        import yfinance as yf
        t0 = time.time()
        logger.info("Downloading daily bars %s -> %s", start.date(), end.date())
        for sym in self.symbols:
            try:
                df = yf.download(sym, start=start.date().isoformat(),
                                 end=end.date().isoformat(), progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                if len(df) >= 210:
                    self.symbols_data[sym] = df
            except Exception:
                pass

        spy_raw = yf.download("SPY", start=start.date().isoformat(),
                              end=end.date().isoformat(), progress=False, auto_adjust=True)
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy_raw.columns = spy_raw.columns.get_level_values(0)
        spy_raw.columns = [c.lower() for c in spy_raw.columns]
        self.spy_prices = spy_raw["close"]

        if self.use_opportunity_score:
            try:
                vix_raw = yf.download("^VIX", start=start.date().isoformat(),
                                      end=end.date().isoformat(), progress=False, auto_adjust=True)
                if isinstance(vix_raw.columns, pd.MultiIndex):
                    vix_raw.columns = vix_raw.columns.get_level_values(0)
                vix_raw.columns = [c.lower() for c in vix_raw.columns]
                if len(vix_raw) >= 5:
                    self.symbols_data["^VIX"] = vix_raw
            except Exception as exc:
                logger.warning("VIX download failed: %s", exc)

        logger.info("Swing data loaded: %d symbols in %.1fs", len(self.symbols_data), time.time() - t0)

    def run_fold(self, fold_idx: int, n_folds: int,
                 tr_start, tr_end, te_start, te_end) -> FoldResult:
        from app.backtesting.agent_simulator import AgentSimulator
        from app.data.universe_history import pit_union as _pit_union, historical_trade_symbols as _hist_syms

        # WF-A2/A3: use Russell 1000 PIT union (matches training universe).
        extra = _hist_syms(tr_start, te_end, trade_type="swing")
        pit_members = set(_pit_union("russell1000", tr_start, te_end, extra_symbols=extra))
        _synthetic = {"^VIX", "VIX", "SPY"}
        fold_symbols_data = {
            s: d for s, d in self.symbols_data.items()
            if s in pit_members or s in _synthetic
        }
        # Build feature cache for this fold (pre-computes all (sym, day) features
        # once in parallel, eliminating per-day per-symbol re-computation).
        feature_cache = None
        if not self.feature_cache_disable:
            try:
                from app.backtesting.feature_cache import build_feature_cache
                import os

                # Collect test-fold trading days
                test_days = sorted({
                    d.date() if hasattr(d, "date") else d
                    for df in fold_symbols_data.values()
                    for d in df.index
                    if te_start <= (d.date() if hasattr(d, "date") else d) <= te_end
                })

                vix_df = fold_symbols_data.get("^VIX") or fold_symbols_data.get("VIX")
                vix_series = vix_df["close"] if vix_df is not None and "close" in vix_df.columns else None

                feature_names = getattr(self.model, "feature_names", None) or []
                cache_workers = self.feature_cache_workers or max(2, min(os.cpu_count() or 4, 12))

                logger.info(
                    "Fold %d: building feature cache (%d symbols × %d days, %d workers, %s)",
                    fold_idx, len(fold_symbols_data), len(test_days),
                    cache_workers, self.feature_cache_executor,
                )
                feature_cache = build_feature_cache(
                    symbols_data=fold_symbols_data,
                    trading_days=test_days,
                    feature_names=feature_names,
                    vix_history=vix_series,
                    workers=cache_workers,
                    executor=self.feature_cache_executor,
                )
            except Exception as exc:
                logger.warning("Feature cache build failed, falling back to live compute: %s", exc)
                feature_cache = None

        sim = AgentSimulator(
            model=self.model,
            atr_stop_mult=self.atr_stop_mult,
            atr_target_mult=self.atr_target_mult,
            meta_model=self.meta_model,
            pm_abstention_vix=self.pm_abstention_vix,
            pm_abstention_spy_ma_days=self.pm_abstention_spy_ma_days,
            pm_abstention_spy_5d=self.pm_abstention_spy_5d,
            transaction_cost_pct=self.transaction_cost_pct,
            use_opportunity_score=self.use_opportunity_score,
            no_prefilters=self.no_prefilters,
            earnings_blackout=self.earnings_blackout,
            feature_cache=feature_cache,
            sim_scan_interval_days=self.sim_scan_interval_days,
        )
        result = sim.run(
            fold_symbols_data,
            start_date=te_start,
            end_date=te_end,
            spy_prices=self.spy_prices,
        )
        stop_exits = result.exit_breakdown.get("STOP", 0)
        stop_rate = stop_exits / max(result.total_trades, 1)
        trade_returns = getattr(result, "trade_returns", [])
        equity_curve = getattr(result, "equity_curve", [])
        years = fold_years(te_start, te_end)
        return FoldResult(
            fold=fold_idx,
            train_start=tr_start, train_end=tr_end,
            test_start=te_start, test_end=te_end,
            trades=result.total_trades,
            win_rate=result.win_rate,
            sharpe=result.sharpe_ratio,
            max_drawdown=result.max_drawdown_pct,
            total_return=result.total_return_pct,
            stop_exit_rate=stop_rate,
            model_version=self.version,
            profit_factor=compute_profit_factor(trade_returns),
            calmar_ratio=compute_calmar(result.total_return_pct, result.max_drawdown_pct, years),
            k_ratio=compute_k_ratio(equity_curve),
        )
