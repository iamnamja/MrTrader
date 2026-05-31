"""
swing.py — SwingStrategy: data fetching and per-fold simulation for swing walk-forward.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

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
        atr_stop_mult: float = 1.5,   # v216: 1.5×ATR for ranker model (LambdaRank doesn't use triple_barrier stops)
        atr_target_mult: float = 3.0,  # v216: 3.0×ATR target for 1:2 R:R
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
        # Rebalance params (passed through to AgentSimulator)
        rebalance_mode: bool = False,
        rebalance_days: int = 20,
        rebalance_target_n: int = 30,
        rebalance_sector_cap: float = 0.30,
        rebalance_add_threshold: int = 15,
        rebalance_drop_threshold: int = 30,
        rebalance_min_adv: float = 20_000_000.0,
        rebalance_regime_gate: bool = False,
        rebalance_regime_spy_ma_days: int = 200,
        rebalance_regime_vix_bull: float = 20.0,
        rebalance_regime_vix_bear: float = 30.0,
        rebalance_inv_vol: bool = False,
        rebalance_inv_vol_lookback: int = 20,
        rebalance_inv_vol_min_mult: float = 0.5,
        rebalance_inv_vol_max_mult: float = 2.0,
        rebalance_spy_vol_damper: bool = False,
        rebalance_spy_vol_damper_scale: float = 0.50,
        # Phase 89: factor stability gate (WF path only; CPCV path wired in Phase 90)
        rebalance_factor_stability_gate: bool = False,
        rebalance_factor_stability_lookback: int = 63,
        rebalance_factor_stability_ic_threshold: float = 0.02,
        # Phase 89 v2: dispersion gate
        rebalance_dispersion_gate: bool = False,
        rebalance_dispersion_k: int = 5,
        rebalance_dispersion_L: int = 126,
        factor_scorer: Optional[Callable[..., Any]] = None,
        no_atr_stops: bool = False,
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
        self.rebalance_mode = rebalance_mode
        self.rebalance_days = rebalance_days
        self.rebalance_target_n = rebalance_target_n
        self.rebalance_sector_cap = rebalance_sector_cap
        self.rebalance_add_threshold = rebalance_add_threshold
        self.rebalance_drop_threshold = rebalance_drop_threshold
        self.rebalance_min_adv = rebalance_min_adv
        self.rebalance_regime_gate = rebalance_regime_gate
        self.rebalance_regime_spy_ma_days = rebalance_regime_spy_ma_days
        self.rebalance_regime_vix_bull = rebalance_regime_vix_bull
        self.rebalance_regime_vix_bear = rebalance_regime_vix_bear
        self.rebalance_inv_vol = rebalance_inv_vol
        self.rebalance_inv_vol_lookback = rebalance_inv_vol_lookback
        self.rebalance_inv_vol_min_mult = rebalance_inv_vol_min_mult
        self.rebalance_inv_vol_max_mult = rebalance_inv_vol_max_mult
        self.rebalance_spy_vol_damper = rebalance_spy_vol_damper
        self.rebalance_spy_vol_damper_scale = rebalance_spy_vol_damper_scale
        self.rebalance_factor_stability_gate = rebalance_factor_stability_gate
        self.rebalance_factor_stability_lookback = rebalance_factor_stability_lookback
        self.rebalance_factor_stability_ic_threshold = rebalance_factor_stability_ic_threshold
        self.rebalance_dispersion_gate = rebalance_dispersion_gate
        self.rebalance_dispersion_k = rebalance_dispersion_k
        self.rebalance_dispersion_L = rebalance_dispersion_L
        self.factor_scorer = factor_scorer
        self.no_atr_stops = no_atr_stops
        self.symbols_data: Dict[str, pd.DataFrame] = {}
        self.spy_prices: Optional[pd.Series] = None
        # OOS-guard escape hatch: set True to allow test folds inside training period.
        # Results labeled in-sample; gate_passed() always returns False.
        self.allow_in_sample: bool = False

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

        if self.use_opportunity_score or self.rebalance_regime_gate:
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

        # Make SPY available to _make_regime_gate_fn and factor_scorer
        if self.rebalance_mode:
            self.symbols_data["SPY"] = spy_raw

        # C11-6: pre-compute regime_map over the full evaluation window so that VIX
        # quartile thresholds are stable across folds (not re-computed per test window).
        try:
            from scripts.walkforward.regime import load_regime_map as _lrm
            self._global_regime_map = _lrm(start.date(), end.date())
        except Exception:
            self._global_regime_map = {}

        logger.info("Swing data loaded: %d symbols in %.1fs", len(self.symbols_data), time.time() - t0)

    def run_fold(self, fold_idx: int, n_folds: int,
                 tr_start, tr_end, te_start, te_end) -> FoldResult:
        from app.backtesting.agent_simulator import AgentSimulator
        from app.data.universe_history import pit_union as _pit_union, historical_trade_symbols as _hist_syms

        # WF-A2/A3: use Russell 1000 PIT union (matches training universe).
        # Upper bound is tr_end, NOT te_end: names that joined the index between
        # tr_end and te_end are only known in the future — including them is
        # forward-looking survivorship bias. The live trader also cannot know
        # about index joiners before they actually join.
        extra = _hist_syms(tr_start, tr_end, trade_type="swing")
        pit_members = set(_pit_union("russell1000", tr_start, tr_end, extra_symbols=extra))
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

                vix_df = fold_symbols_data.get("^VIX")
                if vix_df is None:
                    vix_df = fold_symbols_data.get("VIX")
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

        _regime_gate_fn = None
        if self.rebalance_mode and self.rebalance_regime_gate:
            from scripts.walkforward_tier3 import _make_regime_gate_fn
            _regime_gate_fn = _make_regime_gate_fn(
                fold_symbols_data,
                spy_ma_days=self.rebalance_regime_spy_ma_days,
                vix_bull=self.rebalance_regime_vix_bull,
                vix_bear=self.rebalance_regime_vix_bear,
            )

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
            factor_scorer=self.factor_scorer,
            no_atr_stops=self.no_atr_stops,
            rebalance_mode=self.rebalance_mode,
            rebalance_days=self.rebalance_days,
            rebalance_target_n=self.rebalance_target_n,
            rebalance_sector_cap=self.rebalance_sector_cap,
            rebalance_add_threshold=self.rebalance_add_threshold,
            rebalance_drop_threshold=self.rebalance_drop_threshold,
            rebalance_min_adv=self.rebalance_min_adv,
            rebalance_regime_fn=_regime_gate_fn,
            rebalance_inv_vol=self.rebalance_inv_vol,
            rebalance_inv_vol_lookback=self.rebalance_inv_vol_lookback,
            rebalance_inv_vol_min_mult=self.rebalance_inv_vol_min_mult,
            rebalance_inv_vol_max_mult=self.rebalance_inv_vol_max_mult,
            rebalance_spy_vol_damper=self.rebalance_spy_vol_damper,
            rebalance_spy_vol_damper_scale=self.rebalance_spy_vol_damper_scale,
        )
        import uuid as _uuid
        sim._wf_run_id = f"wf-fold{fold_idx}-{_uuid.uuid4().hex[:8]}"
        result = sim.run(
            fold_symbols_data,
            start_date=te_start,
            end_date=te_end,
            spy_prices=self.spy_prices,
        )
        stop_exits = result.exit_breakdown.get("STOP", 0)
        stop_rate = stop_exits / max(result.total_trades, 1)
        # Extract per-trade pnl_pct from result.trades for k_ratio.
        # For profit_factor we use result.profit_factor directly (already computed
        # by AgentSimulator with _PF_NO_LOSS_SENTINEL=5.0) to avoid double computation.
        trades_list = getattr(result, "trades", None) or []
        trade_returns = [t.pnl_pct for t in trades_list if hasattr(t, "pnl_pct")]
        equity_curve = getattr(result, "equity_curve", [])
        # n_obs = trading-day return observations for DSR. equity_curve is a list
        # of (date, equity) tuples (one per trading day); diffs give daily returns,
        # hence len-1. Required by deflated_sharpe_ratio; mirrors intraday.py.
        n_obs = max(len(equity_curve) - 1, 0)
        years = fold_years(te_start, te_end)
        # Extract daily returns for Calmar vol-floor (MEDIUM-1)
        _eq_vals = [v for _, v in equity_curve]
        _daily_rets = [((_eq_vals[i] - _eq_vals[i - 1]) / max(_eq_vals[i - 1], 1e-9))
                       for i in range(1, len(_eq_vals))] if len(_eq_vals) >= 2 else []
        from scripts.walkforward.regime import compute_regime_sharpes as _crs
        regime_sharpes = _crs(equity_curve, te_start, te_end,
                              regime_map=getattr(self, "_global_regime_map", None))
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
            profit_factor=getattr(result, "profit_factor", compute_profit_factor(trade_returns)),
            calmar_ratio=compute_calmar(result.total_return_pct, result.max_drawdown_pct, years,
                                        daily_returns=_daily_rets),
            k_ratio=compute_k_ratio(equity_curve),
            n_obs=n_obs,
            regime_sharpes=regime_sharpes,
            avg_capital_deployed_pct=getattr(result, "avg_capital_deployed_pct", 0.0),
            deployment_adjusted_sharpe=getattr(result, "deployment_adjusted_sharpe", 0.0),
            low_deployment_warning=getattr(result, "low_deployment_warning", False),
        )
