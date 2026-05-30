"""
intraday.py — IntradayStrategy: data fetching and per-fold simulation for intraday walk-forward.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from scripts.walkforward.gates import (
    FoldResult, compute_profit_factor, compute_calmar, compute_k_ratio, fold_years,
)

logger = logging.getLogger(__name__)


class IntradayStrategy:
    """Encapsulates everything intraday-specific: 5-min data loading, fold simulation."""

    def __init__(
        self,
        model,
        version: int,
        symbols: List[str],
        meta_model=None,
        pm_abstention_vix: float = 0.0,
        pm_abstention_spy_ma_days: int = 0,
        scan_offsets: Optional[List[int]] = None,
        transaction_cost_pct: float = 0.0015,
        use_opportunity_score: bool = False,
        use_dispersion_gate: bool = False,
        earnings_blackout: Optional[Dict[str, set]] = None,
    ):
        self.model = model
        self.version = version
        self.symbols = symbols
        self.meta_model = meta_model
        self.pm_abstention_vix = pm_abstention_vix
        self.pm_abstention_spy_ma_days = pm_abstention_spy_ma_days
        self.scan_offsets = scan_offsets
        self.transaction_cost_pct = transaction_cost_pct
        self.use_opportunity_score = use_opportunity_score
        self.use_dispersion_gate = use_dispersion_gate
        self.earnings_blackout = earnings_blackout
        self.symbols_data: Dict[str, pd.DataFrame] = {}
        self.spy_data: Optional[pd.DataFrame] = None
        self.spy_daily_data: Optional[pd.DataFrame] = None
        self.all_days_sorted: list = []
        # OOS-guard escape hatch: set True to allow test folds inside training period.
        # Results labeled in-sample; gate_passed() always returns False.
        self.allow_in_sample: bool = False

    def fetch_data(self, start, end) -> None:
        """Load 5-min data from Polygon cache (or yfinance fallback)."""
        from app.data.intraday_cache import load_many, available_symbols as poly_syms
        cache_syms = set(poly_syms())
        if cache_syms:
            logger.info("Loading from Polygon cache (%d symbols available)", len(cache_syms))
            self.symbols_data = load_many(
                [s for s in self.symbols if s in cache_syms],
                start=start, end=end,
            )
        else:
            logger.warning("Polygon cache empty - falling back to yfinance (<=55 days)")
            import yfinance as yf
            self.symbols_data = {}
            for sym in self.symbols[:100]:
                try:
                    df = yf.download(sym, period="55d", interval="5m",
                                     progress=False, auto_adjust=True)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.columns = [c.lower() for c in df.columns]
                    if len(df) >= 12:
                        self.symbols_data[sym] = df
                except Exception:
                    pass

        self.spy_data = self.symbols_data.get("SPY")

        try:
            import yfinance as yf
            _spy_daily = yf.download("SPY", period="3y", progress=False, auto_adjust=True)
            if isinstance(_spy_daily.columns, pd.MultiIndex):
                _spy_daily.columns = _spy_daily.columns.get_level_values(0)
            _spy_daily.columns = [c.lower() for c in _spy_daily.columns]
            if len(_spy_daily) >= 6:
                self.spy_daily_data = _spy_daily
        except Exception as exc:
            logger.warning("SPY daily fetch failed: %s", exc)

        if self.use_opportunity_score:
            try:
                import yfinance as yf
                _vix_daily = yf.download("^VIX", period="3y", progress=False, auto_adjust=True)
                if isinstance(_vix_daily.columns, pd.MultiIndex):
                    _vix_daily.columns = _vix_daily.columns.get_level_values(0)
                _vix_daily.columns = [c.lower() for c in _vix_daily.columns]
                if len(_vix_daily) >= 5:
                    self.symbols_data["^VIX"] = _vix_daily
            except Exception as exc:
                logger.warning("VIX download failed: %s", exc)

        self.all_days_sorted = sorted({
            d for df in self.symbols_data.values()
            for d in pd.to_datetime(df.index).date
        })
        logger.info("Intraday data loaded: %d symbols", len(self.symbols_data))

    def run_fold(self, fold_idx: int, n_folds: int,
                 tr_start, tr_end, te_start, te_end) -> FoldResult:
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        from app.data.universe_history import members_at as _members_at

        pit_members = set(_members_at("russell1000", tr_start))
        fold_symbols_data = {s: d for s, d in self.symbols_data.items() if s in pit_members}
        sim = IntradayAgentSimulator(
            model=self.model,
            meta_model=self.meta_model,
            pm_abstention_vix=self.pm_abstention_vix,
            pm_abstention_spy_ma_days=self.pm_abstention_spy_ma_days,
            scan_offsets=self.scan_offsets,
            transaction_cost_pct=self.transaction_cost_pct,
            use_opportunity_score=self.use_opportunity_score,
            use_dispersion_gate=self.use_dispersion_gate,
            earnings_blackout=self.earnings_blackout,
        )
        result = sim.run(
            fold_symbols_data,
            spy_data=self.spy_data,
            start_date=te_start,
            end_date=te_end,
            spy_daily_data=self.spy_daily_data,
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
