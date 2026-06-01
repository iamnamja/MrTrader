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
        # MEDIUM-3: provenance hint for the data span gate / report header.
        self.data_source: str = "unknown"
        # OOS-guard escape hatch: set True to allow test folds inside training period.
        # Results labeled in-sample; gate_passed() always returns False.
        self.allow_in_sample: bool = False
        # ── Per-fold retraining (Phase 2: true out-of-sample WF/CPCV) ─────────
        # When per_fold_retrain=True, run_fold trains a fresh intraday model on
        # this fold's [tr_start, tr_end] window via _train_cache (no 5-min
        # re-fetch). Otherwise the frozen self.model is scored across all test
        # windows (generalization test, cannot promote).
        self.retrainer = None
        self.per_fold_retrain: bool = False
        self._train_cache = None
        self._purge_days: Optional[int] = None
        self._embargo_days: Optional[int] = None
        # Per-symbol DAILY bars (vol-percentile / 52w features). Fetched ONCE on
        # first per-fold use (cheap vs 5-min) and reused across all folds. The
        # 5-min path never re-fetches per fold (constraint #4).
        self._daily_data: Optional[Dict[str, pd.DataFrame]] = None
        # Liquidity cap forced on by the runner for intraday per-fold feasibility.
        self.top_n_by_liquidity: Optional[int] = None

    def fetch_data(self, start, end) -> None:
        """Load 5-min data from Polygon cache (or yfinance fallback)."""
        from app.data.intraday_cache import load_many, available_symbols as poly_syms
        cache_syms = set(poly_syms())
        # MEDIUM-3: record data provenance for the span gate and report header.
        self.data_source = "polygon-cache" if cache_syms else "yfinance-fallback (<=55d)"
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
        # BUG-22: if SPY 5-min isn't in the Polygon cache, fetch it from yfinance
        # (limited to trailing ~55 days) so the simulator always has a SPY overlay.
        if self.spy_data is None:
            try:
                import yfinance as yf
                _spy5 = yf.download("SPY", period="55d", interval="5m",
                                    progress=False, auto_adjust=True)
                if isinstance(_spy5.columns, pd.MultiIndex):
                    _spy5.columns = _spy5.columns.get_level_values(0)
                _spy5.columns = [c.lower() for c in _spy5.columns]
                if len(_spy5) > 0:
                    self.spy_data = _spy5
                    self.symbols_data["SPY"] = _spy5
                    logger.info("SPY 5-min fetched from yfinance (%d bars)", len(_spy5))
            except Exception as exc:
                logger.warning("SPY 5-min fallback failed: %s — running without SPY overlay.", exc)

        # BUG-10 fix: use explicit start/end rather than period="3y" which leaks
        # future data when as_of is set to a historical date. Add a 1-year buffer
        # before start so MA and rolling-regime indicators have warm-up data.
        try:
            import yfinance as yf
            _spy_start = (start - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
            _spy_end = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            _spy_daily = yf.download("SPY", start=_spy_start, end=_spy_end,
                                     progress=False, auto_adjust=True)
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
                _vix_daily = yf.download("^VIX", start=_spy_start, end=_spy_end,
                                         progress=False, auto_adjust=True)
                if isinstance(_vix_daily.columns, pd.MultiIndex):
                    _vix_daily.columns = _vix_daily.columns.get_level_values(0)
                _vix_daily.columns = [c.lower() for c in _vix_daily.columns]
                if len(_vix_daily) >= 5:
                    self.symbols_data["^VIX"] = _vix_daily
            except Exception as exc:
                logger.warning("VIX download failed: %s", exc)

        # C14-1: defensively clamp the trading-day axis to the requested
        # [start, end] window. load_many() already filters by date, but the fold
        # boundaries (engine._build_trading_day_folds) AND the per-fold train
        # matrix (build_train_matrix_for_window) must agree on the exact same
        # day set, otherwise an early fold can get tr_start before any bars exist
        # → empty train matrix → "no training samples" (the 2nd per-fold empty-
        # matrix bug). Clamping here makes all_days_sorted authoritative and
        # independent of any provider/cache filtering quirk.
        _start_d = start.date() if hasattr(start, "date") else start
        _end_d = end.date() if hasattr(end, "date") else end
        self.all_days_sorted = sorted({
            d for df in self.symbols_data.values()
            for d in pd.to_datetime(df.index).date
            if _start_d <= d <= _end_d
        })
        # C14-1: also clamp the per-symbol 5-min bars to the same window so the
        # matrix builder's train_days (derived from symbols_data) can never span
        # a day outside all_days_sorted. Keeps the two day-axes in lock-step.
        # ^VIX is a DAILY overlay deliberately fetched with a 1-year warm-up
        # buffer (line ~138) — do NOT clamp it, or its rolling/quantile lookback
        # would be stripped. Only the per-symbol 5-min equity bars are clamped.
        _clamped: Dict[str, pd.DataFrame] = {}
        for _sym, _df in self.symbols_data.items():
            if _df is None or len(_df) == 0:
                continue
            if _sym in ("^VIX", "VIX"):
                _clamped[_sym] = _df
                continue
            _idx = pd.to_datetime(_df.index)
            _mask = (_idx.date >= _start_d) & (_idx.date <= _end_d)
            _sub = _df[_mask]
            if len(_sub) > 0:
                _clamped[_sym] = _sub
        self.symbols_data = _clamped
        # Re-point SPY after clamping (object identity changed).
        self.spy_data = self.symbols_data.get("SPY", self.spy_data)
        # C11-6: pre-compute regime_map over the full evaluation window so VIX quartile
        # thresholds are stable across all folds (not re-computed per test window).
        try:
            from scripts.walkforward.regime import load_regime_map as _lrm
            _start_d = start.date() if hasattr(start, "date") else start
            _end_d = end.date() if hasattr(end, "date") else end
            self._global_regime_map = _lrm(_start_d, _end_d)
        except Exception:
            self._global_regime_map = {}
        logger.info("Intraday data loaded: %d symbols", len(self.symbols_data))

    def _ensure_daily_data(self) -> Dict[str, pd.DataFrame]:
        """Per-fold path only: fetch per-symbol DAILY bars ONCE and cache them.

        _symbol_to_rows needs per-symbol daily bars for the vol-percentile / 52w
        features (daily_df). IntradayStrategy.fetch_data deliberately loads only
        the 5-min bars (+ SPY daily overlay), so we lazily fetch the daily bars
        here on first per-fold use. Daily bars for the (reduced) universe over
        ~2yr are cheap relative to the 5-min data and are reused across every
        fold — the hot 5-min path is never re-fetched. Mirrors
        IntradayModelTrainer._fetch_daily_all (same provider helper)."""
        if self._daily_data is not None:
            return self._daily_data
        from datetime import datetime, timedelta
        from app.ml.intraday_training import IntradayModelTrainer
        # Universe = the 5-min symbols actually loaded (excl. daily overlays).
        syms = [s for s in self.symbols_data.keys() if s not in ("^VIX", "VIX")]
        if not self.all_days_sorted:
            self._daily_data = {}
            return self._daily_data
        # C14-1: anchor the daily-bar fetch to the EARLIEST fold train start
        # (all_days_sorted[0], now clamped to the requested window in fetch_data).
        # _fetch_daily_all subtracts a further 365 calendar days internally, so the
        # 52-week / vol-percentile features get ~1yr of daily warm-up BEFORE the
        # first training day. This keeps daily coverage = [first_fold_start - 1yr,
        # last_day] in lock-step with the 5-min day-axis the folds are built from.
        start = datetime.combine(self.all_days_sorted[0], datetime.min.time())
        end = datetime.combine(self.all_days_sorted[-1], datetime.min.time()) + timedelta(days=1)
        # NOTE: the provider arg here only sets the (unused-for-daily) 5-min
        # self._provider. _fetch_daily_all sources the per-symbol DAILY bars from
        # INTRADAY_DAILY_FEATURE_PROVIDER (full-history, default yfinance), NOT from
        # self._provider — so the 52w/vol features get full daily coverage here too.
        trainer = IntradayModelTrainer(provider="alpaca")
        try:
            self._daily_data = trainer._fetch_daily_all(syms, start, end) or {}
        except Exception as exc:
            logger.warning("Per-fold daily-bar fetch failed (%s) — vol/52w features "
                           "will use defaults for this run.", exc)
            self._daily_data = {}
        logger.info("Per-fold: cached daily bars for %d/%d symbols (fetched once)",
                    len(self._daily_data), len(syms))

        # Feasibility: cap the universe to top-N by 20-day median dollar volume.
        # Full Russell-1000 per-fold intraday is OOM-infeasible (5-min features
        # rebuilt per training window). Applied ONCE here; shrinks both the 5-min
        # symbols_data the matrix builder iterates and the cached daily_data.
        top_n = getattr(self, "top_n_by_liquidity", None)
        if top_n is not None and self._daily_data:
            dv_scores: Dict[str, float] = {}
            for sym, df in self._daily_data.items():
                if df is None or len(df) < 5:
                    continue
                tail = df.tail(20)
                dv = (tail["close"] * tail["volume"]).median()
                dv_scores[sym] = float(dv) if pd.notna(dv) else 0.0
            keep = set(sorted(dv_scores, key=lambda s: dv_scores[s], reverse=True)[:top_n])
            before = len([s for s in self.symbols_data if s not in ("^VIX", "VIX", "SPY")])
            self.symbols_data = {
                s: d for s, d in self.symbols_data.items()
                if s in keep or s in ("^VIX", "VIX", "SPY")
            }
            self._daily_data = {s: d for s, d in self._daily_data.items() if s in keep}
            logger.info("Per-fold liquidity filter: kept %d/%d symbols (top-%d by 20d "
                        "median dollar volume)", len(keep), before, top_n)
        return self._daily_data

    def run_fold(self, fold_idx: int, n_folds: int,
                 tr_start, tr_end, te_start, te_end) -> FoldResult:
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        from app.data.universe_history import members_at as _members_at

        # ── Per-fold retraining: train a fresh model on THIS fold's training
        # window (true out-of-sample). Otherwise use the frozen self.model. ──
        if self.per_fold_retrain:
            daily_data = self._ensure_daily_data()
            fold_model = self._train_cache.get(
                tr_start, tr_end, self.symbols_data,
                self.spy_data, daily_data, self.spy_daily_data,
            )
            from scripts.walkforward.oos_guard import assert_model_oos
            # Intraday uses TRADING-day purge — pass the trading_day_set so the
            # purge gap is counted in trading days, not calendar days.
            assert_model_oos(
                trained_through=fold_model.trained_through,
                fold_boundaries=[(tr_start, tr_end, te_start, te_end)],
                purge_days=self._purge_days if self._purge_days is not None else 0,
                model_label=f"intraday per-fold@{tr_end}",
                allow_in_sample=getattr(self, "allow_in_sample", False),
                trading_day_set=set(self.all_days_sorted) if self.all_days_sorted else None,
            )
            _fold_model = fold_model
        else:
            _fold_model = self.model

        # BUG-9 fix: use PIT membership at te_start, not tr_start.
        # _members_at(tr_start) includes stocks that delisted/crashed between tr_start
        # and te_start (de-listing bias), while additions after tr_start are correctly
        # excluded. Using te_start as the PIT date eliminates de-listing bias without
        # any look-ahead into the test window (te_start is the first day we trade).
        pit_members = set(_members_at("russell1000", te_start))
        fold_symbols_data = {s: d for s, d in self.symbols_data.items() if s in pit_members}
        sim = IntradayAgentSimulator(
            model=_fold_model,
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
        # Extract per-trade pnl_pct from result.trades (SimResult.trades is the
        # canonical source; result.trade_returns does not exist on SimResult).
        trades_list = getattr(result, "trades", None) or []
        trade_returns = [t.pnl_pct for t in trades_list if hasattr(t, "pnl_pct")]
        equity_curve = getattr(result, "equity_curve", [])
        years = fold_years(te_start, te_end)
        # n_obs: number of daily-return observations (one per trading day in the fold).
        # equity_curve is a list of (date, equity) pairs sampled once per trading day
        # by IntradayAgentSimulator; len - 1 gives the number of day-over-day returns.
        n_obs = max(len(equity_curve) - 1, 0)
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
