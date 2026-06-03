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
        # LX6b/LX8: hard bear-exit + per-position trailing stop (forwarded to AgentSimulator)
        rebalance_hard_exit_bear: bool = False,
        rebalance_flat_stop_pct: float = 0.0,
        # Phase 89: factor stability gate (WF path only; CPCV path wired in Phase 90)
        rebalance_factor_stability_gate: bool = False,
        rebalance_factor_stability_lookback: int = 63,
        rebalance_factor_stability_ic_threshold: float = 0.02,
        # Phase 89 v2: dispersion gate
        rebalance_dispersion_gate: bool = False,
        rebalance_dispersion_k: int = 5,
        rebalance_dispersion_L: int = 126,
        # RANKER v2 Spike A — dollar-neutral L/S short leg. Defaults equal the
        # AgentSimulator constructor defaults so the long-only path is byte-identical
        # when enable_shorts=False (the swing default). NOT forwarded before Spike A.
        enable_shorts: bool = False,
        long_gross: float = 0.95,
        short_gross: float = 0.55,
        short_target_n: int = 30,
        short_min_adv: float = 50_000_000.0,
        short_add_threshold: int = 15,
        short_drop_threshold: int = 30,
        # RANKER v2 §3.1 re-architecture: NET-sector cap (Failure B fix) + SPY
        # beta-hedge overlay. Defaults OFF → long-only/existing L-S paths byte-identical.
        net_sector_cap: bool = False,
        spy_beta_hedge: bool = False,
        spy_beta_lookback: int = 60,
        spy_hedge_max_gross: float = 0.30,
        factor_scorer: Optional[Callable[..., Any]] = None,
        no_atr_stops: bool = False,
        # RANKER v2 §3.1 — realized net-exposure capture (PURE-ADDITIVE). Defaults
        # to enable_shorts so the dollar-neutral arm captures it automatically and
        # the long-only default path keeps it OFF (byte-identical).
        capture_net_exposure: Optional[bool] = None,
        net_beta_lookback: int = 60,
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
        self.rebalance_hard_exit_bear = rebalance_hard_exit_bear
        self.rebalance_flat_stop_pct = rebalance_flat_stop_pct
        self.rebalance_factor_stability_gate = rebalance_factor_stability_gate
        self.rebalance_factor_stability_lookback = rebalance_factor_stability_lookback
        self.rebalance_factor_stability_ic_threshold = rebalance_factor_stability_ic_threshold
        self.rebalance_dispersion_gate = rebalance_dispersion_gate
        self.rebalance_dispersion_k = rebalance_dispersion_k
        self.rebalance_dispersion_L = rebalance_dispersion_L
        # RANKER v2 Spike A — dollar-neutral L/S short-leg kwargs (forwarded to
        # AgentSimulator in run_fold). enable_shorts=False keeps the long-only
        # path byte-identical.
        self.enable_shorts = enable_shorts
        self.long_gross = long_gross
        self.short_gross = short_gross
        self.short_target_n = short_target_n
        self.short_min_adv = short_min_adv
        self.short_add_threshold = short_add_threshold
        self.short_drop_threshold = short_drop_threshold
        self.net_sector_cap = net_sector_cap
        self.spy_beta_hedge = spy_beta_hedge
        self.spy_beta_lookback = spy_beta_lookback
        self.spy_hedge_max_gross = spy_hedge_max_gross
        self.factor_scorer = factor_scorer
        self.no_atr_stops = no_atr_stops
        # RANKER v2 §3.1 — net-exposure capture. None → follow enable_shorts (on for
        # the dollar-neutral arm, off for long-only). Explicit bool overrides.
        self.capture_net_exposure = (
            bool(enable_shorts) if capture_net_exposure is None else bool(capture_net_exposure)
        )
        self.net_beta_lookback = int(net_beta_lookback)
        # Symbol -> sector for net-sector exposure (populated in fetch_data).
        self.sector_map: Dict[str, str] = {}
        self.symbols_data: Dict[str, pd.DataFrame] = {}
        self.spy_prices: Optional[pd.Series] = None
        # OOS-guard escape hatch: set True to allow test folds inside training period.
        # Results labeled in-sample; gate_passed() always returns False.
        self.allow_in_sample: bool = False
        # ── Per-fold retraining (true out-of-sample WF/CPCV) ──────────────────
        # When per_fold_retrain=True, run_fold trains a fresh model on this fold's
        # [tr_start, tr_end] window via _train_cache (no re-fetch). Otherwise the
        # frozen self.model is scored across all test windows (generalization test).
        self.retrainer = None
        self.per_fold_retrain: bool = False
        self._train_cache = None
        self._purge_days: Optional[int] = None
        self._embargo_days: Optional[int] = None

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

        # RANKER v2 §3.1 — symbol→sector map for net-sector exposure capture and the
        # rebalance sector cap. Static membership (non-leaking attribute).
        try:
            from app.data.sector_map import get_sector_map as _gsm
            self.sector_map = _gsm([s for s in self.symbols_data
                                    if s not in ("SPY", "^VIX", "VIX")]) or {}
        except Exception:
            try:
                from app.utils.constants import SECTOR_MAP as _SM
                self.sector_map = dict(_SM)
            except Exception:
                self.sector_map = {}

        logger.info("Swing data loaded: %d symbols in %.1fs", len(self.symbols_data), time.time() - t0)

    def run_fold(self, fold_idx: int, n_folds: int,
                 tr_start, tr_end, te_start, te_end) -> FoldResult:
        from app.backtesting.agent_simulator import AgentSimulator
        from app.data.universe_history import pit_union as _pit_union, historical_trade_symbols as _hist_syms

        # ── Per-fold retraining: train a fresh model on THIS fold's training
        # window (true out-of-sample). Otherwise use the frozen self.model. ──
        if self.per_fold_retrain:
            fold_model = self._train_cache.get(
                tr_start, tr_end, self.symbols_data,
                self.spy_prices, getattr(self, "_global_regime_map", None),
            )
            from scripts.walkforward.oos_guard import assert_model_oos
            assert_model_oos(
                trained_through=fold_model.trained_through,
                fold_boundaries=[(tr_start, tr_end, te_start, te_end)],
                purge_days=self._purge_days if self._purge_days is not None else 0,
                model_label=f"swing per-fold@{tr_end}",
                allow_in_sample=getattr(self, "allow_in_sample", False),
                trading_day_set=None,  # swing uses calendar-day purge
            )
            _fold_model = fold_model
        else:
            _fold_model = self.model

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

                feature_names = getattr(_fold_model, "feature_names", None) or []
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
            model=_fold_model,
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
            rebalance_hard_exit_bear=self.rebalance_hard_exit_bear,
            rebalance_flat_stop_pct=self.rebalance_flat_stop_pct,
            # RANKER v2 Spike A — dollar-neutral L/S. When enable_shorts=False
            # (default) these are inert and the long-only path is byte-identical;
            # the values below equal AgentSimulator's own defaults.
            enable_shorts=self.enable_shorts,
            long_gross=self.long_gross,
            short_gross=self.short_gross,
            short_target_n=self.short_target_n,
            short_min_adv=self.short_min_adv,
            short_add_threshold=self.short_add_threshold,
            short_drop_threshold=self.short_drop_threshold,
            # RANKER v2 §3.1 re-architecture — NET-sector cap + SPY beta-hedge overlay.
            net_sector_cap=self.net_sector_cap,
            spy_beta_hedge=self.spy_beta_hedge,
            spy_beta_lookback=self.spy_beta_lookback,
            spy_hedge_max_gross=self.spy_hedge_max_gross,
            # RANKER v2 §3.1 — capture realized net beta/dollar/sector for the L/S
            # book (PURE-ADDITIVE diagnostic; only meaningful when shorts are on).
            # Long-only default leaves capture OFF → byte-identical path.
            capture_net_exposure=self.capture_net_exposure,
            net_beta_lookback=self.net_beta_lookback,
        )
        import uuid as _uuid
        sim._wf_run_id = f"wf-fold{fold_idx}-{_uuid.uuid4().hex[:8]}"
        result = sim.run(
            fold_symbols_data,
            start_date=te_start,
            end_date=te_end,
            spy_prices=self.spy_prices,
            sector_map=self.sector_map or None,
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
        _regime_obs: dict = {}
        regime_sharpes = _crs(equity_curve, te_start, te_end,
                              regime_map=getattr(self, "_global_regime_map", None),
                              obs_counts=_regime_obs)
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
            regime_obs_counts=_regime_obs,
            avg_capital_deployed_pct=getattr(result, "avg_capital_deployed_pct", 0.0),
            deployment_adjusted_sharpe=getattr(result, "deployment_adjusted_sharpe", 0.0),
            low_deployment_warning=getattr(result, "low_deployment_warning", False),
            # RANKER v2 §3.1 — realized net-exposure (PURE-ADDITIVE).
            mean_net_beta=getattr(result, "mean_net_beta", 0.0),
            last_net_beta=getattr(result, "last_net_beta", 0.0),
            max_abs_net_beta=getattr(result, "max_abs_net_beta", 0.0),
            p95_abs_net_beta=getattr(result, "p95_abs_net_beta", 0.0),
            mean_net_dollar=getattr(result, "mean_net_dollar", 0.0),
            max_abs_net_dollar=getattr(result, "max_abs_net_dollar", 0.0),
            max_abs_net_sector=getattr(result, "max_abs_net_sector", 0.0),
            mean_gross=getattr(result, "mean_gross", 0.0),
            net_exposure_captured=getattr(result, "net_exposure_captured", False),
        )
