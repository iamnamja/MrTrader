"""
Tier 3 Agent Simulator — agent-driven historical backtest.

Runs the actual PM / RM / Trader decision logic on historical bars day by day:
  - PM:     FeatureEngineer + model.predict() + CS-normalize → ranked proposals
  - Trader: generate_signal() technical filter (EMA crossover, RSI, trend)
  - RM:     validate_* rule functions from risk_rules.py against live portfolio state
  - Entry:  fill at next-day open (simulates swing limit orders)
  - Exit:   check_exit() trailing-stop logic bar by bar

This is one step above Tier 2 (StrategySimulator) because:
  - Uses actual agent decision code, not just pre-recorded trade P&L
  - PM's ML score + feature engineering runs on historical bars
  - Trader's technical signal is required before entry
  - RM rules gate every entry against portfolio state
"""

import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from app.ml.retrain_config import MAX_WORKERS
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.backtesting.metrics import Trade
from app.backtesting.strategy_simulator import SimResult, STARTING_CAPITAL, TRANSACTION_COST
from app.agents.risk_rules import (
    RiskLimits,
    validate_buying_power,
    validate_position_size,
    validate_sector_concentration,
    validate_daily_loss,
    validate_account_drawdown,
    validate_portfolio_heat,
)
from app.ml.cs_normalize import cs_normalize
from app.strategy.signals import generate_signal, check_exit
from app.strategy.position_sizer import size_position

logger = logging.getLogger(__name__)

# ── Simulation defaults ────────────────────────────────────────────────────────
MIN_CONFIDENCE = 0.40  # LambdaRank scores cluster near 0.5; floor is a sanity check only
TOP_N_STOCKS = 10  # PM proposal count per day
MIN_BARS_SIGNAL = 210  # minimum bars for generate_signal (needs EMA-200)
# ATR multipliers must match training label thresholds (training.py ATR_MULT_TARGET/STOP)
# so the model's learned signal aligns with backtest exit behavior.
ATR_STOP_MULT = 0.5    # 0.5× ATR — matches training label stop
ATR_TARGET_MULT = 1.5  # 1.5× ATR — matches training label target
SWING_STOP_PCT = 0.02  # fallback only when ATR unavailable
SWING_TARGET_PCT = 0.06
TX_COST_PCT = TRANSACTION_COST  # 5bps per side


@dataclass
class _Position:
    """Internal position state tracked during simulation."""
    symbol: str
    entry_date: date
    entry_price: float
    stop_price: float
    target_price: float
    quantity: int
    highest_price: float
    bars_held: int = 0
    confidence: float = 0.0
    sector: str = "UNKNOWN"
    direction: str = "long"  # "long" or "short"


@dataclass
class _PortfolioState:
    """Mutable portfolio state passed through the simulation loop."""
    cash: float
    peak_equity: float
    positions: Dict[str, _Position] = field(default_factory=dict)
    daily_pnl: float = 0.0
    sector_values: Dict[str, float] = field(default_factory=dict)

    @property
    def position_market_value(self) -> float:
        return sum(p.entry_price * p.quantity for p in self.positions.values())

    @property
    def equity(self) -> float:
        return self.cash + self.position_market_value

    @property
    def buying_power(self) -> float:
        return self.cash


class AgentSimulator:
    """
    Portfolio-level backtest that runs the actual PM/Trader/RM decision code
    on historical daily bars.

    Usage:
        from app.backtesting.agent_simulator import AgentSimulator
        from app.ml.features import FeatureEngineer

        sim = AgentSimulator(model=loaded_model)
        result = sim.run(symbols_data, start_date=date(2024,1,1), end_date=date(2025,1,1))
        result.print_report()
    """

    def __init__(
        self,
        model=None,
        starting_capital: float = STARTING_CAPITAL,
        limits: Optional[RiskLimits] = None,
        min_confidence: float = MIN_CONFIDENCE,
        top_n: int = TOP_N_STOCKS,
        transaction_cost_pct: float = TX_COST_PCT,
        atr_stop_mult: float = ATR_STOP_MULT,
        atr_target_mult: float = ATR_TARGET_MULT,
        max_vol_pct: Optional[float] = None,
        regime_bear_max_positions: int = 3,   # Phase 35: cut exposure in bear regime
        vix_fear_threshold: float = 30.0,      # Phase 35: skip new longs when VIX > this
        meta_model=None,                        # Phase 37: Expected-R gate (MetaLabelModel)
        min_expected_r: float = 0.002,          # Phase 37: skip entries where E[R] < this
        pm_abstention_vix: float = 0.0,         # Phase 45 P3-Parallel: abstain if VIX >= this (0=off)
        pm_abstention_spy_ma_days: int = 0,     # Phase 45 P3-Parallel: abstain if SPY < N-day SMA (0=off)
        pm_abstention_spy_5d: bool = False,     # Phase 55: abstain if SPY 5d return <= 0 (negative momentum)
        use_opportunity_score: bool = False,    # Phase 2a: continuous PM opportunity score gate
        no_prefilters: bool = False,             # Phase 3a: bypass RSI/EMA20/50 trader pre-filters
        earnings_blackout: Optional[Dict[str, set]] = None,  # Phase 2b: symbol→{date,...} of earnings
        swing_blackout_days_before: int = 3,     # Phase 2b: skip new entries N days before earnings
        macro_blocked_dates: Optional[set] = None,  # WF-5a: FOMC/NFP/CPI/GDP blocked dates
        benign_blocked_dates: Optional[set] = None,  # P1: dates where regime score < threshold
        regime_score_history: Optional[Dict[date, float]] = None,  # WF-C1: PIT daily regime score
        feature_cache=None,          # FeatureCache: pre-computed raw features (WF speedup)
        sim_scan_interval_days: int = 1,  # score every N days (1=daily, 5=weekly)
        factor_scorer=None,          # Phase D: callable(day, symbols_data, vix_history) -> [(sym, conf)]
        max_hold_bars_override: Optional[int] = None,  # Phase H+: force hold cap (bars) for both legs
    ):
        self.model = model
        self.starting_capital = starting_capital
        self.limits = limits or RiskLimits()
        self.min_confidence = min_confidence
        self.top_n = top_n
        self.transaction_cost_pct = transaction_cost_pct
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.max_vol_pct = max_vol_pct  # Phase 26d: block entries above this vol_percentile_52w
        self.regime_bear_max_positions = regime_bear_max_positions
        self.vix_fear_threshold = vix_fear_threshold
        self.meta_model = meta_model
        self.min_expected_r = min_expected_r
        self.pm_abstention_vix = pm_abstention_vix
        self.pm_abstention_spy_ma_days = pm_abstention_spy_ma_days
        self.pm_abstention_spy_5d = pm_abstention_spy_5d
        self.use_opportunity_score = use_opportunity_score
        self.no_prefilters = no_prefilters
        self.earnings_blackout = earnings_blackout or {}
        self.swing_blackout_days_before = swing_blackout_days_before
        self.macro_blocked_dates: set = macro_blocked_dates or set()
        self.benign_blocked_dates: set = benign_blocked_dates or set()
        self.regime_score_history: Dict[date, float] = regime_score_history or {}
        self.feature_cache = feature_cache
        self.sim_scan_interval_days = max(1, sim_scan_interval_days)
        self.factor_scorer = factor_scorer  # Phase D: optional callable override
        self.max_hold_bars_override = max_hold_bars_override  # Phase H+: PEAD short hold

        # Lazy-load FeatureEngineer (imports may be heavy)
        self._feature_engineer = None
        # Warn once per run if TS norm state is absent (legacy model)
        self._ts_norm_warned = False
        # Pre-built O(1) date lookup maps (built in run())
        self._sym_date_to_row: Dict[str, Dict[date, int]] = {}
        self._sym_date_arr: Dict[str, np.ndarray] = {}

    def run(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        spy_prices: Optional[pd.Series] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> SimResult:
        """
        Run the agent-driven backtest.

        Args:
            symbols_data:  dict of symbol → daily OHLCV DataFrame (>= 210 rows ideally)
            start_date:    first date to trade (defaults to earliest bar + MIN_BARS_SIGNAL)
            end_date:      last date to trade
            spy_prices:    daily SPY close series for benchmark
            sector_map:    symbol → sector string for concentration rules
        """
        if not symbols_data:
            return self._empty_result()

        sector_map = sector_map or {}

        # Collect all trading days across all symbols
        all_days = sorted({
            d.date() if hasattr(d, "date") else d
            for df in symbols_data.values()
            for d in df.index
        })

        if not all_days:
            return self._empty_result()

        start_date = start_date or all_days[0]
        end_date = end_date or all_days[-1]
        trading_days = [d for d in all_days if start_date <= d <= end_date]

        if not trading_days:
            return self._empty_result()

        # Build O(1) date-to-row index per symbol (used by _bars_up_to / _bars_on).
        self._sym_date_to_row = {}
        self._sym_date_arr = {}
        for sym, df in symbols_data.items():
            dates = np.array([d.date() if hasattr(d, "date") else d for d in df.index])
            self._sym_date_arr[sym] = dates
            self._sym_date_to_row[sym] = {d: i for i, d in enumerate(dates)}

        portfolio = _PortfolioState(
            cash=self.starting_capital,
            peak_equity=self.starting_capital,
        )

        accepted_trades: List[Trade] = []
        equity_by_date: Dict[date, float] = {}
        tx_costs_total = 0.0

        # Phase 35: Build SPY close series and VIX series for regime gate
        spy_df = symbols_data.get("SPY")
        _spy_closes: Optional[pd.Series] = None
        if spy_df is not None and "close" in spy_df.columns:
            _spy_closes = spy_df["close"]
        elif spy_prices is not None:
            _spy_closes = spy_prices

        _vix_closes: Optional[pd.Series] = None
        _vix_df = symbols_data.get("^VIX")
        if _vix_df is None:
            _vix_df = symbols_data.get("VIX")
        if _vix_df is not None and "close" in _vix_df.columns:
            _vix_closes = _vix_df["close"]

        for day_idx, day in enumerate(trading_days):
            # 1. Advance bars_held for all open positions, mark equity
            for pos in portfolio.positions.values():
                pos.bars_held += 1

            # 2. Exit check: scan today's bar for stop/target hits
            closed = self._process_exits(day, symbols_data, portfolio)
            for trade, tx_cost in closed:
                accepted_trades.append(trade)
                tx_costs_total += tx_cost
            if portfolio.equity > portfolio.peak_equity:
                portfolio.peak_equity = portfolio.equity

            # 3. PM: score all symbols using bars up to yesterday.
            # When sim_scan_interval_days > 1, skip scoring on off-days
            # (exits still run daily). Proposals from last scan day are reused.
            _scan_day = (day_idx % self.sim_scan_interval_days == 0)
            if _scan_day:
                proposals = self._pm_score(day, symbols_data, vix_history=_vix_closes)
            # else: proposals unchanged from previous scan day

            # 4. Phase 35: Market regime gate — cut exposure in bear/fear regimes
            # Factor portfolio L/S mode: scorer manages its own regime logic (longs suppressed
            # in bear market, everything blocked at VIX >= 40). Skip simulator-level gates.
            _skip_entries = False
            _max_pos_today = self.limits.MAX_OPEN_POSITIONS
            if self.factor_scorer is None:
                if _spy_closes is not None:
                    try:
                        spy_idx = _spy_closes.index
                        spy_dates = spy_idx.date if hasattr(spy_idx, 'date') else pd.DatetimeIndex(spy_idx).date
                        spy_hist = _spy_closes.loc[spy_dates <= day]
                        if len(spy_hist) >= 200:
                            spy_ema200 = float(spy_hist.ewm(span=200, adjust=False).mean().iloc[-1])
                            spy_close = float(spy_hist.iloc[-1])
                            if spy_close < spy_ema200:
                                _max_pos_today = self.regime_bear_max_positions
                                logger.debug("Bear regime on %s: SPY %.2f < EMA200 %.2f — max_pos=%d",
                                             day, spy_close, spy_ema200, _max_pos_today)
                    except Exception:
                        pass
                if _vix_closes is not None:
                    try:
                        vix_idx = _vix_closes.index
                        vix_dates = vix_idx.date if hasattr(vix_idx, 'date') else pd.DatetimeIndex(vix_idx).date
                        vix_today = _vix_closes.loc[vix_dates <= day]
                        if len(vix_today) > 0:
                            vix_val = float(vix_today.iloc[-1])
                            if vix_val > self.vix_fear_threshold:
                                _skip_entries = True
                                logger.debug("Fear spike on %s: VIX %.1f > %.1f — skipping new entries",
                                             day, vix_val, self.vix_fear_threshold)
                    except Exception:
                        pass

            # Phase 45 P3-Parallel: PM abstention gate (VIX >= threshold OR SPY < N-day SMA)
            if not _skip_entries and (self.pm_abstention_vix > 0
                                      or self.pm_abstention_spy_ma_days > 0
                                      or self.pm_abstention_spy_5d):
                try:
                    if self.pm_abstention_vix > 0 and _vix_closes is not None:
                        vix_idx = _vix_closes.index
                        vix_dates = vix_idx.date if hasattr(vix_idx, 'date') else pd.DatetimeIndex(vix_idx).date
                        vix_today = _vix_closes.loc[vix_dates <= day]
                        if len(vix_today) > 0 and float(vix_today.iloc[-1]) >= self.pm_abstention_vix:
                            _skip_entries = True
                            logger.debug("PM abstention (VIX) on %s: %.1f >= %.1f",
                                         day, float(vix_today.iloc[-1]), self.pm_abstention_vix)
                    if not _skip_entries and self.pm_abstention_spy_ma_days > 0 and _spy_closes is not None:
                        spy_idx = _spy_closes.index
                        spy_dates = spy_idx.date if hasattr(spy_idx, 'date') else pd.DatetimeIndex(spy_idx).date
                        spy_hist = _spy_closes.loc[spy_dates <= day]
                        if len(spy_hist) >= self.pm_abstention_spy_ma_days:
                            spy_ma = float(spy_hist.tail(self.pm_abstention_spy_ma_days).mean())
                            if float(spy_hist.iloc[-1]) < spy_ma:
                                _skip_entries = True
                                logger.debug("PM abstention (SPY MA%d) on %s: %.2f < %.2f",
                                             self.pm_abstention_spy_ma_days, day,
                                             float(spy_hist.iloc[-1]), spy_ma)
                    if not _skip_entries and self.pm_abstention_spy_5d and _spy_closes is not None:
                        spy_idx = _spy_closes.index
                        spy_dates = spy_idx.date if hasattr(spy_idx, 'date') else pd.DatetimeIndex(spy_idx).date
                        spy_hist = _spy_closes.loc[spy_dates <= day]
                        if len(spy_hist) >= 6:
                            spy_5d_ret = float(spy_hist.iloc[-1]) / float(spy_hist.iloc[-6]) - 1.0
                            if spy_5d_ret <= 0:
                                _skip_entries = True
                                logger.debug("PM abstention (SPY 5d) on %s: 5d ret=%.3f",
                                             day, spy_5d_ret)
                except Exception:
                    pass

            # Phase 2a: continuous PM opportunity score gate (same formula as live PM)
            if not _skip_entries and self.use_opportunity_score and _spy_closes is not None:
                try:
                    spy_idx = _spy_closes.index
                    spy_dates = spy_idx.date if hasattr(spy_idx, 'date') else pd.DatetimeIndex(spy_idx).date
                    spy_hist = _spy_closes.loc[spy_dates <= day]
                    if len(spy_hist) >= 20:
                        spy_close = float(spy_hist.iloc[-1])
                        spy_ma20 = float(spy_hist.tail(20).mean())
                        spy_5d_ret = (spy_close / float(spy_hist.iloc[-6]) - 1.0) if len(spy_hist) >= 6 else 0.0
                        ma_score = 1.0 if spy_close >= spy_ma20 else 0.4
                        mom_score = float(np.clip(0.5 + spy_5d_ret * 25.0, 0.0, 1.0))
                        vix_score, vix_trend = 1.0, 1.0
                        if _vix_closes is not None:
                            vix_idx = _vix_closes.index
                            vix_dates_v = vix_idx.date if hasattr(vix_idx, 'date') else pd.DatetimeIndex(vix_idx).date
                            vix_hist = _vix_closes.loc[vix_dates_v <= day]
                            if len(vix_hist) > 0:
                                vix_level = float(vix_hist.iloc[-1])
                                vix_score = float(np.clip(1.0 - (vix_level - 15.0) / 20.0, 0.0, 1.0))
                                vix_5d_avg = float(vix_hist.tail(5).mean()) if len(vix_hist) >= 5 else vix_level
                                vix_trend = float(np.clip(1.0 - (vix_level - vix_5d_avg) / 5.0, 0.0, 1.0))
                        # Phase 5b sync: use config-driven weights (matches live PM).
                        # breadth/dispersion unavailable in historical simulation → weight=0,
                        # renormalize remaining 4 components (same logic as live PM).
                        from app.config import settings as _s
                        _w_vix = _s.opp_score_vix_weight
                        _w_vt = _s.opp_score_vix_trend_weight
                        _w_ma = _s.opp_score_ma_weight
                        _w_mom = _s.opp_score_mom_weight
                        _w_total = _w_vix + _w_vt + _w_ma + _w_mom or 1.0
                        opp_score = (
                            _w_vix * vix_score + _w_vt * vix_trend
                            + _w_ma * ma_score + _w_mom * mom_score
                        ) / _w_total
                        if opp_score < 0.35:
                            _skip_entries = True
                            logger.debug("Opp score %.2f < 0.35 on %s — skipping entries", opp_score, day)
                        elif opp_score < 0.65:
                            _max_pos_today = min(_max_pos_today, 2)
                            logger.debug("Opp score %.2f on %s — capping candidates at 2", opp_score, day)
                except Exception:
                    pass

            # WF-5a: macro event gate — block new entries on FOMC/NFP/CPI/GDP days
            if not _skip_entries and self.macro_blocked_dates and day in self.macro_blocked_dates:
                _skip_entries = True
                logger.debug("Macro gate blocked entries on %s", day)

            # P1 BenignGate: block new entries on days with adverse macro regime score
            if not _skip_entries and self.benign_blocked_dates and day in self.benign_blocked_dates:
                _skip_entries = True
                logger.debug("BenignGate blocked entries on %s (adverse regime)", day)

            # 5. Trader signal + RM rules + entry
            if proposals and not _skip_entries:
                new_trades, new_tx = self._process_entries(
                    day, proposals, symbols_data, portfolio, sector_map,
                    max_positions=_max_pos_today, vix_history=_vix_closes,
                )
                accepted_trades.extend(new_trades)
                tx_costs_total += new_tx

            equity_by_date[day] = portfolio.equity
            portfolio.daily_pnl = 0.0  # reset for next day

        # Force-close any remaining open positions at last bar close
        for sym, pos in list(portfolio.positions.items()):
            df = symbols_data.get(sym)
            if df is None or len(df) == 0:
                continue
            exit_price = float(df["close"].iloc[-1])
            trade, tx = self._close_position(pos, end_date, exit_price, "FORCE_CLOSE", portfolio)
            accepted_trades.append(trade)
            tx_costs_total += tx

        if not accepted_trades:
            return self._empty_result()

        return self._compute_result(
            accepted_trades, equity_by_date, tx_costs_total,
            start_date, end_date, spy_prices,
        )

    # ─── PM: score all symbols ─────────────────────────────────────────────────

    def _pm_score(
        self, day: date, symbols_data: Dict[str, pd.DataFrame],
        vix_history: Optional["pd.Series"] = None,
    ) -> List[Tuple[str, float]]:
        """Return list of (symbol, confidence) sorted by confidence desc.

        When a FeatureCache is available, dispatches to _pm_score_cached for
        O(1) feature lookup instead of re-computing features per symbol.
        When factor_scorer is set, delegates entirely to the callable.
        """
        # Phase D: factor portfolio path — bypasses ML model entirely
        if self.factor_scorer is not None:
            try:
                return self.factor_scorer(day, symbols_data, vix_history)
            except Exception as _fsc_exc:
                logger.warning("factor_scorer failed on %s: %s", day, _fsc_exc)
                return []

        if self.model is None or not getattr(self.model, "is_trained", False):
            return []

        if self.feature_cache is not None and self.feature_cache.n_symbols > 0:
            return self._pm_score_cached(day, vix_history)

        fe = self._get_feature_engineer()
        features_by_symbol: Dict[str, dict] = {}
        _regime_score = self.regime_score_history.get(day, 0.5)

        def _compute_feats(sym_df_pair):
            sym, df = sym_df_pair
            bars_to_yesterday = self._bars_up_to(df, day, exclude_today=True)
            if bars_to_yesterday is None or len(bars_to_yesterday) < 60:
                return sym, None
            try:
                feats = fe.engineer_features(
                    sym, bars_to_yesterday, fetch_fundamentals=False,
                    as_of_date=day, regime_score=_regime_score,
                    vix_history=vix_history,
                )
                return sym, feats
            except Exception:
                return sym, None

        _n_workers = min(os.cpu_count() or 4, MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=_n_workers) as pool:
            for sym, feats in pool.map(_compute_feats, symbols_data.items()):
                if feats is not None:
                    features_by_symbol[sym] = feats

        if not features_by_symbol:
            return []

        sym_list = list(features_by_symbol.keys())
        model_feat_names = getattr(self.model, "feature_names", None)
        try:
            if model_feat_names:
                X = np.array([
                    [features_by_symbol[s].get(f, 0.0) for f in model_feat_names]
                    for s in sym_list
                ])
            else:
                X = np.array([list(features_by_symbol[s].values()) for s in sym_list])
            X = np.nan_to_num(X, nan=0.0)
            X = self._normalize_for_inference(X, sym_list, day)
            vix_now = self._vix_at(vix_history, day)
            _, probas = self.model.predict_with_vix(X, vix_level=vix_now)
        except Exception as exc:
            logger.warning("PM score failed on %s (%s): %s", day, type(exc).__name__, exc)
            return []

        ranked = sorted(zip(sym_list, probas), key=lambda x: x[1], reverse=True)
        proposals = []
        for sym, prob in ranked:
            if float(prob) < self.min_confidence:
                continue
            # Phase 26d: vol filter — skip stocks in top vol_percentile_52w bucket
            if self.max_vol_pct is not None:
                vol_pct = features_by_symbol[sym].get("vol_percentile_52w", 0.0)
                if vol_pct > self.max_vol_pct / 100.0:
                    continue
            proposals.append((sym, float(prob)))
            if len(proposals) >= self.top_n:
                break
        return proposals

    def _pm_score_cached(
        self, day: date, vix_history: Optional["pd.Series"] = None,
    ) -> List[Tuple[str, float]]:
        """Cache-backed PM scoring: O(1) feature lookup per symbol per day.

        Replaces the engineer_features() call with a numpy array row lookup,
        eliminating the per-day per-symbol feature recomputation bottleneck.
        """
        cache = self.feature_cache
        model_feat_names = getattr(self.model, "feature_names", None) or cache.feature_names

        # Gather cached rows for symbols with data on this day
        sym_list = []
        rows = []
        vol_pcts = []
        vol_col_idx = None
        if "vol_percentile_52w" in cache.feature_names and self.max_vol_pct is not None:
            vol_col_idx = cache.feature_names.index("vol_percentile_52w")

        for sym, idx_map in cache.date_index.items():
            row_idx = idx_map.get(day)
            if row_idx is None:
                continue
            raw_row = cache.matrix[sym][row_idx]
            # Reorder to model feature order if cache order differs
            if model_feat_names is not cache.feature_names:
                row = np.array([
                    float(raw_row[cache.feature_names.index(f)])
                    if f in cache.feature_names else 0.0
                    for f in model_feat_names
                ], dtype=np.float32)
            else:
                row = raw_row
            sym_list.append(sym)
            rows.append(row)
            vol_pcts.append(float(raw_row[vol_col_idx]) if vol_col_idx is not None else 0.0)

        if not sym_list:
            return []

        try:
            X = np.nan_to_num(np.vstack(rows), nan=0.0)
            sym_arr = np.array(sym_list)
            X = self._normalize_for_inference(X, sym_arr, day)
            vix_now = self._vix_at(vix_history, day)
            _, probas = self.model.predict_with_vix(X, vix_level=vix_now)
        except Exception as exc:
            logger.warning("PM score (cached) failed on %s (%s): %s", day, type(exc).__name__, exc)
            return []

        proposals = []
        for i, (sym, prob) in enumerate(
            sorted(zip(sym_list, probas), key=lambda x: x[1], reverse=True)
        ):
            if float(prob) < self.min_confidence:
                continue
            if self.max_vol_pct is not None:
                if vol_pcts[sym_list.index(sym)] > self.max_vol_pct / 100.0:
                    continue
            proposals.append((sym, float(prob)))
            if len(proposals) >= self.top_n:
                break
        return proposals

    # ─── Inference helpers ─────────────────────────────────────────────────────

    def _normalize_for_inference(
        self, X: np.ndarray, symbols: List[str], day: date
    ) -> np.ndarray:
        """Mirror live PM normalization: TS norm for v185+ models, cs_normalize fallback.

        Key divergence from live PM: window_id = day.toordinal() (one per sim day)
        rather than date.today().toordinal(). This is intentional — in the sim each
        historical day accumulates its own per-symbol trailing history, whereas live
        PM always processes today as window N.
        """
        ts_state = getattr(self.model, "_ts_norm_state", None)
        if ts_state is not None:
            try:
                from app.ml.ts_normalize import transform as _ts_transform
                window_id = day.toordinal()
                X_norm, _ = _ts_transform(X, symbols, [window_id] * len(symbols), ts_state)
                return X_norm
            except Exception as exc:
                logger.warning("TS normalize failed on %s, falling back to cs_normalize: %s", day, exc)
        else:
            if not self._ts_norm_warned:
                logger.info(
                    "Model has no TS norm state — using cs_normalize (legacy pre-v185 path)"
                )
                self._ts_norm_warned = True
        return cs_normalize(X)

    def _vix_at(self, vix_history: Optional["pd.Series"], day: date) -> Optional[float]:
        """Return last available VIX close on or before `day`. None if unavailable."""
        if vix_history is None or vix_history.empty:
            return None
        try:
            idx = vix_history.index
            idx_dates = [i.date() if hasattr(i, "date") else i for i in idx]
            candidates = [(d, v) for d, v in zip(idx_dates, vix_history.values) if d <= day]
            if candidates:
                return float(candidates[-1][1])
        except Exception:
            pass
        return None

    # ─── Trader: technical signal gate ────────────────────────────────────────

    def _trader_signal(
        self, symbol: str, bars_up_to_day: pd.DataFrame
    ) -> Tuple[bool, float, float]:
        """
        Compute entry clearance and stop/target levels from historical bars.

        In the live system Trader polls generate_signal() every 5 minutes and
        enters on the EMA crossover signal during the session. In a daily
        backtest that's too restrictive — the crossover may fire intraday but
        not at bar close. Instead, PM's ML score is the primary entry gate;
        here we use generate_signal() only to compute ATR-based stop/target
        levels, and require a basic trend filter (price > EMA-200) rather than
        an exact crossover.
        """
        if bars_up_to_day is None or len(bars_up_to_day) < 200:
            return False, 0.0, 0.0
        try:
            generate_signal(
                symbol, bars_up_to_day,
                check_earnings=False,
                check_regime=False,
                backtest_mode=True,
            )
            closes = bars_up_to_day["close"]
            close = float(closes.iloc[-1])

            # Long-term trend: price above EMA-200
            ema200 = float(closes.ewm(span=200, adjust=False).mean().iloc[-1])
            if not self.no_prefilters and close <= ema200:
                return False, 0.0, 0.0

            # Near-term trend: price above EMA-20 and EMA-50
            ema20 = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
            if not self.no_prefilters and (close <= ema20 or close <= ema50):
                return False, 0.0, 0.0

            # RSI zone: not overbought, not in freefall (40–70)
            prices_arr = closes.to_numpy(dtype=float)
            deltas = np.diff(prices_arr[-16:])
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)
            avg_gain = float(np.mean(gains[-14:])) if len(gains) >= 14 else float(np.mean(gains))
            avg_loss = float(np.mean(losses[-14:])) if len(losses) >= 14 else float(np.mean(losses))
            rsi = 100.0 - (100.0 / (1.0 + avg_gain / max(avg_loss, 1e-8)))
            if not self.no_prefilters and not (40.0 <= rsi <= 70.0):
                return False, 0.0, 0.0

            # Volume confirmation: today's volume >= 80% of 20-day avg
            if "volume" in bars_up_to_day.columns:
                vol_today = float(bars_up_to_day["volume"].iloc[-1])
                vol_avg20 = float(bars_up_to_day["volume"].iloc[-20:].mean())
                if vol_avg20 > 0 and vol_today < 0.8 * vol_avg20:
                    return False, 0.0, 0.0

            # Phase 39: Setup-aware stops — anchor to structure, not just ATR%
            # Momentum (EMA crossover): stop below recent pivot low (10-bar)
            # Pullback (RSI dip): stop below EMA20 (natural support)
            close = float(bars_up_to_day["close"].iloc[-1])
            try:
                high = bars_up_to_day["high"].to_numpy(dtype=float)
                low = bars_up_to_day["low"].to_numpy(dtype=float)
                cl = bars_up_to_day["close"].to_numpy(dtype=float)
                tr = np.maximum(
                    high[1:] - low[1:],
                    np.maximum(abs(high[1:] - cl[:-1]), abs(low[1:] - cl[:-1])))
                atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
                atr_pct = atr / close

                ema_fast_v = float(bars_up_to_day["close"].ewm(span=20, adjust=False).mean().iloc[-1])
                ema_fast_p = float(bars_up_to_day["close"].ewm(span=20, adjust=False).mean().iloc[-2])
                ema_slow_v = float(bars_up_to_day["close"].ewm(span=50, adjust=False).mean().iloc[-1])
                ema_slow_p = float(bars_up_to_day["close"].ewm(span=50, adjust=False).mean().iloc[-2])
                is_ema_crossover = ema_fast_v > ema_slow_v and ema_fast_p <= ema_slow_p

                if is_ema_crossover:
                    # Momentum: stop below 10-bar pivot low (breakout level)
                    pivot_low = float(np.min(low[-11:-1])) if len(low) >= 11 else close * 0.97
                    stop_price = max(pivot_low * 0.995, close * (1 - 0.08))
                else:
                    # Pullback/RSI-dip: stop below EMA20 with small buffer
                    stop_price = max(ema20 * 0.995, close * (1 - 0.06))

                # Target: 1.5× ATR above entry (fixed — matches training labels)
                target_pct = float(np.clip(self.atr_target_mult * atr_pct, 0.01, 0.16))
                target_price = close * (1 + target_pct)
            except Exception:
                stop_price = close * (1 - SWING_STOP_PCT)
                target_price = close * (1 + SWING_TARGET_PCT)
            return True, stop_price, target_price
        except Exception as exc:
            logger.debug("generate_signal failed %s: %s", symbol, exc)
            return False, 0.0, 0.0

    # ─── RM: validate against portfolio state ──────────────────────────────────

    def _rm_validate(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        quantity: int,
        portfolio: _PortfolioState,
        sector: str,
    ) -> Tuple[bool, str]:
        """Run key RM rules against current portfolio state. Returns (ok, reason)."""
        trade_cost = entry_price * quantity
        equity = portfolio.equity

        ok, msg = validate_buying_power(trade_cost, portfolio.buying_power, self.limits)
        if not ok:
            return False, msg

        ok, msg = validate_position_size(trade_cost, equity, self.limits)
        if not ok:
            return False, msg

        sector_val = portfolio.sector_values.get(sector, 0.0)
        ok, msg = validate_sector_concentration(trade_cost, sector_val, equity, sector, self.limits)
        if not ok:
            return False, msg

        ok, msg = validate_daily_loss(portfolio.daily_pnl, equity, self.limits)
        if not ok:
            return False, msg

        # open-position cap is enforced in _process_entries (regime-adjusted)
        ok, msg = validate_account_drawdown(equity, portfolio.peak_equity, self.limits)
        if not ok:
            return False, msg

        # Portfolio heat: represent open positions as list of dicts for validate_portfolio_heat
        open_pos_dicts = [
            {"stop_price": p.stop_price, "entry_price": p.entry_price, "quantity": p.quantity}
            for p in portfolio.positions.values()
        ]
        new_trade_risk = (entry_price - stop_price) * quantity
        ok, msg = validate_portfolio_heat(new_trade_risk, open_pos_dicts, equity, self.limits)
        if not ok:
            return False, msg

        return True, "ok"

    # ─── Entry simulation ──────────────────────────────────────────────────────

    def _process_entries(
        self,
        day: date,
        proposals: List[Tuple[str, float]],
        symbols_data: Dict[str, pd.DataFrame],
        portfolio: _PortfolioState,
        sector_map: Dict[str, str],
        max_positions: Optional[int] = None,
        vix_history: Optional["pd.Series"] = None,
    ) -> Tuple[List[Trade], float]:
        entered_trades: List[Trade] = []
        tx_costs_total = 0.0
        _max_pos = max_positions if max_positions is not None else self.limits.MAX_OPEN_POSITIONS

        # Entry price: today's open (PM runs premarket; Trader executes post-open)
        for proposal in proposals:
            # Support both 2-tuple (sym, conf) legacy and 3-tuple (sym, conf, direction)
            if len(proposal) == 3:
                sym, confidence, direction = proposal
            else:
                sym, confidence = proposal
                direction = "long"
            is_short = direction == "short"
            if len(portfolio.positions) >= _max_pos:
                break  # Phase 35: respect regime-adjusted position cap

            if sym in portfolio.positions:
                continue  # already holding

            # Phase 2b: earnings blackout — skip if within N days of earnings
            if self.earnings_blackout and sym in self.earnings_blackout:
                _in_blackout = False
                for _e_date in self.earnings_blackout[sym]:
                    _delta = (_e_date - day).days
                    if 0 <= _delta <= self.swing_blackout_days_before:
                        _in_blackout = True
                        break
                if _in_blackout:
                    continue

            df = symbols_data.get(sym)
            if df is None:
                continue

            bars_yesterday = self._bars_up_to(df, day, exclude_today=True)
            today_bar = self._bars_on(df, day)
            if today_bar is None:
                continue

            entry_price = float(today_bar["open"])
            if entry_price <= 0:
                continue

            # Phase 34: No-chase filter — skip large overnight gaps and extended entries
            # Not applied in factor portfolio mode (monthly rebalance enters regardless of daily gap)
            if self.factor_scorer is None and bars_yesterday is not None and len(bars_yesterday) >= 14:
                try:
                    prior_close = float(bars_yesterday["close"].iloc[-1])
                    hi = bars_yesterday["high"].to_numpy(dtype=float)[-15:]
                    lo = bars_yesterday["low"].to_numpy(dtype=float)[-15:]
                    cl = bars_yesterday["close"].to_numpy(dtype=float)[-15:]
                    tr = np.maximum(hi[1:] - lo[1:],
                                    np.maximum(abs(hi[1:] - cl[:-1]), abs(lo[1:] - cl[:-1])))
                    atr_pct = float(np.mean(tr[-14:])) / max(prior_close, 1e-6)
                    if entry_price > prior_close * (1 + 0.75 * atr_pct):
                        continue  # skip overnight chase
                    ema20 = float(bars_yesterday["close"].ewm(span=20, adjust=False).mean().iloc[-1])
                    if entry_price > ema20 * (1 + 1.5 * atr_pct):
                        continue  # skip too-extended entry
                except Exception:
                    pass

            # Phase 37: Meta-label Expected-R gate
            if self.meta_model is not None:
                try:
                    # Use cache if available, else re-compute
                    if self.feature_cache is not None:
                        feats = self.feature_cache.get_features(sym, day)
                    else:
                        fe = self._get_feature_engineer()
                        feats = fe.engineer_features(sym, bars_yesterday,
                                                     fetch_fundamentals=False, as_of_date=day,
                                                     vix_history=vix_history)
                    if feats is not None and not self.meta_model.should_enter(feats):
                        continue
                except Exception:
                    pass

            # Trader: technical signal gate (longs only; shorts bypass technical filter)
            if not is_short:
                if self.factor_scorer is not None:
                    # Factor portfolio mode: monthly rebalance, no active stops.
                    # Wide stop (20%) acts as circuit-breaker only; exit via max_hold_bars.
                    stop_price = entry_price * (1 - 0.20)
                    target_price = entry_price * 2.0  # effectively never fires
                else:
                    should_enter, stop_price, target_price = self._trader_signal(sym, bars_yesterday)
                    if not should_enter:
                        continue
                    # Use signal stops if valid, else fallback percentages
                    if stop_price <= 0 or stop_price >= entry_price:
                        stop_price = entry_price * (1 - SWING_STOP_PCT)
                    if target_price <= entry_price:
                        target_price = entry_price * (1 + SWING_TARGET_PCT)
            else:
                if self.factor_scorer is not None:
                    # Factor portfolio short: same monthly rebalance model — wide stops
                    stop_price = entry_price * (1 + 0.20)   # 20% circuit-breaker above entry
                    target_price = entry_price * 0.50       # effectively never fires
                else:
                    # Short: stop is above entry, target is below entry
                    stop_price = entry_price * (1 + SWING_STOP_PCT)
                    target_price = entry_price * (1 - SWING_TARGET_PCT)

            # Position sizing — use abs(confidence) for shorts
            conf_for_sizing = abs(confidence)
            quantity = size_position(
                account_equity=portfolio.equity,
                available_cash=portfolio.cash,
                entry_price=entry_price,
                stop_price=stop_price if not is_short else entry_price * (1 - SWING_STOP_PCT),
                ml_score=conf_for_sizing,
            )
            # Apply RM position-size cap so the trade doesn't auto-reject
            max_position_dollars = portfolio.equity * self.limits.MAX_POSITION_SIZE_PCT
            quantity = min(quantity, max(1, int(max_position_dollars / entry_price)))
            if quantity <= 0:
                continue

            sector = sector_map.get(sym, "UNKNOWN")
            ok, reason = self._rm_validate(sym, entry_price, stop_price, quantity, portfolio, sector)
            if not ok:
                logger.debug("RM rejected %s on %s: %s", sym, day, reason)
                continue

            trade_cost = entry_price * quantity
            tx_cost = trade_cost * self.transaction_cost_pct  # entry side

            if not is_short:
                portfolio.cash -= trade_cost + tx_cost
            else:
                # Short: receive proceeds; deduct tx cost and post margin
                portfolio.cash += trade_cost - tx_cost
                portfolio.cash -= trade_cost  # margin held = notional value

            portfolio.positions[sym] = _Position(
                symbol=sym,
                entry_date=day,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                quantity=quantity,
                highest_price=entry_price,
                confidence=confidence,
                sector=sector,
                direction=direction,
            )
            portfolio.sector_values[sector] = (
                portfolio.sector_values.get(sector, 0.0) + trade_cost
            )
            tx_costs_total += tx_cost

        return entered_trades, tx_costs_total

    # ─── Exit simulation ───────────────────────────────────────────────────────

    def _process_exits(
        self,
        day: date,
        symbols_data: Dict[str, pd.DataFrame],
        portfolio: _PortfolioState,
    ) -> List[Tuple[Trade, float]]:
        closed: List[Tuple[Trade, float]] = []

        for sym in list(portfolio.positions.keys()):
            pos = portfolio.positions[sym]
            df = symbols_data.get(sym)
            if df is None:
                continue

            today_bar = self._bars_on(df, day)
            if today_bar is None:
                continue

            today_high = float(today_bar["high"])
            today_low = float(today_bar["low"])
            today_close = float(today_bar["close"])
            is_short = getattr(pos, "direction", "long") == "short"

            # Daily borrow cost for short positions (0.5% annualised)
            if is_short:
                borrow_cost = pos.entry_price * pos.quantity * 0.005 / 252
                portfolio.cash -= borrow_cost
                portfolio.daily_pnl -= borrow_cost

            pos.highest_price = max(pos.highest_price, today_high)

            if not is_short:
                # Long: use standard check_exit with trailing stop
                should_exit, exit_reason, new_stop = check_exit(
                    symbol=sym,
                    current_price=today_close,
                    entry_price=pos.entry_price,
                    stop_price=pos.stop_price,
                    target_price=pos.target_price,
                    highest_price=pos.highest_price,
                    bars_held=pos.bars_held,
                    min_hold_bars=1,
                    max_hold_bars=(self.max_hold_bars_override
                                   if self.max_hold_bars_override is not None
                                   else self.limits.MAX_OPEN_POSITIONS * 4),
                )
                pos.stop_price = new_stop
                # Intrabar stop/target override
                if not should_exit:
                    if today_low <= pos.stop_price:
                        should_exit = True
                        exit_reason = "stop_hit"
                        today_close = pos.stop_price
                    elif today_high >= pos.target_price:
                        should_exit = True
                        exit_reason = "target_hit"
                        today_close = pos.target_price
            else:
                # Short: stop is above entry (upside), target is below entry (downside)
                should_exit = False
                exit_reason = ""
                if today_high >= pos.stop_price:
                    should_exit = True
                    exit_reason = "stop_hit"
                    today_close = pos.stop_price
                elif today_low <= pos.target_price:
                    should_exit = True
                    exit_reason = "target_hit"
                    today_close = pos.target_price
                elif pos.bars_held >= (self.max_hold_bars_override
                                       if self.max_hold_bars_override is not None
                                       else self.limits.MAX_OPEN_POSITIONS * 4):
                    should_exit = True
                    exit_reason = "max_hold"

            if should_exit:
                trade, tx = self._close_position(pos, day, today_close, exit_reason, portfolio)
                closed.append((trade, tx))
                del portfolio.positions[sym]

        return closed

    def _close_position(
        self,
        pos: _Position,
        exit_date: date,
        exit_price: float,
        reason: str,
        portfolio: Optional[_PortfolioState] = None,
    ) -> Tuple[Trade, float]:
        is_short = getattr(pos, "direction", "long") == "short"
        tx_cost = exit_price * pos.quantity * self.transaction_cost_pct

        if not is_short:
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
            net_pnl = gross_pnl - tx_cost
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
            if portfolio is not None:
                portfolio.cash += exit_price * pos.quantity - tx_cost
        else:
            # Short: profit when price falls (entry - exit)
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity
            net_pnl = gross_pnl - tx_cost
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price
            if portfolio is not None:
                # Return margin + realised P&L; deduct buy-to-cover cost
                portfolio.cash += pos.entry_price * pos.quantity + gross_pnl - tx_cost

        if portfolio is not None:
            portfolio.daily_pnl += net_pnl
            sector = getattr(pos, "sector", "UNKNOWN")
            cost_basis = pos.entry_price * pos.quantity
            portfolio.sector_values[sector] = max(
                0.0, portfolio.sector_values.get(sector, 0.0) - cost_basis
            )

        trade = Trade(
            symbol=pos.symbol,
            entry_date=pos.entry_date,
            exit_date=exit_date,
            entry_price=round(pos.entry_price, 4),
            exit_price=round(exit_price, 4),
            quantity=pos.quantity,
            pnl=round(net_pnl, 2),
            pnl_pct=round(pnl_pct, 6),
            hold_bars=pos.bars_held,
            exit_reason=self._normalize_reason(reason),
            trade_type="swing",
        )
        return trade, tx_cost

    # ─── Metrics ───────────────────────────────────────────────────────────────

    def _compute_result(
        self,
        accepted_trades: List[Trade],
        equity_by_date: Dict[date, float],
        tx_costs_total: float,
        start_date: date,
        end_date: date,
        spy_prices: Optional[pd.Series],
    ) -> SimResult:
        # Rebuild equity from trades since portfolio.cash tracking can drift
        # (simpler: use equity_by_date as-is)
        equity_curve = sorted(equity_by_date.items())
        eq_vals = [v for _, v in equity_curve]
        final_equity = eq_vals[-1] if eq_vals else self.starting_capital

        total_return = (final_equity - self.starting_capital) / self.starting_capital
        n_days = max((end_date - start_date).days, 1)
        ann_return = (1 + total_return) ** (365 / n_days) - 1

        daily_rets = [
            (eq_vals[i] - eq_vals[i-1]) / max(eq_vals[i-1], 1e-9)
            for i in range(1, len(eq_vals))
        ]
        ret_series = daily_rets if len(daily_rets) >= 2 else [t.pnl_pct for t in accepted_trades]

        from app.backtesting.strategy_simulator import StrategySimulator
        sharpe = StrategySimulator._sharpe(ret_series, 252)
        sortino = StrategySimulator._sortino(ret_series, 252)
        max_dd = StrategySimulator._max_drawdown(eq_vals)
        calmar = ann_return / max(max_dd, 1e-9)

        winners = [t for t in accepted_trades if t.pnl_pct > 0]
        losers = [t for t in accepted_trades if t.pnl_pct <= 0]
        win_rate = len(winners) / max(len(accepted_trades), 1)
        avg_pnl = sum(t.pnl_pct for t in accepted_trades) / max(len(accepted_trades), 1)
        avg_hold = sum(t.hold_bars for t in accepted_trades) / max(len(accepted_trades), 1)
        gross_win = sum(t.pnl_pct for t in winners) if winners else 0.0
        gross_loss = max(abs(sum(t.pnl_pct for t in losers)), 1e-9)
        profit_factor = gross_win / gross_loss

        exit_breakdown = defaultdict(int)
        for t in accepted_trades:
            exit_breakdown[t.exit_reason] += 1

        monthly: Dict[str, float] = defaultdict(float)
        for t in accepted_trades:
            monthly[t.entry_date.strftime("%Y-%m")] += t.pnl

        # Benchmark
        benchmark_return = alpha = info_ratio = 0.0
        if spy_prices is not None and len(spy_prices) > 0:
            try:
                spy = spy_prices
                s0 = spy.asof(pd.Timestamp(start_date))
                s1 = spy.asof(pd.Timestamp(end_date))
                if s0 and s1 and s0 > 0:
                    benchmark_return = (s1 - s0) / s0
                    alpha = total_return - benchmark_return
            except Exception:
                pass

        return SimResult(
            model_type="swing_agent",
            starting_capital=self.starting_capital,
            ending_capital=round(final_equity, 2),
            total_return_pct=round(total_return, 4),
            annualized_return_pct=round(ann_return, 4),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            max_drawdown_pct=round(max_dd, 4),
            calmar_ratio=round(calmar, 3),
            benchmark_return_pct=round(benchmark_return, 4),
            alpha_pct=round(alpha, 4),
            information_ratio=round(info_ratio, 3),
            total_trades=len(accepted_trades),
            win_rate=round(win_rate, 4),
            avg_pnl_pct=round(avg_pnl, 6),
            profit_factor=round(profit_factor, 3),
            avg_hold_bars=round(avg_hold, 1),
            transaction_costs_total=round(tx_costs_total, 2),
            exit_breakdown=dict(exit_breakdown),
            equity_curve=equity_curve,
            monthly_pnl=dict(monthly),
            trades=accepted_trades,
        )

    def _empty_result(self) -> SimResult:
        return SimResult(
            model_type="swing_agent",
            starting_capital=self.starting_capital,
            ending_capital=self.starting_capital,
            total_return_pct=0.0, annualized_return_pct=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0,
            max_drawdown_pct=0.0, calmar_ratio=0.0,
        )

    # ─── Helpers ───────────────────────────────────────────────────────────────

    def _get_feature_engineer(self):
        if self._feature_engineer is None:
            from app.ml.features import FeatureEngineer
            self._feature_engineer = FeatureEngineer()
        return self._feature_engineer

    def _bars_up_to(
        self, df: pd.DataFrame, day: date, exclude_today: bool = True
    ) -> Optional[pd.DataFrame]:
        idx = pd.DatetimeIndex(df.index)
        if exclude_today:
            mask = idx.normalize().date < day
        else:
            mask = idx.normalize().date <= day
        result = df.loc[mask]
        return result if len(result) > 0 else None

    def _bars_on(self, df: pd.DataFrame, day: date) -> Optional[pd.Series]:
        idx = pd.DatetimeIndex(df.index)
        mask = idx.normalize().date == day
        rows = df.loc[mask]
        if len(rows) == 0:
            return None
        return rows.iloc[0]

    @staticmethod
    def _normalize_reason(reason: str) -> str:
        """Map Trader exit reasons to canonical backtest reasons."""
        r = reason.lower()
        if "target" in r:
            return "TARGET"
        if "stop" in r:
            return "STOP"
        if "max_hold" in r or "force" in r:
            return "MAX_HOLD"
        return reason.upper()
