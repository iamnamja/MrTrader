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
from app.ml.schema_log import log_features, log_normalize, log_predict

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
STOP_SLIPPAGE_PCT = 0.0005   # 5bps adverse slippage on stop-triggered exits (realistic stop-order fill)
ENTRY_SLIPPAGE_PCT = 0.0003  # 3bps MOO slippage vs printed open (market-on-open fills)
DEFAULT_MAX_HOLD_BARS = 40  # ~8 weeks; decoupled from MAX_OPEN_POSITIONS to avoid L/S position-count bloat


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
    # MTM equity cached each bar before entries run — used for all sizing/RM decisions.
    # None until first bar closes; falls back to entry-price equity if not set.
    _cached_mtm_equity: float = field(default=None, repr=False)
    # Collateral reserved for open short positions (short entry notional).
    # Prevents the same dollars being used to size both long and short books.
    short_collateral: float = 0.0

    @property
    def position_market_value(self) -> float:
        # Uses entry price (no MTM) — kept for backward compat with live PM code paths.
        return sum(p.entry_price * p.quantity for p in self.positions.values())

    def equity_mtm(self, today_closes: dict) -> float:
        """Mark-to-market equity: longs at today's close, shorts as unrealized PnL.

        For longs:  contribution = today_close * qty
        For shorts: contribution = (entry_price - today_close) * qty
                    (profit when price falls; cash already holds entry proceeds as margin)
        Falls back to entry_price if today_close is unavailable for a symbol.
        """
        pmv = 0.0
        for sym, pos in self.positions.items():
            close = today_closes.get(sym, pos.entry_price)
            is_short = getattr(pos, "direction", "long") == "short"
            if is_short:
                pmv += (pos.entry_price - close) * pos.quantity
            else:
                pmv += close * pos.quantity
        return self.cash + pmv

    def update_mtm(self, today_closes: dict) -> float:
        """Compute MTM equity and cache it for use in sizing/RM decisions this bar."""
        self._cached_mtm_equity = self.equity_mtm(today_closes)
        return self._cached_mtm_equity

    @property
    def equity(self) -> float:
        return self.cash + self.position_market_value

    @property
    def equity_decision(self) -> float:
        """MTM-aware equity for position sizing and RM rules.
        Uses the cached MTM equity if available (set by update_mtm each bar),
        otherwise falls back to entry-price equity (safe at t=0 when no positions open).
        """
        if self._cached_mtm_equity is not None:
            return self._cached_mtm_equity
        return self.equity

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
        short_borrow_rate_annual: float = 0.05,  # Bug fix: realistic borrow cost (5%/yr default; was 0.005)
        proposal_pool_size: Optional[int] = None,  # Lockstep: candidates forwarded to Trader/RM (default: max(top_n*5, 50))
        no_atr_stops: bool = False,  # Phase 4: disable ATR stops entirely (hold to HOLD_DAYS target)
        # Phase RA — REBALANCE mode
        rebalance_mode: bool = False,        # bypass signal layer; use top-N rebalance
        rebalance_days: int = 20,            # rebalance every N simulation bars
        rebalance_target_n: int = 30,        # target number of long positions
        rebalance_sector_cap: float = 0.30,
        rebalance_add_threshold: int = 15,
        rebalance_drop_threshold: int = 30,
        rebalance_min_adv: float = 20_000_000.0,
        rebalance_regime_fn=None,            # callable(day) -> float multiplier; None = 1.0
        rebalance_inv_vol: bool = False,     # Phase RB.2: use inverse-vol sizing
        rebalance_inv_vol_lookback: int = 20,
        rebalance_inv_vol_min_mult: float = 0.5,
        rebalance_inv_vol_max_mult: float = 2.0,
        rebalance_spy_vol_damper: bool = False,  # Phase 91: halve gross_mult when SPY 21d vol > 80th pct
        rebalance_spy_vol_damper_scale: float = 0.50,  # scale factor applied when vol is elevated
        # Phase 2 — L/S extension
        enable_shorts: bool = False,        # False = pure long-only (backward-compat default)
        short_target_n: int = 30,           # number of short positions
        long_gross: float = 0.95,           # long book gross as fraction of equity
        short_gross: float = 0.55,          # short book gross as fraction of equity
        short_add_threshold: int = 15,      # hysteresis: add short when rank-from-bottom ≤ this
        short_drop_threshold: int = 30,     # hysteresis: drop short when rank-from-bottom > this
        short_min_adv: float = 50_000_000.0,    # tighter ADV for shorts (need to borrow)
        short_regime_fn=None,               # callable(day)->float; asymmetric regime multiplier
        long_regime_fn=None,                # explicit long-side regime fn (overrides rebalance_regime_fn when set)
    ):
        self.model = model
        self.starting_capital = starting_capital
        self.limits = limits or RiskLimits()
        self.min_confidence = min_confidence
        self.top_n = top_n
        self.transaction_cost_pct = transaction_cost_pct
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.no_atr_stops = no_atr_stops
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
        self.short_borrow_rate_annual = short_borrow_rate_annual  # Bug fix: configurable borrow
        # Lockstep: how many top-ranked candidates to forward to Trader/RM before entry cap.
        # 5× top_n gives RM/technical gates headroom without reaching into low-conviction tail.
        # Widening to 20× degraded win rate (rank 50-200 names have no edge vs noise stops).
        self.proposal_pool_size = proposal_pool_size if proposal_pool_size is not None else max(self.top_n * 5, 50)

        # Phase RA — REBALANCE mode state
        self.rebalance_mode = rebalance_mode
        self.rebalance_days = rebalance_days
        self.rebalance_target_n = rebalance_target_n
        self.rebalance_sector_cap = rebalance_sector_cap
        self.rebalance_add_threshold = rebalance_add_threshold
        self.rebalance_drop_threshold = rebalance_drop_threshold
        self.rebalance_min_adv = rebalance_min_adv
        self.rebalance_regime_fn = rebalance_regime_fn  # callable(day)->float or None
        self.rebalance_inv_vol = rebalance_inv_vol
        self.rebalance_inv_vol_lookback = rebalance_inv_vol_lookback
        self.rebalance_inv_vol_min_mult = rebalance_inv_vol_min_mult
        self.rebalance_inv_vol_max_mult = rebalance_inv_vol_max_mult
        self.rebalance_spy_vol_damper = rebalance_spy_vol_damper
        self.rebalance_spy_vol_damper_scale = rebalance_spy_vol_damper_scale
        # Phase 2 — L/S state
        self.enable_shorts = enable_shorts
        self.short_target_n = short_target_n
        self.long_gross = long_gross
        self.short_gross = short_gross
        self.short_add_threshold = short_add_threshold
        self.short_drop_threshold = short_drop_threshold
        self.short_min_adv = short_min_adv
        self.short_regime_fn = short_regime_fn
        self.long_regime_fn = long_regime_fn
        self._rebalance_bar_idx = 0  # counts simulation bars for cadence

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

        # Anchor trading_days to SPY's calendar if available (C3 fix: union of all symbol
        # indices includes stale/missing-bar days that inject zero-return artifacts into Sharpe).
        _spy_data = symbols_data.get("SPY")
        if _spy_data is None:
            _spy_data = symbols_data.get("spy")
        if _spy_data is not None and len(_spy_data) > 0:
            all_days = sorted({
                d.date() if hasattr(d, "date") else d
                for d in _spy_data.index
            })
        else:
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
            # 0. Bug fix (look-ahead): cache MTM equity for sizing/RM decisions made at
            # TODAY'S OPEN. Previously used today's close for the cached equity, which
            # leaked future information into position sizing and risk-rule validation.
            # We now use the last available close STRICTLY BEFORE today (i.e. yesterday's
            # close) — the latest information available to the decision-maker at the open.
            _prior_closes = {}
            for _sym, _pos in portfolio.positions.items():
                _df = symbols_data.get(_sym)
                if _df is None:
                    continue
                _prior = self._bars_up_to(_df, day, exclude_today=True)
                if _prior is not None and len(_prior) > 0 and "close" in _prior.columns:
                    _prior_closes[_sym] = float(_prior["close"].iloc[-1])
            portfolio.update_mtm(_prior_closes)

            # 1. Advance bars_held for all open positions
            for pos in portfolio.positions.values():
                pos.bars_held += 1

            # 2. Exit check: scan today's bar for stop/target hits
            closed = self._process_exits(day, symbols_data, portfolio)
            for trade, tx_cost in closed:
                accepted_trades.append(trade)
                tx_costs_total += tx_cost
            # peak_equity is updated at EOD (after MTM); not mid-bar to avoid inflating DD gate

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
                        spy_hist = _spy_closes.loc[spy_dates < day]
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
                        vix_today = _vix_closes.loc[vix_dates < day]
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
                        vix_today = _vix_closes.loc[vix_dates < day]
                        if len(vix_today) > 0 and float(vix_today.iloc[-1]) >= self.pm_abstention_vix:
                            _skip_entries = True
                            logger.debug("PM abstention (VIX) on %s: %.1f >= %.1f",
                                         day, float(vix_today.iloc[-1]), self.pm_abstention_vix)
                    if not _skip_entries and self.pm_abstention_spy_ma_days > 0 and _spy_closes is not None:
                        spy_idx = _spy_closes.index
                        spy_dates = spy_idx.date if hasattr(spy_idx, 'date') else pd.DatetimeIndex(spy_idx).date
                        spy_hist = _spy_closes.loc[spy_dates < day]
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
                        spy_hist = _spy_closes.loc[spy_dates < day]
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
                    spy_hist = _spy_closes.loc[spy_dates < day]
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
                            vix_hist = _vix_closes.loc[vix_dates_v < day]
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

            # 5. Entry: REBALANCE mode or SIGNAL mode
            if self.rebalance_mode:
                if self._rebalance_bar_idx % self.rebalance_days == 0:
                    new_trades, new_tx = self._process_rebalance(
                        day, symbols_data, portfolio, sector_map,
                        vix_history=_vix_closes,
                    )
                    accepted_trades.extend(new_trades)
                    tx_costs_total += new_tx
                self._rebalance_bar_idx += 1
            elif proposals and not _skip_entries:
                new_trades, new_tx = self._process_entries(
                    day, proposals, symbols_data, portfolio, sector_map,
                    max_positions=_max_pos_today, vix_history=_vix_closes,
                )
                accepted_trades.extend(new_trades)
                tx_costs_total += new_tx

            # Record MTM equity for this day (already computed at step 0 above).
            # Re-run update_mtm with any new positions opened this bar so exits are included.
            # Bug fix (WF deep-review pass 5): when a symbol has no bar today (halt, holiday,
            # delisting mid-fold), equity_mtm previously fell back to entry_price — producing
            # a spurious "snap to entry" that inflated daily-return volatility (and depressed
            # Sharpe / inflated max_dd). We now carry forward the last known close strictly
            # prior to today so a halt day contributes zero return, as it should.
            _today_closes_eod = {}
            for _sym, _pos in portfolio.positions.items():
                _df = symbols_data.get(_sym)
                if _df is None:
                    continue
                _bar = self._bars_on(_df, day)
                if _bar is not None:
                    _today_closes_eod[_sym] = float(_bar["close"])
                else:
                    # Carry-forward: last close strictly before today
                    _prior = self._bars_up_to(_df, day, exclude_today=True)
                    if _prior is not None and len(_prior) > 0 and "close" in _prior.columns:
                        _today_closes_eod[_sym] = float(_prior["close"].iloc[-1])
                    # else: equity_mtm will fall back to entry_price (no prior bar at all)
            equity_by_date[day] = portfolio.equity_mtm(_today_closes_eod)
            if equity_by_date[day] > portfolio.peak_equity:
                portfolio.peak_equity = equity_by_date[day]
            portfolio.daily_pnl = 0.0  # reset for next day

        # Force-close any remaining open positions at last bar close.
        # Bug fix (WF deep-review pass 5): previously used df["close"].iloc[-1] which is the
        # last close in the FULL dataframe — including bars AFTER end_date (look-ahead).
        # Fold dataframes typically carry the full history (training + test + lookahead),
        # so iloc[-1] could be weeks past the fold's test window. We now restrict to the
        # last close on-or-before end_date.
        for sym, pos in list(portfolio.positions.items()):
            df = symbols_data.get(sym)
            exit_price = None
            if df is not None and len(df) > 0:
                _prior = self._bars_up_to(df, end_date, exclude_today=False)
                if _prior is not None and len(_prior) > 0 and "close" in _prior.columns:
                    exit_price = float(_prior["close"].iloc[-1])
            if exit_price is None:
                # Symbol has no data at fold end (delisted/gap) — exit at entry price
                # to record the trade rather than silently discarding it.
                logger.warning("FORCE_CLOSE: no bar data for %s at %s — closing at entry", sym, end_date)
                exit_price = pos.entry_price
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

        Look-ahead audit (deep-review pass 3, VERIFIED CORRECT):
        Features for the day-T scoring decision are built from bars STRICTLY
        BEFORE day T (`_bars_up_to(df, day, exclude_today=True)` — see line
        ~542 below and feature_cache.py mask `df.index.date < day`). The
        resulting proposals are then filled at day T's open inside
        `_process_entries`. This is actually conservative: a real EOD->MOO
        workflow would be allowed to use day-T's close to decide T+1's open
        entry. The simulator uses one day LESS information than the live
        system could legitimately consume — there is no leakage.
        Sector-ETF override likewise uses strict `<` (see feature_cache.py).
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
        _wf_run_id = getattr(self, "_wf_run_id", "wf-unknown")
        try:
            if model_feat_names:
                X = np.array([
                    [features_by_symbol[s].get(f, 0.0) for f in model_feat_names]
                    for s in sym_list
                ])
            else:
                X = np.array([list(features_by_symbol[s].values()) for s in sym_list])
            X = np.nan_to_num(X, nan=0.0)
            _feat_names = list(model_feat_names) if model_feat_names else []
            _feat_hash = log_features("wf", _wf_run_id, day, _feat_names, X, sym_list)
            X = self._normalize_for_inference(X, sym_list, day)
            _norm_name = "none_lambdarank" if getattr(self.model, "model_type", "") == "lambdarank" else "cs_normalize"
            log_normalize("wf", _wf_run_id, day, _norm_name, len(sym_list), X)
            vix_now = self._vix_at(vix_history, day)
            _, probas = self.model.predict_with_vix(X, vix_level=vix_now)
            _model_ver = str(getattr(self.model, "version", "unknown"))
            log_predict("wf", _wf_run_id, day, _model_ver, _feat_hash, probas, sym_list)
        except Exception as exc:
            logger.warning("PM score failed on %s (%s): %s", day, type(exc).__name__, exc)
            return []

        order = np.argsort(probas)[::-1]
        sym_arr = np.array(sym_list)
        proposals = []
        for idx in order:
            sym = sym_arr[idx]
            prob = float(probas[idx])
            if prob < self.min_confidence:
                break
            if self.max_vol_pct is not None:
                vol_pct = features_by_symbol[sym].get("vol_percentile_52w", 0.0)
                if vol_pct > self.max_vol_pct / 100.0:
                    continue
            proposals.append((sym, prob))
            if len(proposals) >= self.proposal_pool_size:
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

        for sym, idx_map in sorted(cache.date_index.items()):
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

        _wf_run_id = getattr(self, "_wf_run_id", "wf-unknown")
        try:
            X = np.nan_to_num(np.vstack(rows), nan=0.0)
            sym_arr = np.array(sym_list)
            _feat_hash = log_features("wf-cached", _wf_run_id, day, list(model_feat_names), X, sym_list)
            _norm_name = "none_lambdarank" if getattr(self.model, "model_type", "") == "lambdarank" else "cs_normalize"
            X = self._normalize_for_inference(X, sym_arr, day)
            log_normalize("wf-cached", _wf_run_id, day, _norm_name, len(sym_list), X)
            vix_now = self._vix_at(vix_history, day)
            _, probas = self.model.predict_with_vix(X, vix_level=vix_now)
            _model_ver = str(getattr(self.model, "version", "unknown"))
            log_predict("wf-cached", _wf_run_id, day, _model_ver, _feat_hash, probas, sym_list)
        except Exception as exc:
            logger.warning("PM score (cached) failed on %s (%s): %s", day, type(exc).__name__, exc)
            return []

        # Lockstep: score full universe (len=N), return proposal_pool_size candidates so
        # Trader/RM gates have headroom to filter and still fill _max_pos_today slots.
        # Fix O(N²): build vol_pcts array aligned to sym_list, then reindex after sort.
        vol_arr = np.array(vol_pcts)
        order = np.argsort(probas)[::-1]
        proposals = []
        for idx in order:
            sym = sym_list[idx]
            prob = float(probas[idx])
            if prob < self.min_confidence:
                break  # sorted desc — remaining will be lower
            if self.max_vol_pct is not None and vol_arr[idx] > self.max_vol_pct / 100.0:
                continue
            proposals.append((sym, prob))
            if len(proposals) >= self.proposal_pool_size:
                break

        logger.debug(
            "PM score (cached) %s: %d symbols scored → %d proposals (pool=%d, top_n=%d)",
            day, len(sym_list), len(proposals), self.proposal_pool_size, self.top_n,
        )
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

        LambdaRank exception: the model's internal StandardScaler handles normalization
        at predict-time. External cs_normalize would double-normalize and corrupt the
        feature distribution (scaler was fit on raw features, not cs-normalized features).
        """
        # LambdaRank has its own internal StandardScaler; do not apply external normalization.
        if getattr(self.model, "model_type", "") == "lambdarank":
            return X

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
        """Return last available VIX close BEFORE `day` (strictly < day).

        Using today's close would be look-ahead: entry decisions happen at open,
        but VIX close is only known at market close. Use yesterday's close.
        """
        if vix_history is None or vix_history.empty:
            return None
        try:
            idx = vix_history.index
            idx_dates = [i.date() if hasattr(i, "date") else i for i in idx]
            candidates = [(d, v) for d, v in zip(idx_dates, vix_history.values) if d < day]
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

                # ATR-based stop — must match training label bounds exactly.
                # training.py:261: clip(0.5*atr_pct, ATR_MIN_TARGET/2, ATR_MAX_TARGET/2)
                #                = clip(0.5*atr_pct, 0.0075, 0.04)
                atr_stop_pct = float(np.clip(self.atr_stop_mult * atr_pct, 0.0075, 0.04))
                if is_ema_crossover:
                    # Momentum: never go tighter than 10-bar pivot low (structure level)
                    pivot_low = float(np.min(low[-11:-1])) if len(low) >= 11 else close * 0.97
                    stop_price = max(pivot_low * 0.995, close * (1 - atr_stop_pct))
                else:
                    # Pullback/RSI-dip: never go tighter than EMA20 with small buffer
                    stop_price = max(ema20 * 0.995, close * (1 - atr_stop_pct))

                # Target: 1.5× ATR — clip matches training.py:260: clip(1.5*atr_pct, 0.015, 0.08)
                target_pct = float(np.clip(self.atr_target_mult * atr_pct, 0.015, 0.08))
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
        direction: str = "BUY",
    ) -> Tuple[bool, str]:
        """Run key RM rules against current portfolio state. Returns (ok, reason)."""
        trade_cost = entry_price * quantity
        equity = portfolio.equity_decision  # MTM-aware: avoids phantom equity from short opens

        ok, msg = validate_buying_power(trade_cost, portfolio.buying_power, self.limits, direction=direction)
        if not ok:
            return False, msg

        ok, msg = validate_position_size(trade_cost, equity, self.limits)
        if not ok:
            return False, msg

        sector_val = portfolio.sector_values.get(sector, 0.0)
        ok, msg = validate_sector_concentration(
            trade_cost, sector_val, equity, sector, self.limits, direction=direction
        )
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
        new_trade_risk = abs(entry_price - stop_price) * quantity
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

            _raw_open = self._scalar(today_bar["open"])
            if _raw_open <= 0:
                continue
            # MOO fills slip vs printed open; longs pay more, shorts receive less
            entry_price = (_raw_open * (1 + ENTRY_SLIPPAGE_PCT) if direction == "long"
                           else _raw_open * (1 - ENTRY_SLIPPAGE_PCT))

            # Phase 34: No-chase filter — skip large overnight gaps and extended entries
            # Not applied in factor portfolio mode (monthly rebalance enters regardless of daily gap)
            # Not applied to shorts: gapping up is favorable for a short entry (selling into strength)
            if not is_short and self.factor_scorer is None and bars_yesterday is not None and len(bars_yesterday) >= 14:
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
                    # Short: ATR-based stops mirror the long path (clip bounds match training labels)
                    _short_atr_stop_pct = SWING_STOP_PCT    # fallback if ATR unavailable
                    _short_atr_tgt_pct = SWING_TARGET_PCT
                    if bars_yesterday is not None and len(bars_yesterday) >= 14:
                        try:
                            _sh = bars_yesterday["high"].to_numpy(dtype=float)
                            _sl = bars_yesterday["low"].to_numpy(dtype=float)
                            _sc = bars_yesterday["close"].to_numpy(dtype=float)
                            _tr = np.maximum(_sh[1:] - _sl[1:],
                                             np.maximum(abs(_sh[1:] - _sc[:-1]),
                                                        abs(_sl[1:] - _sc[:-1])))
                            _s_atr_pct = float(np.mean(_tr[-14:])) / max(float(_sc[-1]), 1e-6)
                            _short_atr_stop_pct = float(np.clip(self.atr_stop_mult * _s_atr_pct, 0.0075, 0.04))
                            _short_atr_tgt_pct = float(np.clip(self.atr_target_mult * _s_atr_pct, 0.015, 0.08))
                        except Exception:
                            pass
                    stop_price = entry_price * (1 + _short_atr_stop_pct)
                    target_price = entry_price * (1 - _short_atr_tgt_pct)

            # Phase 4 isolation: override stops so positions hold to max_hold_bars.
            # Use a very tight synthetic stop for sizing only (5% risk) so quantities
            # are conservative, then widen actual stop to "never triggers."
            if self.no_atr_stops:
                if is_short:
                    stop_price = entry_price * 100.0   # never triggers (short stops above entry)
                    target_price = entry_price * 0.01  # never triggers
                else:
                    stop_price = entry_price * 0.0001  # never triggers (long stops below entry)
                    target_price = entry_price * 100.0  # never triggers

            # Position sizing: size_position requires stop_price < entry_price (long semantics).
            # For shorts, flip the stop so risk-per-share = |entry - stop| is preserved.
            conf_for_sizing = abs(confidence)
            if self.no_atr_stops:
                # Use synthetic 5% stop for sizing to avoid infinite position sizes
                sizing_stop = entry_price * (1 + 0.05) if is_short else entry_price * (1 - 0.05)
                stop_for_sizing = sizing_stop
            else:
                stop_for_sizing = (2 * entry_price - stop_price) if is_short else stop_price
            quantity = size_position(
                account_equity=portfolio.equity_decision,
                available_cash=portfolio.cash,
                entry_price=entry_price,
                stop_price=stop_for_sizing,
                ml_score=conf_for_sizing,
            )
            # Apply RM position-size cap so the trade doesn't auto-reject
            max_position_dollars = portfolio.equity_decision * self.limits.MAX_POSITION_SIZE_PCT
            quantity = min(quantity, max(1, int(max_position_dollars / entry_price)))
            if quantity <= 0:
                continue

            sector = sector_map.get(sym, "UNKNOWN")
            # When no_atr_stops is on, sentinel stop_price is unrealistic for risk sizing.
            # Use the same synthetic 5% stop that was used for quantity sizing.
            _rm_stop = stop_for_sizing if self.no_atr_stops else stop_price
            ok, reason = self._rm_validate(
                sym, entry_price, _rm_stop, quantity, portfolio, sector,
                direction="SELL_SHORT" if is_short else "BUY",
            )
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
            # Use signed delta so same-sector shorts reduce exposure (match live RM behavior)
            _sector_delta = -trade_cost if is_short else trade_cost
            portfolio.sector_values[sector] = (
                portfolio.sector_values.get(sector, 0.0) + _sector_delta
            )
            tx_costs_total += tx_cost

        return entered_trades, tx_costs_total

    # ─── Phase RA: REBALANCE mode entry ────────────────────────────────────────

    def _effective_cash(self, portfolio: "_PortfolioState") -> float:
        """Cash available for new LONG opens, net of short collateral reserve.

        Short entries receive cash proceeds but simultaneously reserve collateral
        equal to the entry notional. Without this, the same dollars would be used
        to size both books (Opus BUG-2/#7).
        """
        return portfolio.cash - portfolio.short_collateral

    def _rebalance_drop_position(
        self,
        sym: str,
        pos: "_Position",
        symbols_data: Dict[str, pd.DataFrame],
        portfolio: "_PortfolioState",
        day: "date",
        regime_label: str,
    ) -> tuple:
        """Close a single position (long or short) at today's open. Returns (trades, tx_cost).

        Separated from _process_rebalance to enable unit-testing of P&L sign (Opus BUG-1).
        """
        from app.backtesting.agent_simulator import Trade  # noqa: local import avoids circular
        is_short = getattr(pos, "direction", "long") == "short"
        df = symbols_data.get(sym)
        exit_price = pos.entry_price
        if df is not None:
            today_bar = self._bars_on(df, day)
            if today_bar is not None:
                open_col = "open" if "open" in today_bar.index else "Open"
                close_col = "close" if "close" in today_bar.index else "Close"
                if open_col in today_bar.index:
                    exit_price = float(today_bar[open_col])
                elif close_col in today_bar.index:
                    exit_price = float(today_bar[close_col])

        cost = exit_price * pos.quantity * self.transaction_cost_pct

        if is_short:
            # Cover: P&L = (entry - exit) × qty — profit when price falls.
            # Short entry already added proceeds to cash and reserved collateral.
            # On cover: pay exit notional + exit tx; release collateral reserve.
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity
            net_pnl = gross_pnl - cost  # entry tx already charged at entry (BUG-29 parity)
            collateral = pos.entry_price * pos.quantity
            portfolio.cash -= exit_price * pos.quantity + cost   # pay to cover + exit tx
            portfolio.short_collateral = max(0.0, portfolio.short_collateral - collateral)
        else:
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
            net_pnl = gross_pnl - cost  # entry tx already charged at entry (BUG-29 fix)
            portfolio.cash += exit_price * pos.quantity - cost

        del portfolio.positions[sym]
        trade = Trade(
            symbol=sym,
            entry_date=pos.entry_date,
            exit_date=day,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=net_pnl,
            pnl_pct=(exit_price / pos.entry_price - 1.0) * (-1 if is_short else 1),
            hold_bars=pos.bars_held,
            exit_reason="REBALANCE_DROP",
            source="REBALANCE",
            rank_at_entry=pos.confidence,
            regime_at_entry=regime_label,
            gross_exposure_mult=1.0,
            rebalance_date=day,
        )
        return [trade], cost

    def _process_rebalance(
        self,
        day: date,
        symbols_data: Dict[str, pd.DataFrame],
        portfolio: _PortfolioState,
        sector_map: Dict[str, str],
        vix_history=None,
    ) -> Tuple[List, float]:
        """Score all symbols, compute target portfolio, close drops, open adds.

        When enable_shorts=True:
          - Long book: top-N from scored list, long_regime_fn multiplier
          - Short book: bottom-N (reversed list), short_regime_fn multiplier
          - Separate liquidity/sector-cap/hysteresis for each book
          - Per-side dollar budgets from split_gross_budgets
          - Short collateral tracked to prevent double-counting
        """
        from app.strategy.portfolio_construction import (
            apply_sector_cap,
            apply_sector_cap_shorts,
            compute_equal_weights,
            compute_inverse_vol_weights,
            compute_target_portfolio,
            compute_target_portfolio_shorts,
            liquidity_filter,
            split_gross_budgets,
        )

        # 1. Score
        scored = self._pm_score(day, symbols_data, vix_history)
        if not scored:
            return [], 0.0
        ranked_symbols = [s for s, _ in scored]
        score_of = {s: float(sc) for s, sc in scored}

        # 2. Liquidity filter (PIT-safe) — tighter threshold for shorts
        long_eligible = liquidity_filter(
            symbols_data, as_of=day,
            min_avg_daily_dollar_vol=self.rebalance_min_adv,
        )
        short_eligible = liquidity_filter(
            symbols_data, as_of=day,
            min_avg_daily_dollar_vol=self.short_min_adv if self.enable_shorts else self.rebalance_min_adv,
        ) if self.enable_shorts else set()

        ranked_eligible = [s for s in ranked_symbols if s in long_eligible] or ranked_symbols

        # Short ranking = reverse of long ranking (worst-first), filtered by short_eligible
        worst_first = list(reversed(ranked_symbols))
        short_ranked_eligible = [s for s in worst_first if s in short_eligible] if self.enable_shorts else []

        # 3. Sector cap (both books)
        capped = apply_sector_cap(
            ranked_eligible, sector_map,
            cap=self.rebalance_sector_cap,
            n_target=self.rebalance_target_n,
        )
        short_capped = apply_sector_cap_shorts(
            short_ranked_eligible, sector_map,
            cap=self.rebalance_sector_cap,
            n_target=self.short_target_n,
        ) if self.enable_shorts else []

        # 4. Hysteresis target — separate current holdings for each book
        current_longs = [s for s, p in portfolio.positions.items()
                         if getattr(p, "direction", "long") == "long"]
        current_shorts = [s for s, p in portfolio.positions.items()
                          if getattr(p, "direction", "long") != "long"]

        delta = compute_target_portfolio(
            capped, current_longs,
            n_target=self.rebalance_target_n,
            add_rank_threshold=self.rebalance_add_threshold,
            drop_rank_threshold=self.rebalance_drop_threshold,
        )
        short_delta = compute_target_portfolio_shorts(
            short_capped, current_shorts,
            n_target=self.short_target_n,
            add_rank_threshold=self.short_add_threshold,
            drop_rank_threshold=self.short_drop_threshold,
        ) if self.enable_shorts else None

        # 5. Per-side regime multipliers (asymmetric gate)
        long_fn = self.long_regime_fn or self.rebalance_regime_fn
        long_mult = long_fn(day) if long_fn else 1.0
        short_mult = self.short_regime_fn(day) if (self.enable_shorts and self.short_regime_fn) else (
            1.0 if self.enable_shorts else 0.0
        )

        def _regime_label(mult: float) -> str:
            if mult >= 0.95:
                return "BULL"
            if mult <= 0.35:
                return "BEAR"
            return "NEUTRAL"

        regime_label = _regime_label(long_mult)

        # 5b. SPY vol damper (applies to long book only — shorts benefit from high vol)
        if self.rebalance_spy_vol_damper:
            import numpy as _np
            _as_of_ts = pd.Timestamp(day)
            _spy_df = symbols_data.get("SPY")
            if _spy_df is not None:
                _spy_col = "close" if "close" in _spy_df.columns else "Close"
                if _spy_col in _spy_df.columns:
                    _spy_past = _spy_df[_spy_col][_spy_df.index <= _as_of_ts].dropna()
                    if len(_spy_past) >= 273:
                        _log_rets = _np.log(_spy_past / _spy_past.shift(1)).dropna()
                        _vol_21 = float(_log_rets.iloc[-21:].std() * _np.sqrt(252))
                        _rolling_21 = _log_rets.rolling(21).std() * _np.sqrt(252)
                        _hist_vols = _rolling_21.dropna().iloc[-252:]
                        if len(_hist_vols) >= 50:
                            _pct_rank = float((_vol_21 > _hist_vols).mean())
                            if _pct_rank > 0.80:
                                long_mult *= self.rebalance_spy_vol_damper_scale
                                regime_label += "+HIGHVOL"

        # 6. Close drops (both books) — direction-aware P&L via _rebalance_drop_position
        tx_costs = 0.0
        closed_trades: List = []

        all_drops = list(delta.to_drop)
        if short_delta is not None:
            all_drops += list(short_delta.to_drop)

        for sym in all_drops:
            if sym not in portfolio.positions:
                continue
            pos = portfolio.positions[sym]
            trades, cost = self._rebalance_drop_position(sym, pos, symbols_data, portfolio, day, regime_label)
            closed_trades.extend(trades)
            tx_costs += cost

        # 7. Open long adds
        equity = portfolio.equity_decision
        if self.enable_shorts:
            long_budget, short_budget = split_gross_budgets(
                equity,
                net_target=self.long_gross - self.short_gross,
                gross_target=self.long_gross + self.short_gross,
                long_regime_mult=long_mult,
                short_regime_mult=short_mult,
            )
        else:
            long_budget = equity * long_mult
            short_budget = 0.0

        if self.rebalance_inv_vol:
            long_weights = compute_inverse_vol_weights(
                delta.to_add, symbols_data, as_of=day,
                total_equity=equity,
                gross_exposure_multiplier=long_mult,
                vol_lookback_days=self.rebalance_inv_vol_lookback,
                min_weight_mult=self.rebalance_inv_vol_min_mult,
                max_weight_mult=self.rebalance_inv_vol_max_mult,
            )
        else:
            long_weights = compute_equal_weights(delta.to_add, equity, long_mult)

        rank_of = {s: i + 1 for i, s in enumerate(ranked_eligible)}
        short_rank_of = {s: i + 1 for i, s in enumerate(short_ranked_eligible)}
        entered_trades: List = []

        for sym in delta.to_add:
            if sym in portfolio.positions:
                continue
            dollar_size = long_weights.get(sym, 0.0)
            if dollar_size <= 0:
                continue
            df = symbols_data.get(sym)
            if df is None:
                continue
            today_bar = self._bars_on(df, day)
            if today_bar is None:
                continue
            open_col = "open" if "open" in today_bar.index else "Open"
            close_col = "close" if "close" in today_bar.index else "Close"
            if open_col not in today_bar.index and close_col not in today_bar.index:
                continue
            entry_price = float(today_bar[open_col if open_col in today_bar.index else close_col])
            if entry_price <= 0:
                continue
            qty = max(1, int(dollar_size / entry_price))
            cost = entry_price * qty * self.transaction_cost_pct
            # Use effective cash (net of short collateral) to prevent double-spend (Opus BUG-2)
            if self._effective_cash(portfolio) < entry_price * qty + cost:
                continue
            portfolio.cash -= entry_price * qty + cost
            tx_costs += cost
            portfolio.positions[sym] = _Position(
                symbol=sym,
                entry_date=day,
                entry_price=entry_price,
                stop_price=entry_price * 0.0001,
                target_price=entry_price * 100.0,
                quantity=qty,
                highest_price=entry_price,
                confidence=float(rank_of.get(sym, 999)),
                direction="long",
            )
            entered_trades.append(Trade(
                symbol=sym,
                entry_date=day,
                exit_date=None,
                entry_price=entry_price,
                exit_price=None,
                quantity=qty,
                pnl=0.0,
                pnl_pct=0.0,
                hold_bars=0,
                exit_reason="OPEN",
                source="REBALANCE",
                rank_at_entry=rank_of.get(sym),
                score_at_entry=score_of.get(sym),
                regime_at_entry=regime_label,
                gross_exposure_mult=long_mult,
                rebalance_date=day,
            ))

        # 8. Open short adds (only when enable_shorts=True)
        if self.enable_shorts and short_delta is not None:
            short_n = max(1, len(short_delta.to_add)) if short_delta.to_add else 1
            short_per_pos = short_budget / short_n if short_delta.to_add else 0.0

            for sym in short_delta.to_add:
                if sym in portfolio.positions:
                    continue
                if short_per_pos <= 0:
                    continue
                df = symbols_data.get(sym)
                if df is None:
                    continue
                today_bar = self._bars_on(df, day)
                if today_bar is None:
                    continue
                open_col = "open" if "open" in today_bar.index else "Open"
                close_col = "close" if "close" in today_bar.index else "Close"
                if open_col not in today_bar.index and close_col not in today_bar.index:
                    continue
                entry_price = float(today_bar[open_col if open_col in today_bar.index else close_col])
                if entry_price <= 0:
                    continue
                qty = max(1, int(short_per_pos / entry_price))
                cost = entry_price * qty * self.transaction_cost_pct
                notional = entry_price * qty
                # Short entry: receive proceeds in cash; reserve same as collateral
                portfolio.cash += notional - cost   # proceeds received, tx paid
                portfolio.short_collateral += notional  # reserve prevents double-spend
                tx_costs += cost
                portfolio.positions[sym] = _Position(
                    symbol=sym,
                    entry_date=day,
                    entry_price=entry_price,
                    stop_price=entry_price * 1.20,   # soft sentinel (20% above entry)
                    target_price=entry_price * 0.70, # soft sentinel (30% below entry)
                    quantity=qty,
                    highest_price=entry_price,
                    confidence=float(short_rank_of.get(sym, 999)),
                    direction="short",
                )
                entered_trades.append(Trade(
                    symbol=sym,
                    entry_date=day,
                    exit_date=None,
                    entry_price=entry_price,
                    exit_price=None,
                    quantity=qty,
                    pnl=0.0,
                    pnl_pct=0.0,
                    hold_bars=0,
                    exit_reason="OPEN_SHORT",
                    source="REBALANCE",
                    rank_at_entry=short_rank_of.get(sym),
                    score_at_entry=score_of.get(sym),
                    regime_at_entry=regime_label,
                    gross_exposure_mult=short_mult,
                    rebalance_date=day,
                ))

        return closed_trades + entered_trades, tx_costs

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

            # Daily borrow cost for short positions (configurable annual rate, default 5%).
            # Use today's close as current notional (M6 fix: entry_price understates HTB tail risk).
            if is_short:
                borrow_cost = today_close * pos.quantity * self.short_borrow_rate_annual / 365
                portfolio.cash -= borrow_cost
                portfolio.daily_pnl -= borrow_cost

            # P0.2 fix: snapshot pre-bar extremes BEFORE updating highest_price.
            # check_exit receives the PRE-bar highest so its trailing-stop ratchet
            # cannot retroactively use today's H/L for the intrabar breach check.
            _orig_stop = pos.stop_price
            _orig_target = pos.target_price
            _pre_bar_highest = pos.highest_price

            # Track best price: longs use highest high, shorts use lowest low.
            # Done after snapshot so the update is only visible to FUTURE bars.
            if is_short:
                pos.highest_price = min(pos.highest_price, today_low)
            else:
                pos.highest_price = max(pos.highest_price, today_high)
            today_open = self._scalar(today_bar["open"])
            should_exit = False
            exit_reason = ""
            fill_price = today_close  # default

            if is_short:
                # Short: stop above entry (breached when high >= stop),
                # target below entry (hit when low <= target).
                if today_open >= _orig_stop:
                    # Gap-up through stop: fill at open (worse than stop) + slippage
                    should_exit = True
                    exit_reason = "stop_hit"
                    fill_price = today_open * (1 + STOP_SLIPPAGE_PCT)
                elif today_high >= _orig_stop:
                    should_exit = True
                    exit_reason = "stop_hit"
                    fill_price = _orig_stop * (1 + STOP_SLIPPAGE_PCT)
                elif today_open <= _orig_target:
                    # Gap-down through target: fill at open (better than target)
                    should_exit = True
                    exit_reason = "target_hit"
                    fill_price = today_open
                elif today_low <= _orig_target:
                    should_exit = True
                    exit_reason = "target_hit"
                    fill_price = _orig_target
            else:
                # Long: stop below entry (breached when low <= stop),
                # target above entry (hit when high >= target).
                if today_open <= _orig_stop:
                    # Gap-down through stop: fill at open (worse than stop) + slippage
                    should_exit = True
                    exit_reason = "stop_hit"
                    fill_price = today_open * (1 - STOP_SLIPPAGE_PCT)
                elif today_low <= _orig_stop:
                    should_exit = True
                    exit_reason = "stop_hit"
                    fill_price = _orig_stop * (1 - STOP_SLIPPAGE_PCT)
                elif today_open >= _orig_target:
                    # Gap-up through target: fill at open (better than target)
                    should_exit = True
                    exit_reason = "target_hit"
                    fill_price = today_open
                elif today_high >= _orig_target:
                    should_exit = True
                    exit_reason = "target_hit"
                    fill_price = _orig_target

            # If no intrabar stop/target breach, defer to check_exit for
            # max-hold/trailing-stop-on-close logic and update the trailing stop
            # for FUTURE bars (never applied retroactively to today's H/L).
            _check_dir = "SELL_SHORT" if is_short else "BUY"
            _max_hold = (self.max_hold_bars_override
                         if self.max_hold_bars_override is not None
                         else DEFAULT_MAX_HOLD_BARS)
            if not should_exit:
                ce_should_exit, ce_reason, new_stop = check_exit(
                    symbol=sym,
                    current_price=today_close,
                    entry_price=pos.entry_price,
                    stop_price=pos.stop_price,
                    target_price=pos.target_price,
                    highest_price=_pre_bar_highest,
                    bars_held=pos.bars_held,
                    min_hold_bars=1,
                    max_hold_bars=_max_hold,
                    direction=_check_dir,
                )
                # Persist trailing-stop ratchet only when ATR stops are active.
                # When no_atr_stops=True, sentinels must not be replaced with real
                # trailing values — otherwise the first profitable bar defeats the
                # no-stop isolation.
                if not self.no_atr_stops:
                    pos.stop_price = new_stop
                if ce_should_exit:
                    should_exit = True
                    exit_reason = ce_reason
                    fill_price = today_close
            else:
                # Even though we're exiting intrabar, refresh the trailing stop
                # state via check_exit so live/sim parity for the (unused) return.
                # Not strictly necessary since position is closing this bar.
                pass

            if should_exit:
                trade, tx = self._close_position(pos, day, fill_price, exit_reason, portfolio)
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
                # Net cash: proceeds already booked at entry; cover cost = exit*qty
                portfolio.cash += gross_pnl - tx_cost

        if portfolio is not None:
            portfolio.daily_pnl += net_pnl
            sector = getattr(pos, "sector", "UNKNOWN")
            cost_basis = pos.entry_price * pos.quantity
            # Reverse the signed delta applied at entry (shorts added negative, longs positive)
            _sector_unwind = cost_basis if is_short else -cost_basis
            portfolio.sector_values[sector] = (
                portfolio.sector_values.get(sector, 0.0) + _sector_unwind
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
        # Sharpe/Sortino must be computed from a daily return series (so the sqrt(252)
        # annualization applies). Falling back to per-trade returns mixes periodicity
        # and produces a meaningless annualized number — return 0 instead.
        from app.backtesting.strategy_simulator import StrategySimulator
        if len(daily_rets) >= 2:
            sharpe = StrategySimulator._sharpe(daily_rets, 252)
            sortino = StrategySimulator._sortino(daily_rets, 252)
        else:
            sharpe = 0.0
            sortino = 0.0
        max_dd = StrategySimulator._max_drawdown(eq_vals)
        calmar = ann_return / max(max_dd, 1e-9)

        winners = [t for t in accepted_trades if t.pnl_pct > 0]
        losers = [t for t in accepted_trades if t.pnl_pct <= 0]
        win_rate = len(winners) / max(len(accepted_trades), 1)
        avg_pnl = sum(t.pnl_pct for t in accepted_trades) / max(len(accepted_trades), 1)
        avg_hold = sum(t.hold_bars for t in accepted_trades) / max(len(accepted_trades), 1)
        gross_win = sum(t.pnl_pct for t in winners) if winners else 0.0
        gross_loss_raw = abs(sum(t.pnl_pct for t in losers)) if losers else 0.0
        # Bug fix (WF deep-review pass 5): previously divided by max(loss, 1e-9), which
        # exploded to ~1e9 when there were zero losing trades — producing a misleading
        # "infinite edge" metric. We now report 0.0 when there are no winners, and a
        # large but finite sentinel (999.0) when there are winners but no losers (small
        # folds; "undefined" PF). This is consistent with common backtest libraries.
        if gross_loss_raw <= 0:
            profit_factor = 0.0 if gross_win <= 0 else 999.0
        else:
            profit_factor = gross_win / gross_loss_raw

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
        row = rows.iloc[0]
        # Flatten MultiIndex columns to scalars so callers can do float(bar["open"])
        if isinstance(row.index, pd.MultiIndex):
            row = pd.Series(
                {col[0]: val for col, val in row.items()},
                name=row.name,
            )
        return row

    @staticmethod
    def _scalar(val) -> float:
        """Extract a scalar float from a value that may be a Series or array."""
        if hasattr(val, "iloc"):
            return float(val.iloc[0])
        return float(val)

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
