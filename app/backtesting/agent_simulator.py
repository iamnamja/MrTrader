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
from collections import defaultdict
from dataclasses import dataclass, field
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

        # Lazy-load FeatureEngineer (imports may be heavy)
        self._feature_engineer = None

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
        _vix_df = symbols_data.get("^VIX") or symbols_data.get("VIX")
        if _vix_df is not None and "close" in _vix_df.columns:
            _vix_closes = _vix_df["close"]

        for day in trading_days:
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

            # 3. PM: score all symbols using bars up to yesterday
            proposals = self._pm_score(day, symbols_data, vix_history=_vix_closes)

            # 4. Phase 35: Market regime gate — cut exposure in bear/fear regimes
            _skip_entries = False
            _max_pos_today = self.limits.MAX_OPEN_POSITIONS
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
                        opp_score = 0.35 * vix_score + 0.20 * vix_trend + 0.30 * ma_score + 0.15 * mom_score
                        if opp_score < 0.35:
                            _skip_entries = True
                            logger.debug("Opp score %.2f < 0.35 on %s — skipping entries", opp_score, day)
                        elif opp_score < 0.65:
                            _max_pos_today = min(_max_pos_today, 2)
                            logger.debug("Opp score %.2f on %s — capping candidates at 2", opp_score, day)
                except Exception:
                    pass

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
        """Return list of (symbol, confidence) sorted by confidence desc."""
        if self.model is None or not getattr(self.model, "is_trained", False):
            return []

        fe = self._get_feature_engineer()
        features_by_symbol: Dict[str, dict] = {}

        for sym, df in symbols_data.items():
            bars_to_yesterday = self._bars_up_to(df, day, exclude_today=True)
            if bars_to_yesterday is None or len(bars_to_yesterday) < 60:
                continue
            try:
                feats = fe.engineer_features(
                    sym, bars_to_yesterday, fetch_fundamentals=False,
                    as_of_date=day, regime_score=0.5,
                    vix_history=vix_history,
                )
                if feats is not None:
                    features_by_symbol[sym] = feats
            except Exception:
                continue

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
            X = cs_normalize(X)
            _, probas = self.model.predict(X)
        except Exception as exc:
            logger.debug("PM score failed on %s: %s", day, exc)
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
            )
            closes = bars_up_to_day["close"]
            close = float(closes.iloc[-1])

            # Long-term trend: price above EMA-200
            ema200 = float(closes.ewm(span=200, adjust=False).mean().iloc[-1])
            if close <= ema200:
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
        for sym, confidence in proposals:
            if len(portfolio.positions) >= _max_pos:
                break  # Phase 35: respect regime-adjusted position cap

            if sym in portfolio.positions:
                continue  # already holding

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
            if bars_yesterday is not None and len(bars_yesterday) >= 14:
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
                    fe = self._get_feature_engineer()
                    feats = fe.engineer_features(sym, bars_yesterday,
                                                 fetch_fundamentals=False, as_of_date=day,
                                                 vix_history=vix_history)
                    if feats is not None and not self.meta_model.should_enter(feats):
                        continue
                except Exception:
                    pass

            # Trader: technical signal gate
            should_enter, stop_price, target_price = self._trader_signal(sym, bars_yesterday)
            if not should_enter:
                continue

            # Use signal stops if valid, else fallback percentages
            if stop_price <= 0 or stop_price >= entry_price:
                stop_price = entry_price * (1 - SWING_STOP_PCT)
            if target_price <= entry_price:
                target_price = entry_price * (1 + SWING_TARGET_PCT)

            # Position sizing via actual size_position(), then cap to RM position limit
            quantity = size_position(
                account_equity=portfolio.equity,
                available_cash=portfolio.cash,
                entry_price=entry_price,
                stop_price=stop_price,
                ml_score=confidence,
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

            portfolio.cash -= trade_cost + tx_cost
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

            pos.highest_price = max(pos.highest_price, today_high)

            # Use check_exit() — the actual Trader exit logic
            should_exit, exit_reason, new_stop = check_exit(
                symbol=sym,
                current_price=today_close,
                entry_price=pos.entry_price,
                stop_price=pos.stop_price,
                target_price=pos.target_price,
                highest_price=pos.highest_price,
                bars_held=pos.bars_held,
                min_hold_bars=1,
                max_hold_bars=self.limits.MAX_OPEN_POSITIONS * 4,  # ~20 days
            )
            pos.stop_price = new_stop

            # Intrabar stop/target override (check_exit uses close; we check H/L)
            if not should_exit:
                if today_low <= pos.stop_price:
                    should_exit = True
                    exit_reason = "stop_hit"
                    today_close = pos.stop_price
                elif today_high >= pos.target_price:
                    should_exit = True
                    exit_reason = "target_hit"
                    today_close = pos.target_price

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
        tx_cost = exit_price * pos.quantity * self.transaction_cost_pct
        gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        net_pnl = gross_pnl - tx_cost

        if portfolio is not None:
            # Credit back proceeds minus exit transaction cost
            portfolio.cash += exit_price * pos.quantity - tx_cost
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
            pnl_pct=round((exit_price - pos.entry_price) / pos.entry_price, 6),
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
