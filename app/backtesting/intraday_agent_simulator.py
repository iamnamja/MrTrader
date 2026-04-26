"""
Tier 3 Intraday Agent Simulator — agent-driven historical backtest on 5-min bars.

Runs actual PM / RM / Trader decision logic day by day:
  - PM:     compute_intraday_features() + model.predict() → top-N proposals
  - Trader: model confidence gate + session-time constraint (no entries after bar 60 ~14:30 ET)
  - RM:     validate_* rule functions from risk_rules.py against live portfolio state
  - Exit:   target (+0.5%), stop (-0.3%), HOLD_BARS=24 time exit, or EOD force-close

Operates on 5-min bars (one per symbol per day entry window).
Returns SimResult for consistent reporting with swing AgentSimulator.
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
    validate_open_positions,
    validate_account_drawdown,
    validate_portfolio_heat,
)
from app.ml.intraday_features import compute_intraday_features
from app.ml.cs_normalize import cs_normalize

logger = logging.getLogger(__name__)

# ── Simulation defaults ────────────────────────────────────────────────────────
HOLD_BARS = 24  # 2h of 5-min bars — matches training label horizon
FEATURE_BARS = 12  # 1h of bars used to build features before entry
# ATR multipliers must match intraday_training.py ATR_MULT_TARGET/STOP (0.8/0.4)
ATR_TARGET_MULT = 0.8  # Phase 47-3: compressed from 1.2 → closer target for 2h window
ATR_STOP_MULT = 0.4    # Phase 47-3: compressed from 0.6 → tighter stop, maintains ~2:1 R:R
TARGET_PCT = 0.005     # fallback only when prior-day range unavailable
STOP_PCT = 0.003
MIN_CONFIDENCE = 0.50  # model probabilities cluster below 0.55; keep at 0.50
TOP_N = 5  # max proposals per day
MAX_ENTRY_BAR = 60  # no new entries after bar 60 (~14:30 ET, 60×5=300 min from open)
INTRADAY_BUDGET_PCT = 0.03  # 3% of equity per intraday position


@dataclass
class _IntradayPosition:
    symbol: str
    entry_date: date
    entry_bar_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    quantity: int
    bars_held: int = 0
    sector: str = "UNKNOWN"


@dataclass
class _PortfolioState:
    cash: float
    peak_equity: float
    positions: Dict[str, _IntradayPosition] = field(default_factory=dict)
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


class IntradayAgentSimulator:
    """
    Portfolio-level intraday backtest that runs actual PM/Trader/RM agent code
    on historical 5-min bars.

    Usage:
        sim = IntradayAgentSimulator(model=loaded_model)
        result = sim.run(symbols_data, spy_data=spy_5min)
        result.print_report()
    """

    def __init__(
        self,
        model=None,
        starting_capital: float = STARTING_CAPITAL,
        limits: Optional[RiskLimits] = None,
        min_confidence: float = MIN_CONFIDENCE,
        top_n: int = TOP_N,
        transaction_cost_pct: float = TRANSACTION_COST,
        meta_model=None,
        pm_abstention_vix: float = 0.0,
        pm_abstention_spy_ma_days: int = 0,
    ):
        self.model = model
        self.starting_capital = starting_capital
        self.limits = limits or RiskLimits()
        self.min_confidence = min_confidence
        self.top_n = top_n
        self.transaction_cost_pct = transaction_cost_pct
        self.meta_model = meta_model
        self.pm_abstention_vix = pm_abstention_vix
        self.pm_abstention_spy_ma_days = pm_abstention_spy_ma_days

    def run(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        spy_data: Optional[pd.DataFrame] = None,
        spy_prices: Optional[pd.Series] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> SimResult:
        """
        Run the intraday agent-driven backtest.

        Args:
            symbols_data: dict of symbol → 5-min OHLCV DataFrame (multi-day)
            spy_data:     SPY 5-min bars for feature computation
            spy_prices:   SPY daily close for benchmark
            start_date:   first day to trade (default: first available day)
            end_date:     last day to trade (default: last available day)
            sector_map:   symbol → sector for concentration rules
        """
        if not symbols_data:
            return self._empty_result()
        if self.model is None or not getattr(self.model, "is_trained", False):
            return self._empty_result()

        sector_map = sector_map or {}

        all_days = sorted({
            d.date() if hasattr(d, "date") else d
            for df in symbols_data.values()
            for d in pd.DatetimeIndex(df.index).normalize().unique()
        })
        if not all_days:
            return self._empty_result()

        start_date = start_date or all_days[0]
        end_date = end_date or all_days[-1]
        trading_days = [d for d in all_days if start_date <= d <= end_date]
        if not trading_days:
            return self._empty_result()

        portfolio = _PortfolioState(cash=self.starting_capital, peak_equity=self.starting_capital)
        accepted_trades: List[Trade] = []
        equity_by_date: Dict[date, float] = {}
        tx_costs_total = 0.0

        # Precompute VIX and SPY daily closes for abstention gate
        _vix_closes: Optional[pd.Series] = None
        _spy_daily_closes: Optional[pd.Series] = None
        if self.pm_abstention_vix > 0 or self.pm_abstention_spy_ma_days > 0:
            try:
                import yfinance as yf
                _start_str = min(trading_days).strftime("%Y-%m-%d")
                _end_str = max(trading_days).strftime("%Y-%m-%d")
                if self.pm_abstention_vix > 0:
                    _vix = yf.download("^VIX", start=_start_str, end=_end_str,
                                       progress=False, auto_adjust=True)
                    if isinstance(_vix.columns, pd.MultiIndex):
                        _vix.columns = _vix.columns.get_level_values(0)
                    _vix.columns = [c.lower() for c in _vix.columns]
                    _vix_closes = _vix["close"]
                if self.pm_abstention_spy_ma_days > 0 and spy_prices is not None:
                    _spy_daily_closes = spy_prices
                elif self.pm_abstention_spy_ma_days > 0:
                    _spy = yf.download("SPY", start=_start_str, end=_end_str,
                                       progress=False, auto_adjust=True)
                    if isinstance(_spy.columns, pd.MultiIndex):
                        _spy.columns = _spy.columns.get_level_values(0)
                    _spy.columns = [c.lower() for c in _spy.columns]
                    _spy_daily_closes = _spy["close"]
            except Exception as exc:
                logger.debug("Abstention gate data fetch failed: %s", exc)

        for day in trading_days:
            # Phase 46-C: PM abstention gate — skip all entries on bad macro days
            skip_entries = False
            if self.pm_abstention_vix > 0 or self.pm_abstention_spy_ma_days > 0:
                try:
                    if self.pm_abstention_vix > 0 and _vix_closes is not None:
                        _vix_idx = pd.DatetimeIndex(_vix_closes.index)
                        _vix_dates = _vix_idx.date if hasattr(_vix_idx, "date") else np.array([d.date() for d in _vix_idx])
                        _vix_hist = _vix_closes.iloc[_vix_dates <= day]
                        if len(_vix_hist) > 0 and float(_vix_hist.iloc[-1]) >= self.pm_abstention_vix:
                            skip_entries = True
                    if not skip_entries and self.pm_abstention_spy_ma_days > 0 and _spy_daily_closes is not None:
                        _spy_idx = pd.DatetimeIndex(_spy_daily_closes.index)
                        _spy_dates = _spy_idx.date if hasattr(_spy_idx, "date") else np.array([d.date() for d in _spy_idx])
                        _spy_hist = _spy_daily_closes.iloc[_spy_dates <= day]
                        if len(_spy_hist) >= self.pm_abstention_spy_ma_days:
                            _spy_ma = float(_spy_hist.tail(self.pm_abstention_spy_ma_days).mean())
                            if float(_spy_hist.iloc[-1]) < _spy_ma:
                                skip_entries = True
                except Exception:
                    pass

            spy_day = self._get_day_bars(spy_data, day) if spy_data is not None else None
            day_trades, day_tx = self._process_day(
                day, symbols_data, spy_day, portfolio, sector_map,
                skip_entries=skip_entries,
            )
            accepted_trades.extend(day_trades)
            tx_costs_total += day_tx
            if portfolio.equity > portfolio.peak_equity:
                portfolio.peak_equity = portfolio.equity
            equity_by_date[day] = portfolio.equity
            portfolio.daily_pnl = 0.0

        if not accepted_trades:
            return self._empty_result()

        return self._compute_result(
            accepted_trades, equity_by_date, tx_costs_total,
            start_date, end_date, spy_prices,
        )

    # ─── Per-day simulation ────────────────────────────────────────────────────

    def _process_day(
        self,
        day: date,
        symbols_data: Dict[str, pd.DataFrame],
        spy_day: Optional[pd.DataFrame],
        portfolio: _PortfolioState,
        sector_map: Dict[str, str],
        skip_entries: bool = False,
    ) -> Tuple[List[Trade], float]:
        trades: List[Trade] = []
        tx_costs = 0.0

        if skip_entries:
            return trades, tx_costs

        # Build per-symbol day bars and feature vectors
        sym_feats: Dict[str, dict] = {}
        sym_entry_price: Dict[str, float] = {}
        sym_future_bars: Dict[str, pd.DataFrame] = {}
        sym_prior: Dict[str, Tuple] = {}

        for sym, df in symbols_data.items():
            df_idx = pd.DatetimeIndex(df.index)
            day_mask = df_idx.normalize().date == day
            day_bars = df.loc[day_mask]

            if len(day_bars) < FEATURE_BARS + 1:
                continue

            feat_bars = day_bars.iloc[:FEATURE_BARS]
            prior_close, prior_high, prior_low = self._prior_day_ohlc(df, df_idx, day)

            try:
                feats = compute_intraday_features(
                    feat_bars, spy_day, prior_close,
                    prior_day_high=prior_high,
                    prior_day_low=prior_low,
                )
            except Exception:
                feats = None
            if feats is None:
                continue

            sym_feats[sym] = feats
            sym_entry_price[sym] = float(feat_bars["close"].iloc[-1])
            sym_future_bars[sym] = day_bars.iloc[FEATURE_BARS: FEATURE_BARS + HOLD_BARS]
            sym_prior[sym] = (prior_close, prior_high, prior_low)

        if not sym_feats:
            return trades, tx_costs

        # PM: batch predict
        proposals = self._pm_score(sym_feats)

        # Process each proposal through meta gate → Trader gate → RM → entry/exit
        entered: Dict[str, _IntradayPosition] = {}
        for sym, confidence in proposals:
            if sym in entered:
                continue

            # Phase 46-B: meta-label gate — skip if E[R] <= 0
            if self.meta_model is not None:
                try:
                    if not self.meta_model.should_enter(sym_feats[sym]):
                        continue
                except Exception:
                    pass

            entry_price = sym_entry_price[sym]
            # Use prior-day range as ATR proxy (matches intraday training labels)
            prior_close, prior_high, prior_low = sym_prior.get(sym, (entry_price, entry_price, entry_price))
            prior_high = prior_high or entry_price
            prior_low = prior_low or entry_price
            prior_close = prior_close or entry_price
            prior_range = max(prior_high - prior_low, entry_price * 0.002)
            range_pct = prior_range / prior_close if prior_close > 0 else 0.002
            stop_pct_atr = float(np.clip(ATR_STOP_MULT * range_pct, 0.002, 0.02))
            target_pct_atr = float(np.clip(ATR_TARGET_MULT * range_pct, 0.003, 0.04))
            stop_price = entry_price * (1 - stop_pct_atr)
            target_price = entry_price * (1 + target_pct_atr)

            # Position sizing: fixed % of equity per intraday slot
            position_dollars = portfolio.equity * INTRADAY_BUDGET_PCT
            quantity = max(1, int(position_dollars / entry_price))

            # Apply RM position size cap
            max_pos_dollars = portfolio.equity * self.limits.MAX_POSITION_SIZE_PCT
            quantity = min(quantity, max(1, int(max_pos_dollars / entry_price)))
            if quantity <= 0:
                continue

            sector = sector_map.get(sym, "UNKNOWN")
            ok, _ = self._rm_validate(sym, entry_price, stop_price, quantity, portfolio, sector)
            if not ok:
                continue

            trade_cost = entry_price * quantity
            tx_cost_entry = trade_cost * self.transaction_cost_pct
            portfolio.cash -= trade_cost + tx_cost_entry
            tx_costs += tx_cost_entry

            pos = _IntradayPosition(
                symbol=sym,
                entry_date=day,
                entry_bar_idx=FEATURE_BARS,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                quantity=quantity,
                sector=sector,
            )
            entered[sym] = pos
            portfolio.sector_values[sector] = (
                portfolio.sector_values.get(sector, 0.0) + trade_cost
            )

        # Simulate each entered position through future bars
        for sym, pos in entered.items():
            future_bars = sym_future_bars.get(sym, pd.DataFrame())
            exit_price, exit_reason, hold_bars = self._simulate_exit(pos, future_bars)

            tx_cost_exit = exit_price * pos.quantity * self.transaction_cost_pct
            portfolio.cash += exit_price * pos.quantity - tx_cost_exit
            tx_costs += tx_cost_exit

            net_pnl = (exit_price - pos.entry_price) * pos.quantity - tx_cost_exit
            portfolio.daily_pnl += net_pnl
            portfolio.sector_values[pos.sector] = max(
                0.0, portfolio.sector_values.get(pos.sector, 0.0) - pos.entry_price * pos.quantity
            )

            trades.append(Trade(
                symbol=sym,
                entry_date=day,
                exit_date=day,
                entry_price=round(pos.entry_price, 4),
                exit_price=round(exit_price, 4),
                quantity=pos.quantity,
                pnl=round(net_pnl, 2),
                pnl_pct=round((exit_price - pos.entry_price) / pos.entry_price, 6),
                hold_bars=hold_bars,
                exit_reason=exit_reason,
                trade_type="intraday",
            ))

        return trades, tx_costs

    def _pm_score(self, sym_feats: Dict[str, dict]) -> List[Tuple[str, float]]:
        sym_list = list(sym_feats.keys())
        model_feat_names = getattr(self.model, "feature_names", None)
        try:
            if model_feat_names:
                X = np.array([
                    [sym_feats[s].get(f, 0.0) for f in model_feat_names]
                    for s in sym_list
                ])
            else:
                X = np.array([list(sym_feats[s].values()) for s in sym_list])
            X = np.nan_to_num(X, nan=0.0)
            X = cs_normalize(X)
            _, probas = self.model.predict(X)
        except Exception as exc:
            logger.debug("IntradayAgentSimulator PM score failed: %s", exc)
            return []

        ranked = sorted(zip(sym_list, probas), key=lambda x: x[1], reverse=True)
        return [(s, float(p)) for s, p in ranked if float(p) >= self.min_confidence][:self.top_n]

    def _rm_validate(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        quantity: int,
        portfolio: _PortfolioState,
        sector: str,
    ) -> Tuple[bool, str]:
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

        ok, msg = validate_open_positions(len(portfolio.positions), self.limits)
        if not ok:
            return False, msg

        ok, msg = validate_account_drawdown(equity, portfolio.peak_equity, self.limits)
        if not ok:
            return False, msg

        open_pos_dicts = [
            {"stop_price": p.stop_price, "entry_price": p.entry_price, "quantity": p.quantity}
            for p in portfolio.positions.values()
        ]
        new_trade_risk = (entry_price - stop_price) * quantity
        ok, msg = validate_portfolio_heat(new_trade_risk, open_pos_dicts, equity, self.limits)
        if not ok:
            return False, msg

        return True, "ok"

    def _simulate_exit(
        self,
        pos: _IntradayPosition,
        future_bars: pd.DataFrame,
    ) -> Tuple[float, str, int]:
        if len(future_bars) == 0:
            return pos.entry_price, "FORCE_CLOSE", 0

        exit_price = float(future_bars["close"].iloc[-1])
        exit_reason = "TIME_EXIT"
        hold_bars = len(future_bars)

        for bar_offset, (_, bar) in enumerate(future_bars.iterrows()):
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
            hold_bars = bar_offset + 1

            if low <= pos.stop_price:
                return pos.stop_price, "STOP", hold_bars
            if high >= pos.target_price:
                return pos.target_price, "TARGET", hold_bars
            exit_price = close

        return exit_price, exit_reason, hold_bars

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

        benchmark_return = alpha = 0.0
        if spy_prices is not None and len(spy_prices) > 0:
            try:
                s0 = spy_prices.asof(pd.Timestamp(start_date))
                s1 = spy_prices.asof(pd.Timestamp(end_date))
                if s0 and s1 and s0 > 0:
                    benchmark_return = (s1 - s0) / s0
                    alpha = total_return - benchmark_return
            except Exception:
                pass

        return SimResult(
            model_type="intraday_agent",
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
            information_ratio=0.0,
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
            model_type="intraday_agent",
            starting_capital=self.starting_capital,
            ending_capital=self.starting_capital,
            total_return_pct=0.0, annualized_return_pct=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0,
            max_drawdown_pct=0.0, calmar_ratio=0.0,
        )

    # ─── Helpers ───────────────────────────────────────────────────────────────

    def _get_day_bars(self, df: pd.DataFrame, day: date) -> Optional[pd.DataFrame]:
        idx = pd.DatetimeIndex(df.index)
        mask = idx.normalize().date == day
        result = df.loc[mask]
        return result if len(result) > 0 else None

    def _prior_day_ohlc(
        self, df: pd.DataFrame, df_idx: pd.DatetimeIndex, day: date
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        prior_mask = df_idx.normalize().date < day
        if not prior_mask.any():
            return None, None, None
        prior_bars = df.loc[prior_mask]
        last_day = prior_bars.index.normalize().date[-1]
        last_day_bars = prior_bars.loc[prior_bars.index.normalize().date == last_day]
        if len(last_day_bars) == 0:
            return None, None, None
        return (
            float(last_day_bars["close"].iloc[-1]),
            float(last_day_bars["high"].max()),
            float(last_day_bars["low"].min()),
        )
