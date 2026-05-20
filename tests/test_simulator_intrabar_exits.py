"""
Tests for AgentSimulator intrabar stop/target exit logic (P0.2 fix) and
short position P&L accounting.

Covers:
- Long stop hit intraday (low <= stop) → fill at stop_price, reason "STOP_HIT"
- Long target hit intraday (high >= target) → fill at target_price, reason "TARGET"
- Long gap-down through stop (open <= stop) → fill at open
- Long gap-up through target (open >= target) → fill at open
- Short stop hit intraday (high >= stop) → fill at stop_price
- Short target hit intraday (low <= target) → fill at target_price
- Short gap-up through stop → fill at open
- Short gap-down through target → fill at open
- No look-ahead: trailing stop ratchet does NOT fire intrabar check on same bar
- Short MTM equity: (entry - close) * qty contributes correctly
- Short PnL sign: profit when price falls, loss when price rises
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date
from unittest.mock import MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

_DAY = date(2023, 1, 4)
_DAY_STR = "2023-01-04"


def _single_bar(open_: float, high: float, low: float, close: float,
                day: str = _DAY_STR) -> pd.DataFrame:
    """One-row OHLCV DataFrame indexed by the given date."""
    return pd.DataFrame(
        [{"open": open_, "high": high, "low": low, "close": close, "volume": 1_000_000}],
        index=pd.DatetimeIndex([day]),
    )


def _make_sim():
    from app.backtesting.agent_simulator import AgentSimulator
    from app.agents.risk_rules import RiskLimits
    return AgentSimulator(
        model=None,
        starting_capital=50_000.0,
        limits=RiskLimits(MAX_OPEN_POSITIONS=5),
    )


def _open_position(sim, symbol: str, entry: float, stop: float, target: float,
                   qty: int = 10, direction: str = "long"):
    """Inject a position into a fresh _PortfolioState and return (portfolio, pos)."""
    from app.backtesting.agent_simulator import _Position, _PortfolioState
    pos = _Position(
        symbol=symbol,
        entry_date=date(2023, 1, 2),
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        quantity=qty,
        highest_price=entry,
        bars_held=2,
        direction=direction,
    )
    portfolio = _PortfolioState(
        cash=sim.starting_capital - entry * qty,
        peak_equity=sim.starting_capital,
        positions={symbol: pos},
    )
    return portfolio, pos


def _exit(sim, direction: str, entry: float, stop: float, target: float,
          bar_open: float, bar_high: float, bar_low: float, bar_close: float):
    """Run _process_exits with one position and one bar. Returns (trade, fill) or None."""
    sym = "TEST"
    portfolio, _ = _open_position(sim, sym, entry, stop, target, direction=direction)
    bars_df = _single_bar(bar_open, bar_high, bar_low, bar_close)
    closed = sim._process_exits(_DAY, {sym: bars_df}, portfolio)
    if not closed:
        return None
    trade, tx = closed[0]
    return trade


# ── Long intrabar exits ────────────────────────────────────────────────────────

class TestLongIntrabarStop:
    def test_low_touches_stop_fills_at_stop(self):
        sim = _make_sim()
        # low=94 < stop=95: should fill at stop (95), not close (97)
        trade = _exit(sim, "long", entry=100, stop=95, target=110,
                      bar_open=99, bar_high=100, bar_low=94, bar_close=97)
        assert trade is not None
        assert trade.exit_price == pytest.approx(95.0), "should fill at stop_price"
        assert "stop" in trade.exit_reason.lower()

    def test_gap_down_through_stop_fills_at_open(self):
        sim = _make_sim()
        # open=93 < stop=95: gap-down → fill at open
        trade = _exit(sim, "long", entry=100, stop=95, target=110,
                      bar_open=93, bar_high=94, bar_low=92, bar_close=93)
        assert trade is not None
        assert trade.exit_price == pytest.approx(93.0), "gap-down: fill at open"
        assert "stop" in trade.exit_reason.lower()

    def test_high_touches_target_fills_at_target(self):
        sim = _make_sim()
        # high=111 >= target=110: fill at target (110), not close (108)
        trade = _exit(sim, "long", entry=100, stop=95, target=110,
                      bar_open=101, bar_high=111, bar_low=100, bar_close=108)
        assert trade is not None
        assert trade.exit_price == pytest.approx(110.0), "should fill at target_price"
        assert "target" in trade.exit_reason.lower()

    def test_gap_up_through_target_fills_at_open(self):
        sim = _make_sim()
        # open=112 >= target=110: gap-up → fill at open (better than target)
        trade = _exit(sim, "long", entry=100, stop=95, target=110,
                      bar_open=112, bar_high=114, bar_low=111, bar_close=113)
        assert trade is not None
        assert trade.exit_price == pytest.approx(112.0), "gap-up: fill at open"
        assert "target" in trade.exit_reason.lower()

    def test_normal_bar_no_exit(self):
        sim = _make_sim()
        # Within range: low=98 > stop=95, high=102 < target=110
        trade = _exit(sim, "long", entry=100, stop=95, target=110,
                      bar_open=100, bar_high=102, bar_low=98, bar_close=101)
        assert trade is None


# ── Short intrabar exits ──────────────────────────────────────────────────────

class TestShortIntrabarStop:
    def test_high_touches_stop_fills_at_stop(self):
        sim = _make_sim()
        # Short: stop above entry (210), high=211 → cover at stop (210)
        trade = _exit(sim, "short", entry=200, stop=210, target=185,
                      bar_open=202, bar_high=211, bar_low=199, bar_close=205)
        assert trade is not None
        assert trade.exit_price == pytest.approx(210.0), "short stop: fill at stop"
        assert "stop" in trade.exit_reason.lower()

    def test_gap_up_through_stop_fills_at_open(self):
        sim = _make_sim()
        # Short: open=215 >= stop=210 → gap-up: fill at open (worse than stop)
        trade = _exit(sim, "short", entry=200, stop=210, target=185,
                      bar_open=215, bar_high=217, bar_low=213, bar_close=215)
        assert trade is not None
        assert trade.exit_price == pytest.approx(215.0), "short gap-up: fill at open"
        assert "stop" in trade.exit_reason.lower()

    def test_low_touches_target_fills_at_target(self):
        sim = _make_sim()
        # Short: target below entry (185), low=184 → cover at target (185)
        trade = _exit(sim, "short", entry=200, stop=210, target=185,
                      bar_open=198, bar_high=199, bar_low=184, bar_close=187)
        assert trade is not None
        assert trade.exit_price == pytest.approx(185.0), "short target: fill at target"
        assert "target" in trade.exit_reason.lower()

    def test_gap_down_through_target_fills_at_open(self):
        sim = _make_sim()
        # Short: open=182 <= target=185 → gap-down: fill at open (better than target)
        trade = _exit(sim, "short", entry=200, stop=210, target=185,
                      bar_open=182, bar_high=183, bar_low=181, bar_close=182)
        assert trade is not None
        assert trade.exit_price == pytest.approx(182.0), "short gap-down: fill at open"
        assert "target" in trade.exit_reason.lower()

    def test_normal_bar_no_exit(self):
        sim = _make_sim()
        # Short within range: high=202 < stop=210, low=195 > target=185
        trade = _exit(sim, "short", entry=200, stop=210, target=185,
                      bar_open=197, bar_high=202, bar_low=195, bar_close=198)
        assert trade is None


# ── Short P&L sign ────────────────────────────────────────────────────────────

class TestShortPnL:
    def test_short_profit_when_price_falls(self):
        """Short exit below entry → positive PnL = (entry - exit) * qty."""
        sim = _make_sim()
        # Target hit at 185: PnL = (200 - 185) * 10 = 150
        trade = _exit(sim, "short", entry=200, stop=210, target=185,
                      bar_open=195, bar_high=196, bar_low=184, bar_close=187)
        assert trade is not None
        assert trade.pnl > 0, "short profit should be positive when price falls"
        assert trade.pnl == pytest.approx(150.0, abs=5.0)

    def test_short_loss_when_price_rises(self):
        """Short exit above entry (stop) → negative PnL = (entry - stop) * qty."""
        sim = _make_sim()
        # Stop hit at 210: PnL = (200 - 210) * 10 = -100
        trade = _exit(sim, "short", entry=200, stop=210, target=185,
                      bar_open=202, bar_high=211, bar_low=201, bar_close=209)
        assert trade is not None
        assert trade.pnl < 0, "short loss should be negative when price rises"
        assert trade.pnl == pytest.approx(-100.0, abs=5.0)


# ── Short MTM equity ─────────────────────────────────────────────────────────

class TestShortMTMEquity:
    def test_mtm_profit_when_price_below_entry(self):
        """Unrealized short gain: close < entry → positive MTM contribution."""
        from app.backtesting.agent_simulator import _Position, _PortfolioState
        portfolio = _PortfolioState(
            cash=50_000.0, peak_equity=50_000.0,
            positions={
                "TSLA": _Position(
                    symbol="TSLA", entry_date=date(2023, 1, 2),
                    entry_price=200.0, stop_price=210.0, target_price=180.0,
                    quantity=10, highest_price=200.0, direction="short",
                )
            },
        )
        # close=190 → short MTM = (200 - 190) * 10 = 100 profit
        equity = portfolio.equity_mtm({"TSLA": 190.0})
        assert equity == pytest.approx(50_000.0 + 100.0)

    def test_mtm_loss_when_price_above_entry(self):
        """Unrealized short loss: close > entry → negative MTM contribution."""
        from app.backtesting.agent_simulator import _Position, _PortfolioState
        portfolio = _PortfolioState(
            cash=50_000.0, peak_equity=50_000.0,
            positions={
                "TSLA": _Position(
                    symbol="TSLA", entry_date=date(2023, 1, 2),
                    entry_price=200.0, stop_price=210.0, target_price=180.0,
                    quantity=10, highest_price=200.0, direction="short",
                )
            },
        )
        # close=210 → short MTM = (200 - 210) * 10 = -100 loss
        equity = portfolio.equity_mtm({"TSLA": 210.0})
        assert equity == pytest.approx(50_000.0 - 100.0)


# ── No look-ahead: trailing stop ratchet must not retroactively fire ──────────

class TestNoLookahead:
    def test_trailing_ratchet_does_not_exit_on_same_bar(self):
        """
        Today's H/L update drives trailing-stop ratchet, but that ratchet
        must NOT be used for the intrabar breach check on the SAME bar.

        Long: stop=90, target=115. Today: high=108 (would ratchet stop up),
        low=96 (above original stop=90 but might be <= ratcheted stop).
        Position should REMAIN OPEN — the intrabar check uses the pre-bar stop (90).
        """
        sim = _make_sim()
        sym = "AAPL"
        portfolio, pos = _open_position(sim, sym, entry=100, stop=90, target=115)
        pos.highest_price = 100.0  # yesterday's peak

        bars_df = _single_bar(open_=101, high=108, low=96, close=104)
        closed = sim._process_exits(_DAY, {sym: bars_df}, portfolio)

        assert len(closed) == 0, (
            "Trailing-stop ratchet from today's high must not retroactively "
            "trigger an intrabar stop breach on the same bar"
        )
        assert sym in portfolio.positions
