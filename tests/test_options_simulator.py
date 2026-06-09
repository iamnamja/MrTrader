"""OPT-2: contract-level options simulator + cost model — golden-path P&L.

Hand-computed payoffs (long call to expiry, vertical cap at strike width, short-put
assignment), cost-sweep monotonicity, daily-MTM equity curve, and contract conformance.
These are what make an options backtest's P&L trustworthy.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.options.contracts import OptionsSpreadCostModel as CostProto, OptionContractSim
from app.backtesting.options_simulator import (
    OptionsSimulator, OptionsSpreadCostModel, OptionLeg, OptionPosition,
    daily_returns_dated, MULTIPLIER,
)
from app.data.options_provider import BARS_COLS

# Contracts (expire 2025-12-19)
C100 = "O:SPY251219C00100000"
C110 = "O:SPY251219C00110000"
P100 = "O:SPY251219P00100000"


def _bars(rows):
    """rows: list of (contract, 'YYYY-MM-DD', close)."""
    recs = [{"underlying": "SPY", "contract": c, "date": pd.Timestamp(d),
             "open": v, "high": v, "low": v, "close": v, "volume": 100.0,
             "knowable_date": pd.Timestamp(d)} for c, d, v in rows]
    return pd.DataFrame(recs, columns=BARS_COLS)


# ── Cost model ────────────────────────────────────────────────────────────────

def test_cost_model_scales_with_premium_and_mult():
    cm = OptionsSpreadCostModel(spread_pct=0.01, per_contract_fee=0.65)
    # premium 5.0: 5*0.01*100 = 5.0 spread + 0.65 fee
    assert cm.entry_exit_cost(5.0, 1.0) == pytest.approx(5.65)
    assert cm.entry_exit_cost(5.0, 2.0) == pytest.approx(10.65)  # spread doubles, fee flat
    assert cm.entry_exit_cost(5.0, 3.0) == pytest.approx(15.65)


def test_cost_model_satisfies_contract():
    assert isinstance(OptionsSpreadCostModel(), CostProto)


# ── Golden-path P&L ───────────────────────────────────────────────────────────

def test_long_call_to_expiry_hand_computed():
    # Buy 1 call K=100 at 5.0 on 12-01; underlying 110 at expiry -> intrinsic 10.
    # Gross P&L = (10 - 5) * 100 = 500. Entry cost (1×) = 5*0.01*100 + 0.65 = 5.65.
    # No exit cost (held to expiry). Net = 500 - 5.65 = 494.35.
    bars = _bars([(C100, "2025-12-01", 5.0)])
    und = {"SPY": {date(2025, 12, 19): 110.0}}
    sim = OptionsSimulator(bars, underlying_prices=und, starting_capital=100_000)
    pos = OptionPosition([OptionLeg(C100, +1, 1)], entry_date=date(2025, 12, 1))
    res = sim.run([pos], date(2025, 12, 1), date(2025, 12, 19))
    assert res.ending_capital == pytest.approx(100_000 + 494.35, abs=1e-6)
    assert res.total_trades == 1 and res.win_rate == 1.0


def test_otm_long_call_expires_worthless():
    # Underlying 90 at expiry < K=100 -> intrinsic 0. Lose premium + entry cost.
    bars = _bars([(C100, "2025-12-01", 5.0)])
    und = {"SPY": {date(2025, 12, 19): 90.0}}
    sim = OptionsSimulator(bars, underlying_prices=und, starting_capital=100_000)
    pos = OptionPosition([OptionLeg(C100, +1, 1)], entry_date=date(2025, 12, 1))
    res = sim.run([pos], date(2025, 12, 1), date(2025, 12, 19))
    # P&L = (0-5)*100 - 5.65 = -505.65
    assert res.ending_capital == pytest.approx(100_000 - 505.65, abs=1e-6)


def test_vertical_spread_caps_at_strike_width():
    # Long 100C @5, short 110C @2 (net debit 3). Underlying 130 at expiry.
    # Long intrinsic 30, short intrinsic -20 -> net 10 (= strike width), capped.
    # Gross = (30-5)*100 + (-1)*(20-2)*100 = 2500 - 1800 = 700 = (10 - 3 debit)*100. ✓
    bars = _bars([(C100, "2025-12-01", 5.0), (C110, "2025-12-01", 2.0)])
    und = {"SPY": {date(2025, 12, 19): 130.0}}
    sim = OptionsSimulator(bars, underlying_prices=und, starting_capital=100_000)
    pos = OptionPosition([OptionLeg(C100, +1, 1), OptionLeg(C110, -1, 1)],
                         entry_date=date(2025, 12, 1))
    res = sim.run([pos], date(2025, 12, 1), date(2025, 12, 19))
    gross = res.ending_capital - 100_000 + res.trades[0].pnl * 0  # costs inside pnl
    # net of entry costs on both legs (5.65 + 2.65 = 8.30):
    assert (res.ending_capital - 100_000) == pytest.approx(700 - (5.65 + 2.65), abs=1e-6)
    assert gross is not None


def test_vertical_spread_max_loss_bounded():
    # Same spread, underlying 90 at expiry -> both calls worthless. Loss = net debit only.
    bars = _bars([(C100, "2025-12-01", 5.0), (C110, "2025-12-01", 2.0)])
    und = {"SPY": {date(2025, 12, 19): 90.0}}
    sim = OptionsSimulator(bars, underlying_prices=und, starting_capital=100_000)
    pos = OptionPosition([OptionLeg(C100, +1, 1), OptionLeg(C110, -1, 1)],
                         entry_date=date(2025, 12, 1))
    res = sim.run([pos], date(2025, 12, 1), date(2025, 12, 19))
    # Gross = (0-5)*100 + (-1)*(0-2)*100 = -500 + 200 = -300 (= -net debit*100).
    assert (res.ending_capital - 100_000) == pytest.approx(-300 - (5.65 + 2.65), abs=1e-6)


def test_short_put_assignment_at_expiry():
    # Sell 1 put K=100 @4 on 12-01; underlying 90 at expiry -> intrinsic 10.
    # Short P&L = (-1)*(10 - 4)*100 = -600. Entry cost = 4*0.01*100+0.65 = 4.65.
    bars = _bars([(P100, "2025-12-01", 4.0)])
    und = {"SPY": {date(2025, 12, 19): 90.0}}
    sim = OptionsSimulator(bars, underlying_prices=und, starting_capital=100_000)
    pos = OptionPosition([OptionLeg(P100, -1, 1)], entry_date=date(2025, 12, 1))
    res = sim.run([pos], date(2025, 12, 1), date(2025, 12, 19))
    assert (res.ending_capital - 100_000) == pytest.approx(-600 - 4.65, abs=1e-6)


def test_short_put_expires_otm_keeps_premium():
    # Underlying 110 at expiry > K=100 -> put worthless. Short keeps premium minus cost.
    bars = _bars([(P100, "2025-12-01", 4.0)])
    und = {"SPY": {date(2025, 12, 19): 110.0}}
    sim = OptionsSimulator(bars, underlying_prices=und, starting_capital=100_000)
    pos = OptionPosition([OptionLeg(P100, -1, 1)], entry_date=date(2025, 12, 1))
    res = sim.run([pos], date(2025, 12, 1), date(2025, 12, 19))
    # (-1)*(0-4)*100 = +400, minus 4.65 entry = 395.35
    assert (res.ending_capital - 100_000) == pytest.approx(395.35, abs=1e-6)


# ── Cost sweep monotonicity (the mandatory stress knob) ────────────────────────

def test_cost_sweep_monotonic():
    bars = _bars([(C100, "2025-12-01", 5.0)])
    und = {"SPY": {date(2025, 12, 19): 110.0}}
    sim = OptionsSimulator(bars, underlying_prices=und, starting_capital=100_000)
    pos = OptionPosition([OptionLeg(C100, +1, 1)], entry_date=date(2025, 12, 1))
    e1 = sim.run([pos], date(2025, 12, 1), date(2025, 12, 19), spread_mult=1.0).ending_capital
    e2 = sim.run([pos], date(2025, 12, 1), date(2025, 12, 19), spread_mult=2.0).ending_capital
    e3 = sim.run([pos], date(2025, 12, 1), date(2025, 12, 19), spread_mult=3.0).ending_capital
    assert e1 > e2 > e3  # higher modeled spread -> strictly more cost -> lower equity


# ── Daily MTM + early close ───────────────────────────────────────────────────

def test_daily_mtm_equity_curve_and_close_before_expiry():
    # Mark daily off real closes, then CLOSE at market on 12-10 (before expiry) -> pays
    # exit cost; equity tracks the close path.
    bars = _bars([(C100, "2025-12-01", 5.0), (C100, "2025-12-05", 7.0),
                  (C100, "2025-12-10", 8.0)])
    sim = OptionsSimulator(bars, starting_capital=100_000)
    pos = OptionPosition([OptionLeg(C100, +1, 1)], entry_date=date(2025, 12, 1),
                         exit_date=date(2025, 12, 10))
    res = sim.run([pos], date(2025, 12, 1), date(2025, 12, 10))
    curve = dict(res.equity_curve)
    # 12-05 unrealized = (7-5)*100 = +200, minus entry cost 5.65
    assert curve[date(2025, 12, 5)] == pytest.approx(100_000 + 200 - 5.65, abs=1e-6)
    # closed 12-10: gross (8-5)*100=300, entry 5.65 + exit (8*0.01*100+0.65=8.65)
    assert res.ending_capital == pytest.approx(100_000 + 300 - 5.65 - 8.65, abs=1e-6)
    dr = daily_returns_dated(res)
    assert set(dr).issubset(set(curve))


def test_simulator_satisfies_contract():
    assert isinstance(OptionsSimulator(_bars([(C100, "2025-12-01", 5.0)])), OptionContractSim)
    assert MULTIPLIER == 100


# ── Review-driven regressions ──────────────────────────────────────────────────

def test_multi_day_forward_fill_mark():
    # No bar on 12-05 (only 12-01 and 12-08). The open mark on 12-05 must forward-fill the
    # 12-01 close (not jump or vanish).
    bars = _bars([(C100, "2025-12-01", 5.0), (C100, "2025-12-08", 9.0)])
    und = {"SPY": {date(2025, 12, 5): 0.0}}  # adds 12-05 to the calendar
    sim = OptionsSimulator(bars, underlying_prices=und, starting_capital=100_000)
    pos = OptionPosition([OptionLeg(C100, +1, 1)], entry_date=date(2025, 12, 1),
                         exit_date=date(2025, 12, 8))
    res = sim.run([pos], date(2025, 12, 1), date(2025, 12, 8))
    curve = dict(res.equity_curve)
    # 12-05 forward-fills 5.0 -> unrealized 0, minus entry cost 5.65
    assert curve[date(2025, 12, 5)] == pytest.approx(100_000 - 5.65, abs=1e-6)


def test_intermediate_short_mtm_sign():
    # Short call: when the option's price RISES intraperiod, the short is at a LOSS.
    bars = _bars([(C100, "2025-12-01", 4.0), (C100, "2025-12-05", 7.0)])
    sim = OptionsSimulator(bars, starting_capital=100_000)
    pos = OptionPosition([OptionLeg(C100, -1, 1)], entry_date=date(2025, 12, 1),
                         exit_date=date(2025, 12, 5))
    res = sim.run([pos], date(2025, 12, 1), date(2025, 12, 5))
    curve = dict(res.equity_curve)
    # day-1 unrealized for a short with flat price = 0; entry cost 4*0.01*100+0.65=4.65
    assert curve[date(2025, 12, 1)] == pytest.approx(100_000 - 4.65, abs=1e-6)
    # closed 12-05: short loss (-1)*(7-4)*100 = -300; +exit cost 7*0.01*100+0.65=7.65
    assert res.ending_capital == pytest.approx(100_000 - 300 - 4.65 - 7.65, abs=1e-6)


def test_qty_scaling():
    # 3 contracts -> 3× the P&L and 3× the per-contract cost.
    bars = _bars([(C100, "2025-12-01", 5.0)])
    und = {"SPY": {date(2025, 12, 19): 110.0}}
    sim = OptionsSimulator(bars, underlying_prices=und, starting_capital=100_000)
    pos = OptionPosition([OptionLeg(C100, +1, 3)], entry_date=date(2025, 12, 1))
    res = sim.run([pos], date(2025, 12, 1), date(2025, 12, 19))
    # gross (10-5)*100*3 = 1500; entry cost 5.65*3 = 16.95
    assert (res.ending_capital - 100_000) == pytest.approx(1500 - 16.95, abs=1e-6)


def test_calendar_spread_differing_expirations():
    # Long near 100C (exp 12-19) + long far 100C (exp 2026-01-16). Near settles at its own
    # expiry intrinsic, far at its own — value-correct total.
    NEAR = "O:SPY251219C00100000"
    FAR = "O:SPY260116C00100000"
    bars = _bars([(NEAR, "2025-12-01", 5.0), (FAR, "2025-12-01", 8.0)])
    und = {"SPY": {date(2025, 12, 19): 108.0, date(2026, 1, 16): 120.0}}
    sim = OptionsSimulator(bars, underlying_prices=und, starting_capital=100_000)
    pos = OptionPosition([OptionLeg(NEAR, +1, 1), OptionLeg(FAR, +1, 1)],
                         entry_date=date(2025, 12, 1))
    res = sim.run([pos], date(2025, 12, 1), date(2026, 1, 16))
    # near (8-5)*100=300 ; far (20-8)*100=1200 ; entry costs 5.65+8.65=14.30
    assert (res.ending_capital - 100_000) == pytest.approx(300 + 1200 - 14.30, abs=1e-6)


def test_missing_entry_price_dropped_and_counted():
    bars = _bars([(C100, "2025-12-01", 5.0)])
    und = {"SPY": {date(2025, 12, 19): 110.0}}
    sim = OptionsSimulator(bars, underlying_prices=und, starting_capital=100_000)
    good = OptionPosition([OptionLeg(C100, +1, 1)], entry_date=date(2025, 12, 1))
    bad = OptionPosition([OptionLeg(C110, +1, 1)], entry_date=date(2025, 12, 1))  # no bar
    res = sim.run([good, bad], date(2025, 12, 1), date(2025, 12, 19))
    assert res.total_trades == 1
    assert res.dropped_positions == 1  # the un-enterable one is dropped AND counted


def test_unparseable_contract_dropped():
    bars = _bars([("NOT_AN_OCC_TICKER", "2025-12-01", 5.0)])
    sim = OptionsSimulator(bars, starting_capital=100_000)
    pos = OptionPosition([OptionLeg("NOT_AN_OCC_TICKER", +1, 1)],
                         entry_date=date(2025, 12, 1))
    res = sim.run([pos], date(2025, 12, 1), date(2025, 12, 19))
    assert res.total_trades == 0 and res.dropped_positions == 1


def test_profit_factor_capped_no_losses():
    # All-wins fold -> PF must be a finite sentinel, not inf (which poisons gates).
    bars = _bars([(C100, "2025-12-01", 5.0)])
    und = {"SPY": {date(2025, 12, 19): 110.0}}
    sim = OptionsSimulator(bars, underlying_prices=und, starting_capital=100_000)
    pos = OptionPosition([OptionLeg(C100, +1, 1)], entry_date=date(2025, 12, 1))
    res = sim.run([pos], date(2025, 12, 1), date(2025, 12, 19))
    import math as _m
    assert _m.isfinite(res.profit_factor) and res.profit_factor == pytest.approx(99.0)


def test_blown_up_flag_on_negative_equity():
    # A naked short with a huge adverse move drives equity < 0 -> blown_up flagged.
    bars = _bars([(C100, "2025-12-01", 5.0)])
    und = {"SPY": {date(2025, 12, 19): 2000.0}}  # call settles ~1900 intrinsic
    sim = OptionsSimulator(bars, underlying_prices=und, starting_capital=100_000)
    pos = OptionPosition([OptionLeg(C100, -1, 1)], entry_date=date(2025, 12, 1))
    res = sim.run([pos], date(2025, 12, 1), date(2025, 12, 19))
    assert res.ending_capital < 0 and res.blown_up is True
