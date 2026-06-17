"""Alpha-v9 P3-3 — overnight vs intraday decomposition tests.

Covers the exact decomposition/reconciliation, PIT correctness, the daily round-trip
cost, equal-weight universe aggregation, the cost cliff, and the frozen PASS/KILL verdict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research import overnight_intraday as oi


def _bars_from_legs(overnight_rets, intraday_rets, *, start_close=100.0, start="2015-01-02"):
    """Build OHLC bars that realize EXACTLY the given overnight/intraday per-day returns.
    open[t]=close[t-1]*(1+on[t]); close[t]=open[t]*(1+id[t]). Day 0 is a seed close only."""
    n = len(overnight_rets)
    idx = pd.bdate_range(start, periods=n + 1)
    opens, closes = [np.nan], [start_close]   # day 0: seed close, no open used
    for t in range(n):
        o = closes[-1] * (1.0 + overnight_rets[t])
        c = o * (1.0 + intraday_rets[t])
        opens.append(o)
        closes.append(c)
    return pd.DataFrame({"open": opens, "close": closes}, index=idx)


# ── decomposition / reconciliation ────────────────────────────────────────────
def test_decompose_reconciles_to_close_to_close():
    rng = np.random.default_rng(0)
    on = rng.normal(0.0003, 0.004, 300)
    idy = rng.normal(0.0001, 0.004, 300)
    bars = _bars_from_legs(on, idy)
    d = oi.decompose(bars)
    recon = (1 + d["overnight"]) * (1 + d["intraday"]) - 1.0
    assert np.allclose(recon.to_numpy(), d["close_to_close"].to_numpy(), atol=1e-12)


def test_decompose_drops_first_row_and_is_pit():
    bars = _bars_from_legs([0.01, -0.02, 0.03], [0.0, 0.0, 0.0])
    d = oi.decompose(bars)
    assert len(d) == 3                       # 4 bars -> 3 decomposed (first has no prior close)
    # overnight[t] must equal open[t]/close[t-1]-1 exactly (prior close, not same-day)
    assert d["overnight"].iloc[0] == pytest.approx(0.01)
    assert d["overnight"].iloc[1] == pytest.approx(-0.02)


def test_intraday_leg_uses_same_day_open_close():
    bars = _bars_from_legs([0.0, 0.0], [0.05, -0.04])
    d = oi.decompose(bars)
    assert d["intraday"].iloc[0] == pytest.approx(0.05)
    assert d["intraday"].iloc[1] == pytest.approx(-0.04)


# ── cost handling ─────────────────────────────────────────────────────────────
def test_round_trip_is_two_sides_per_day():
    assert oi._round_trip(1.0) == pytest.approx(2.0 / 1e4)
    assert oi._round_trip(0.0) == 0.0
    assert oi._round_trip(-5.0) == 0.0        # negative cost clamped to 0


def test_cost_cliff_is_monotonically_decreasing():
    rng = np.random.default_rng(1)
    on = rng.normal(0.0005, 0.004, 800)
    bars = _bars_from_legs(on, rng.normal(0.0, 0.004, 800))
    d = oi.decompose_universe({"SPY": bars}, cost_grid=(0.0, 1.0, 2.0, 5.0))
    cliff = d["cost_cliff"]
    vals = [cliff[c] for c in (0.0, 1.0, 2.0, 5.0)]
    assert vals == sorted(vals, reverse=True)   # higher cost -> lower net Sharpe


# ── equal-weight universe ─────────────────────────────────────────────────────
def test_equal_weight_legs_averages_symbols():
    a = _bars_from_legs([0.01, 0.01], [0.0, 0.0])
    b = _bars_from_legs([0.03, 0.03], [0.0, 0.0])
    legs = oi.equal_weight_legs({"A": a, "B": b})
    assert legs["overnight"].iloc[0] == pytest.approx(0.02)   # mean(0.01, 0.03)


def test_equal_weight_handles_ragged_history():
    a = _bars_from_legs([0.01, 0.01, 0.01], [0.0, 0.0, 0.0])
    b = _bars_from_legs([0.05, 0.05], [0.0, 0.0], start="2015-01-05")  # starts later
    legs = oi.equal_weight_legs({"A": a, "B": b})
    assert not legs.empty   # union index, mean over available symbols each day


# ── frozen verdict ────────────────────────────────────────────────────────────
def test_verdict_pass_on_strong_cheap_overnight():
    rng = np.random.default_rng(7)
    on = rng.normal(0.0006, 0.004, 1500)         # strong overnight drift
    idy = rng.normal(0.0, 0.004, 1500)           # ~zero intraday
    bars = _bars_from_legs(on, idy)
    v = oi.overnight_intraday_verdict({"SPY": bars})
    assert v.verdict == "PASS"
    assert v.overnight.net_sharpe >= oi.PAPER_SR_FLOOR
    assert v.overnight.net_cagr > 0
    assert v.overnight.net_sharpe > v.intraday.net_sharpe


def test_verdict_kill_when_cost_erases_premium():
    rng = np.random.default_rng(3)
    on = rng.normal(0.0001, 0.004, 1500)         # tiny premium < daily round-trip (2bps)
    idy = rng.normal(0.0, 0.004, 1500)
    bars = _bars_from_legs(on, idy)
    v = oi.overnight_intraday_verdict({"SPY": bars})
    assert v.verdict == "KILL"
    assert v.overnight.net_cagr <= 0 or v.overnight.net_sharpe < oi.PAPER_SR_FLOOR


def test_verdict_kill_when_intraday_dominates():
    rng = np.random.default_rng(5)
    on = rng.normal(0.0006, 0.004, 1500)
    idy = rng.normal(0.0012, 0.004, 1500)        # intraday stronger than overnight
    bars = _bars_from_legs(on, idy)
    v = oi.overnight_intraday_verdict({"SPY": bars})
    assert v.verdict == "KILL"
    assert "intraday" in v.reason
    assert v.overnight.net_sharpe <= v.intraday.net_sharpe


def test_verdict_carries_registration_id_and_grid():
    bars = _bars_from_legs([0.001] * 400, [0.0] * 400)
    v = oi.overnight_intraday_verdict({"SPY": bars})
    assert v.registration_id == "P3-3-OVERNIGHT-INTRADAY"
    assert set(v.cost_cliff.keys()) == set(oi.DEFAULT_COST_GRID)
    assert "SPY" in v.per_symbol
