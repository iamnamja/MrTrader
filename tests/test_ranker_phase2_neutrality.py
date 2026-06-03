"""Phase 2 (§3.1) — dollar-neutral-at-target-gross tests.

Covers the two fixes that make the L/S book actually neutral AND funded to gross:
  A. `_rebalance_resize_position` re-sizes HELD names to the per-leg budget each
     rebalance. Verifies the cash/short_collateral/qty accounting is MTM-neutral
     except the transaction cost on the traded share delta (long add/sell, short
     add/cover) — so holding both legs at budget keeps net $ ≈ 0, gross at target.
  B. `apply_net_sector_cap` breadth pass fills the short book toward n_target even
     when the loser tail concentrates in sectors the longs don't occupy (preserving
     the breadth the thesis needs), instead of starving the count.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.backtesting.agent_simulator import AgentSimulator, _PortfolioState, _Position
from app.strategy.portfolio_construction import apply_net_sector_cap


def _sim(tx=0.0005):
    return AgentSimulator(model=MagicMock(), starting_capital=100_000.0,
                          transaction_cost_pct=tx)


def _pos(direction, qty, entry):
    return _Position(symbol="X", entry_date=None, entry_price=entry, stop_price=0.0,
                     target_price=0.0, quantity=qty, highest_price=entry, direction=direction)


# ── A. resize accounting is MTM-neutral except tx ────────────────────────────────

@pytest.mark.parametrize("direction,q0,target_dollars,expect_q1", [
    ("long", 100, 7500.0, 150),   # buy more
    ("long", 150, 5000.0, 100),   # sell part
    ("short", 100, 7500.0, 150),  # add to short
    ("short", 150, 5000.0, 100),  # partial cover
])
def test_resize_is_mtm_neutral_except_tx(direction, q0, target_dollars, expect_q1):
    sim = _sim(tx=0.0005)
    price = 50.0
    pos = _pos(direction, q0, price)
    coll0 = price * q0 if direction == "short" else 0.0
    pf = _PortfolioState(cash=100_000.0, peak_equity=100_000.0, short_collateral=coll0)
    pf.positions["X"] = pos
    closes = {"X": price}  # mark at the resize price (fair value)

    eq_before = pf.equity_mtm(closes)
    cost = sim._rebalance_resize_position(pos, target_dollars, price, pf)
    eq_after = pf.equity_mtm(closes)

    assert pos.quantity == expect_q1
    assert cost > 0.0
    # A fair-value resize moves cash<->position; equity only drops by the tx cost.
    assert abs((eq_before - eq_after) - cost) < 1e-6
    if direction == "short":
        assert abs(pf.short_collateral - price * expect_q1) < 1e-6  # collateral tracks entry×qty


def test_short_collateral_invariant_across_blended_add_then_cover():
    # open@50 -> add@60 (blends entry) -> cover@55. The per-position invariant
    # short_collateral == entry_price * qty must hold after each step (the one path
    # a basis-vs-price confusion in the release would silently break).
    sim = _sim(tx=0.0)  # isolate the collateral invariant from tx
    pos = _pos("short", 100, 50.0)
    pf = _PortfolioState(cash=100_000.0, peak_equity=100_000.0, short_collateral=50.0 * 100)
    pf.positions["X"] = pos
    sim._rebalance_resize_position(pos, 150 * 60.0, 60.0, pf)   # add 50 @ 60
    assert pos.quantity == 150
    assert abs(pos.entry_price - (50 * 100 + 60 * 50) / 150) < 1e-9  # blended
    assert abs(pf.short_collateral - pos.entry_price * pos.quantity) < 1e-6
    sim._rebalance_resize_position(pos, 100 * 55.0, 55.0, pf)   # cover 50 @ 55
    assert pos.quantity == 100
    assert abs(pf.short_collateral - pos.entry_price * pos.quantity) < 1e-6


def test_resize_noop_when_already_at_target():
    sim = _sim()
    pos = _pos("long", 100, 50.0)
    pf = _PortfolioState(cash=100_000.0, peak_equity=100_000.0)
    pf.positions["X"] = pos
    assert sim._rebalance_resize_position(pos, 5000.0, 50.0, pf) == 0.0  # 100*50=5000 already
    assert pos.quantity == 100


def test_resize_never_flattens_to_zero():
    sim = _sim()
    pos = _pos("short", 100, 50.0)
    pf = _PortfolioState(cash=100_000.0, peak_equity=100_000.0, short_collateral=5000.0)
    pf.positions["X"] = pos
    # tiny target -> target_qty 0 -> resize must NOT flatten (to_drop owns exits)
    assert sim._rebalance_resize_position(pos, 10.0, 50.0, pf) == 0.0
    assert pos.quantity == 100


def test_resize_long_add_respects_effective_cash():
    sim = _sim()
    pos = _pos("long", 100, 50.0)
    # almost no free cash (cash tied up as short collateral) -> add must be skipped
    pf = _PortfolioState(cash=5_000.0, peak_equity=100_000.0, short_collateral=5_000.0)
    pf.positions["X"] = pos
    cost = sim._rebalance_resize_position(pos, 50_000.0, 50.0, pf)  # wants +900 sh = $45k
    assert cost == 0.0 and pos.quantity == 100  # unaffordable -> unchanged


# ── B. breadth admission ─────────────────────────────────────────────────────────

def test_net_sector_cap_breadth_fills_to_target_when_tail_concentrated():
    longs = [f"L{i}" for i in range(60)]
    shorts = [f"S{i}" for i in range(60)]
    smap = {**{s: "TECH" for s in longs}, **{s: "ENERGY" for s in shorts}}
    out = apply_net_sector_cap(shorts, longs, smap, cap=0.30, n_target=60)
    assert len(out) == 60  # breadth preserved (pre-Phase2 capped ~floor(0.3*60)=18)


def test_net_sector_cap_never_shorts_a_long():
    longs = ["A", "B", "C"]
    shorts = ["C", "D", "E"]  # C is also long — must be skipped
    smap = {s: "TECH" for s in longs + shorts}
    out = apply_net_sector_cap(shorts, longs, smap, cap=0.30, n_target=10)
    assert "C" not in out
    assert set(out) == {"D", "E"}


def test_net_sector_cap_respects_n_target_ceiling():
    shorts = [f"S{i}" for i in range(50)]
    smap = {s: "ENERGY" for s in shorts}
    out = apply_net_sector_cap(shorts, [], smap, cap=0.30, n_target=20)
    assert len(out) == 20  # never exceeds n_target
