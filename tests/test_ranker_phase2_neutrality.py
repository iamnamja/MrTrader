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


# ── C. _pm_score full-ranking when enable_shorts (the real root cause) ───────────
# The L/S arm needs the FULL cross-section (top→long, bottom→short). The long-only
# proposal_pool_size cap + min_confidence floor surfaced only ~50 long-attractive
# names → no bottom to short → long book absorbed them all → empty short / net-long.

def _bars(n=250):
    import numpy as np
    import pandas as pd
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(np.random.default_rng(7).normal(0, 0.5, n))
    return pd.DataFrame({"open": close * 0.999, "high": close * 1.005,
                         "low": close * 0.995, "close": close,
                         "volume": np.full(n, 1_000_000.0)}, index=idx)


def _scoring_model(probas):
    import numpy as np
    m = MagicMock()
    m.is_trained = True
    m.feature_names = ["close", "volume"]
    m._ts_norm_state = None
    m._highvix_sibling = None
    m._regime_split_threshold = None
    arr = np.array(probas, dtype=float)
    m.predict.side_effect = lambda X, *a, **k: (np.ones(len(X), dtype=int), arr[:len(X)])
    m.predict_with_vix.side_effect = lambda X, *a, **k: (np.ones(len(X), dtype=int), arr[:len(X)])
    return m


def test_pm_score_full_ranking_when_enable_shorts():
    from app.backtesting.agent_simulator import AgentSimulator
    probas = [0.70, 0.60, 0.50, 0.30, 0.10]  # 5 names; only 3 clear a 0.40 floor
    syms = [f"S{i}" for i in range(5)]
    data = {s: _bars() for s in syms}
    day = data[syms[0]].index[-1].date()

    # Long-only: capped by proposal_pool_size (and floored by min_confidence)
    sim_long = AgentSimulator(model=_scoring_model(probas), enable_shorts=False,
                              rebalance_mode=True, min_confidence=0.40,
                              proposal_pool_size=2, max_vol_pct=None)
    out_long = sim_long._pm_score(day, data)
    assert len(out_long) <= 2  # long-only proposal pool

    # L/S rebalance: FULL ranking — all 5 names, including the bottom (for shorting)
    sim_ls = AgentSimulator(model=_scoring_model(probas), enable_shorts=True,
                            rebalance_mode=True, min_confidence=0.40,
                            proposal_pool_size=2, max_vol_pct=None)
    out_ls = sim_ls._pm_score(day, data)
    assert len(out_ls) == 5
    assert syms[4] in {s for s, _ in out_ls}  # lowest-ranked present for the short leg

    # Gate: enable_shorts WITHOUT rebalance_mode (signal mode) must NOT bypass the
    # floor/cap (signal mode has no own min_confidence floor). Stays capped.
    sim_signal = AgentSimulator(model=_scoring_model(probas), enable_shorts=True,
                                rebalance_mode=False, min_confidence=0.40,
                                proposal_pool_size=2, max_vol_pct=None)
    assert len(sim_signal._pm_score(day, data)) <= 2
