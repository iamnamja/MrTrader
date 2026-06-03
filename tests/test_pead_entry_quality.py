"""PEAD-aware entry-quality gate tests.

The validated PEAD CPCV backtest (+0.546) enters at the post-earnings OPEN with NO
price-run/spread rejection (its only entry filter is the scorer's 8% report-day move
cap). The generic SWING gate (1.5% run / 0.5% spread) was rejecting exactly the
high-drift post-earnings names PEAD targets — an adverse-selection trap that produced
0 live fills. These tests pin the selector-aware WIDE PEAD bounds AND that swing /
intraday / default behavior is unchanged.
"""
from __future__ import annotations

from app.strategy.entry_quality import (
    check_entry_quality,
    PEAD_MAX_PRICE_RUN_PCT,
    PEAD_MAX_SPREAD_PCT,
)


def _spread_quote(spread_pct: float, mid: float = 100.0):
    half = mid * spread_pct / 2.0
    return {"bid": mid - half, "ask": mid + half}


# ── PEAD admits normal post-earnings gaps that swing rejects ─────────────────────

def test_pead_admits_2pct_run_that_swing_rejects():
    # PANW today: ran +2.0% -> swing rejects (1.5% cap), PEAD admits.
    assert check_entry_quality("PANW", 100.0, 102.0, "swing", selector="").approved is False
    assert check_entry_quality("PANW", 100.0, 102.0, "swing", selector="pead").approved is True


def test_pead_admits_7pct_run_that_swing_rejects():
    # ULTA today: ran +7.0% -> well past swing's 1.5%, within PEAD's 10%.
    assert check_entry_quality("ULTA", 100.0, 107.0, "swing", selector="pead").approved is True
    assert check_entry_quality("ULTA", 100.0, 107.0, "swing", selector="").approved is False


def test_pead_admits_adverse_move_that_swing_rejects():
    # GTLB today: -1.6% adverse -> swing rejects (1.5%), PEAD admits.
    assert check_entry_quality("GTLB", 100.0, 98.4, "swing", selector="pead").approved is True
    assert check_entry_quality("GTLB", 100.0, 98.4, "swing", selector="").approved is False


def test_pead_admits_09pct_spread_that_swing_rejects():
    # HPE today: 0.9% spread -> swing rejects (0.5% cap), PEAD admits (1.0% cap).
    q = _spread_quote(0.009)
    assert check_entry_quality("HPE", 100.0, 100.0, "swing", quote=q, selector="pead").approved is True
    assert check_entry_quality("HPE", 100.0, 100.0, "swing", quote=q, selector="").approved is False


# ── PEAD still blocks egregious moves / illiquid names (sanity ceiling) ──────────

def test_pead_rejects_run_above_sanity_ceiling():
    over = (PEAD_MAX_PRICE_RUN_PCT + 0.02)  # 12% > 10% cap
    r = check_entry_quality("X", 100.0, 100.0 * (1 + over), "swing", selector="pead")
    assert r.approved is False and "price_run" in r.reason


def test_pead_rejects_spread_above_sanity_cap():
    q = _spread_quote(PEAD_MAX_SPREAD_PCT + 0.005)  # 1.5% > 1.0% cap
    r = check_entry_quality("X", 100.0, 100.0, "swing", quote=q, selector="pead")
    assert r.approved is False and "spread" in r.reason


# ── Non-PEAD behavior unchanged (regression) ─────────────────────────────────────

def test_swing_default_unchanged_rejects_2pct_run():
    # default selector "" -> swing thresholds (backward compatible, no selector arg)
    assert check_entry_quality("X", 100.0, 102.0, "swing").approved is False
    assert check_entry_quality("X", 100.0, 101.0, "swing").approved is True  # 1.0% < 1.5%


def test_intraday_unaffected_by_pead_branch():
    # intraday keeps its own 2.5% run cap when selector is not pead
    assert check_entry_quality("X", 100.0, 102.0, "intraday", selector="").approved is True   # 2% < 2.5%
    assert check_entry_quality("X", 100.0, 103.0, "intraday", selector="").approved is False  # 3% > 2.5%


def test_quality_short_selector_keeps_swing_bounds():
    # only "pead" gets the wide bounds; other selectors fall back to trade_type
    assert check_entry_quality("X", 100.0, 102.0, "swing", selector="quality_short").approved is False


def test_pead_precedence_over_intraday_trade_type():
    # selector=="pead" must win even if trade_type=="intraday" (3% run > intraday's
    # 2.5% cap, but < PEAD's 10%) — locks the branch order.
    assert check_entry_quality("X", 100.0, 103.0, "intraday", selector="pead").approved is True
    assert check_entry_quality("X", 100.0, 103.0, "intraday", selector="").approved is False
