"""Regression: PEAD live sizing uses PEAD's OWN stop/target, not generate_signal's swing ATR.

The Opus live-path sweep found the Trader sized PEAD positions off `generate_signal`'s 2.5×ATR
swing stop instead of PEAD's validated 0.5×/1.5×ATR stop/target -> live PEAD systematically
under-sized vs its CPCV (and a no-signal generate_signal returns stop=0 -> degenerate sizing).
`_pead_sized_stop_target` scales the proposal's own stop/target (as a fraction of its scan
price) onto the live entry. This locks that behavior.
"""
from __future__ import annotations

from app.agents.trader import Trader


def test_pead_stop_target_scaled_to_live_entry():
    # PEAD proposal: scan entry 100, stop 98 (2% — PEAD's tight ATR), target 106 (6%).
    proposal = {"selector": "pead", "entry_price": 100.0, "stop_loss": 98.0,
                "profit_target": 106.0}
    # Live entry drifted up to 101 (within the entry-quality gate). Stop/target should keep the
    # SAME fractions (98% / 106%) of the live entry -> 98.98 / 107.06, NOT generate_signal's stop.
    stop, tgt = Trader._pead_sized_stop_target(101.0, proposal)
    assert stop == round(101.0 * 0.98, 2) == 98.98
    assert tgt == round(101.0 * 1.06, 2) == 107.06
    # risk-per-share is ~2% of entry (PEAD's tight stop), not the ~5-8% a swing ATR stop implies
    assert abs((101.0 - stop) / 101.0 - 0.02) < 1e-6


def test_pead_stop_target_short_side_preserved():
    # Short PEAD: stop ABOVE entry, target BELOW -> fractions >1 / <1 preserve the side.
    # NOTE: the HELPER is direction-correct, but downstream size_position returns 0 for
    # stop>=entry (long-only contract), so short PEAD won't actually size until size_position
    # is made abs(entry-stop)-aware. Gated off today (pm.pead_enable_shorts=false); fix before
    # enabling shorts. This test locks only the helper's side-preservation.
    proposal = {"selector": "pead", "entry_price": 100.0, "stop_loss": 102.0,
                "profit_target": 94.0}
    stop, tgt = Trader._pead_sized_stop_target(100.0, proposal)
    assert stop == 102.0 and tgt == 94.0
    assert stop > 100.0 > tgt   # short side intact


def test_pead_sized_returns_none_when_unavailable():
    assert Trader._pead_sized_stop_target(0.0, {"entry_price": 100, "stop_loss": 98}) == (None, None)
    assert Trader._pead_sized_stop_target(100.0, {"entry_price": 0, "stop_loss": 98}) == (None, None)
    # missing stop/target -> that leg is None (left to generate_signal's value)
    assert Trader._pead_sized_stop_target(100.0, {"entry_price": 100.0}) == (None, None)
    s, t = Trader._pead_sized_stop_target(100.0, {"entry_price": 100.0, "stop_loss": 97.0})
    assert s == 97.0 and t is None


# ── PEAD live-sizing CAP fidelity (HIGH-2): size_position honors a custom cap ──────────
# The Trader re-sizes PEAD via size_position; previously it used the global 10%
# MAX_POSITION_PCT, silently overriding the owner-set pm.pead_max_position_pct=0.05
# telemetry cap. size_position now takes max_position_pct so the Trader can pass PEAD's 5%.
from app.strategy.position_sizer import size_position  # noqa: E402


def test_size_position_custom_cap_binds_when_tight_stop():
    """With a tight stop (cap is the binding constraint), a 5% cap halves the position
    vs the 10% default — the PEAD telemetry cap is now enforceable at execution."""
    eq = cash = 100_000.0
    # entry 100, stop 99 -> $1/share risk; risk-based=2000, cash-cap=900, so the
    # per-position cap binds. ml_score=0 -> no conviction multiplier.
    default = size_position(eq, cash, 100.0, 99.0, ml_score=0.0)
    pead = size_position(eq, cash, 100.0, 99.0, ml_score=0.0, max_position_pct=0.05)
    assert default == int(eq * 0.10 / 100.0) == 100
    assert pead == int(eq * 0.05 / 100.0) == 50


def test_size_position_custom_cap_inert_when_risk_based_smaller():
    """When the risk-based size is below both caps, max_position_pct doesn't change it."""
    eq = cash = 100_000.0
    # entry 100, stop 50 -> $50/share; risk-based = 2000*0.02... = 40 shares << caps.
    default = size_position(eq, cash, 100.0, 50.0, ml_score=0.0)
    pead = size_position(eq, cash, 100.0, 50.0, ml_score=0.0, max_position_pct=0.05)
    assert default == pead == 40
