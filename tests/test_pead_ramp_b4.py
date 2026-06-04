"""Tests for B4 aggressive PEAD paper-ramp sizing + config registration."""

from app.agents.portfolio_manager import apply_pead_size_ramp
from app.database.agent_config import _DEFAULTS, CONFIG_SCHEMA


# ── ramp math ─────────────────────────────────────────────────────────────────

def test_ramp_multiplies():
    # 100 sh * 3x = 300, cap 10% of 100k @ $50 = 200 sh -> capped to 200
    assert apply_pead_size_ramp(100, price=50.0, account_value=100_000,
                                size_mult=3.0, max_position_pct=0.10) == 200


def test_ramp_multiplies_under_cap():
    # 100 sh * 3x = 300, cap 10% of 1M @ $50 = 2000 -> not capped
    assert apply_pead_size_ramp(100, price=50.0, account_value=1_000_000,
                                size_mult=3.0, max_position_pct=0.10) == 300


def test_no_op_when_unramped():
    # size_mult=1.0 + no cap -> unchanged
    assert apply_pead_size_ramp(137, price=50.0, account_value=100_000,
                                size_mult=1.0, max_position_pct=None) == 137


def test_cap_only_without_mult():
    # size_mult=1.0 but cap binds: 1000 sh @ $50, cap 5% of 100k = 100 sh
    assert apply_pead_size_ramp(1000, price=50.0, account_value=100_000,
                                size_mult=1.0, max_position_pct=0.05) == 100


def test_never_below_one():
    assert apply_pead_size_ramp(1, price=10_000.0, account_value=1000,
                                size_mult=3.0, max_position_pct=0.01) == 1


def test_zero_price_skips_cap():
    # defensive: price<=0 must not divide-by-zero; cap skipped, mult applies
    assert apply_pead_size_ramp(10, price=0.0, account_value=100_000,
                                size_mult=3.0, max_position_pct=0.10) == 30


# ── config registration (aggressive defaults, live-tunable) ───────────────────

def test_config_keys_registered_with_aggressive_defaults():
    assert _DEFAULTS["pm.pead_size_mult"] == 3.0
    assert _DEFAULTS["pm.pead_max_position_pct"] == 0.10


def test_config_keys_have_sane_bounds():
    by_key = {s["key"]: s for s in CONFIG_SCHEMA}
    mult = by_key["pm.pead_size_mult"]
    cap = by_key["pm.pead_max_position_pct"]
    assert mult["type"] == "float" and mult["min"] >= 1.0 and mult["max"] <= 10.0
    assert cap["type"] == "float" and 0.0 < cap["min"] and cap["max"] <= 0.25


# ── RM PEAD-aware per-name cap (validate_position_size override) ───────────────

from app.agents.risk_rules import validate_position_size, RiskLimits


def test_position_size_override_allows_pead_10pct():
    # 8% position: rejected under global 5%, allowed under PEAD 10% override
    cost, av = 8_000, 100_000
    ok_global, _ = validate_position_size(cost, av, RiskLimits())
    ok_pead, _ = validate_position_size(cost, av, RiskLimits(), max_pct_override=0.10)
    assert ok_global is False
    assert ok_pead is True


def test_position_size_override_still_caps_above_pead_limit():
    # 12% exceeds even the 10% PEAD ceiling -> rejected
    ok, msg = validate_position_size(12_000, 100_000, RiskLimits(), max_pct_override=0.10)
    assert ok is False


def test_position_size_no_override_unchanged():
    # default global behaviour preserved when no override
    ok5, _ = validate_position_size(5_000, 100_000, RiskLimits())   # 5% exactly = ok
    ok6, _ = validate_position_size(6_000, 100_000, RiskLimits())   # 6% > 5% = reject
    assert ok5 is True and ok6 is False
