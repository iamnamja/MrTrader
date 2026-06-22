"""Alpha-v10 H9 — drawdown de-gross ladder wired into the live trend budget (flag-gated, shadow-default).

Pins: the ladder multiplier maps drawdown-from-HWM to the risk-policy-v1 rungs; it is SHADOW by
default (computes + logs but applies 1.0); fail-safe to 1.0 when the HWM is unreadable / NAV>=HWM /
NAV<=0; and when enabled it is applied to the trend `alloc` UN-floored (the -20% rung can flatten).
"""
from __future__ import annotations

import app.live_trading.trend_sleeve as ts
from app.live_trading.risk_policy import RISK_POLICY_V1


def _patch(monkeypatch, *, enabled, peak):
    monkeypatch.setattr("app.database.agent_config.get_agent_config",
                        lambda db, key: ("true" if enabled else "false")
                        if key == "pm.drawdown_ladder_enabled" else None)
    monkeypatch.setattr("app.database.config_store.get_config",
                        lambda db, key: peak if key == "risk.peak_equity" else None)


# ── the multiplier ───────────────────────────────────────────────────────────
def test_ladder_shadow_off_returns_one_even_in_drawdown(monkeypatch):
    _patch(monkeypatch, enabled=False, peak=100_000.0)
    # NAV 85k vs HWM 100k = -15% -> would be 0.50, but flag off -> shadow -> 1.0
    assert ts._drawdown_ladder_multiplier(object(), 85_000.0) == 1.0


def test_ladder_enabled_applies_each_rung(monkeypatch):
    _patch(monkeypatch, enabled=True, peak=100_000.0)
    # rung breakpoints: -8/-12/-16/-20 -> 0.75/0.50/0.25/0.00
    assert ts._drawdown_ladder_multiplier(object(), 95_000.0) == 1.0    # -5% -> no rung
    assert ts._drawdown_ladder_multiplier(object(), 91_000.0) == 0.75   # -9%
    assert ts._drawdown_ladder_multiplier(object(), 87_000.0) == 0.50   # -13%
    assert ts._drawdown_ladder_multiplier(object(), 83_000.0) == 0.25   # -17%
    assert ts._drawdown_ladder_multiplier(object(), 79_000.0) == 0.00   # -21% -> kill rung (flat)


def test_ladder_new_high_no_degross(monkeypatch):
    _patch(monkeypatch, enabled=True, peak=100_000.0)
    assert ts._drawdown_ladder_multiplier(object(), 105_000.0) == 1.0   # NAV > HWM


def test_ladder_failsafe_when_peak_missing_or_nav_bad(monkeypatch):
    _patch(monkeypatch, enabled=True, peak=None)
    assert ts._drawdown_ladder_multiplier(object(), 80_000.0) == 1.0    # no HWM -> no de-gross
    _patch(monkeypatch, enabled=True, peak=100_000.0)
    assert ts._drawdown_ladder_multiplier(object(), 0.0) == 1.0         # NAV<=0 -> no de-gross


def test_ladder_matches_risk_policy_rungs(monkeypatch):
    # the helper must agree with RISK_POLICY_V1.ladder_multiplier exactly when enabled
    _patch(monkeypatch, enabled=True, peak=100_000.0)
    for nav in (91_000.0, 87_000.0, 83_000.0, 79_000.0):
        dd = nav / 100_000.0 - 1.0
        assert ts._drawdown_ladder_multiplier(object(), nav) == RISK_POLICY_V1.ladder_multiplier(dd)


# ── sleeve-level: the ladder de-grosses the live alloc when enabled (un-floored) ──
def test_ladder_degrosses_alloc_in_rebalance(monkeypatch):
    # reuse the trend-sleeve I/O harness pattern: drive run_trend_rebalance with a fake Alpaca whose
    # NAV is 79k vs a 100k HWM -> enabled ladder -> -21% -> 0.0 -> alloc flattened -> no buys.
    from tests.test_trend_sleeve import _FakeAlpaca, _uptrend_prices

    cfg = {
        "pm.trend_enabled": "true", "pm.trend_shadow": "false",
        "pm.trend_allocation_pct": 0.50, "pm.trend_max_position_pct": 0.25,
        "pm.trend_universe": "SPY,QQQ,TLT", "pm.trend_rebalance_weekday": 0,
        "pm.drawdown_ladder_enabled": "true",
        "pm.whole_book_gate_mode": "off", "pm.reconciliation_mode": "off",
    }
    monkeypatch.setattr("app.database.agent_config.get_agent_config", lambda db, key: cfg.get(key))
    monkeypatch.setattr("app.database.config_store.get_config",
                        lambda db, key: 100_000.0 if key == "risk.peak_equity" else None)
    from app.live_trading.kill_switch import kill_switch
    monkeypatch.setattr(kill_switch, "_active", False, raising=False)
    monkeypatch.setattr("app.database.decision_audit.write_decision", lambda **kw: None)
    monkeypatch.setattr(ts, "_current_trend_positions", lambda db, alpaca: {})
    monkeypatch.setattr(ts, "_sync_trend_trade", lambda db, sym, tgt, price, oid: None)
    monkeypatch.setattr("app.live_trading.trend_tracker.record_daily", lambda **kw: True)

    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]), nav=79_000.0)  # -21% from 100k HWM
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    summary = ts.run_trend_rebalance(db=object())
    assert summary["drawdown_ladder_mult"] == 0.0       # kill rung
    assert fake.orders == []                            # flattened budget -> nothing bought


def test_kill_rung_sells_held_positions(monkeypatch):
    # The point of the -20% kill rung is to FLATTEN an existing book, not merely stop buying.
    # Held SPY/QQQ + NAV at the kill rung (0.0 budget) -> the live path must emit SELLs that
    # fully exit every held trend leg.
    from tests.test_trend_sleeve import _FakeAlpaca, _uptrend_prices

    cfg = {
        "pm.trend_enabled": "true", "pm.trend_shadow": "false",
        "pm.trend_allocation_pct": 0.50, "pm.trend_max_position_pct": 0.25,
        "pm.trend_universe": "SPY,QQQ,TLT", "pm.trend_rebalance_weekday": 0,
        "pm.drawdown_ladder_enabled": "true",
        "pm.whole_book_gate_mode": "off", "pm.reconciliation_mode": "off",
    }
    monkeypatch.setattr("app.database.agent_config.get_agent_config", lambda db, key: cfg.get(key))
    monkeypatch.setattr("app.database.config_store.get_config",
                        lambda db, key: 100_000.0 if key == "risk.peak_equity" else None)
    from app.live_trading.kill_switch import kill_switch
    monkeypatch.setattr(kill_switch, "_active", False, raising=False)
    monkeypatch.setattr("app.database.decision_audit.write_decision", lambda **kw: None)
    monkeypatch.setattr(ts, "_current_trend_positions",
                        lambda db, alpaca: {"SPY": 100, "QQQ": 50})  # held book
    monkeypatch.setattr(ts, "_sync_trend_trade", lambda db, sym, tgt, price, oid: None)
    monkeypatch.setattr("app.live_trading.trend_tracker.record_daily", lambda **kw: True)

    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]), nav=79_000.0)  # -21% -> 0.0
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    class _FakeDB:
        def commit(self): pass
        def rollback(self): pass

    summary = ts.run_trend_rebalance(db=_FakeDB())
    assert summary["drawdown_ladder_mult"] == 0.0
    sells = {o["symbol"]: o for o in fake.orders}
    assert set(sells) == {"SPY", "QQQ"}                 # every held leg exited
    assert all(o["side"] == "sell" for o in fake.orders)
    assert sells["SPY"]["qty"] == 100 and sells["QQQ"]["qty"] == 50  # full exits
