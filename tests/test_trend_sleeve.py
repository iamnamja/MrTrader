"""Tests for the live TSMOM trend sleeve (Alpha-v4 live wiring).

Covers the two pure functions (compute_trend_deltas, apply_risk_gate) in isolation
and the run_trend_rebalance I/O orchestrator via a fake Alpaca client + monkeypatch,
plus config registration (trend keys + the PEAD telemetry dial).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.live_trading import trend_sleeve as ts
from app.live_trading.trend_sleeve import compute_trend_deltas, apply_risk_gate


# ── compute_trend_deltas (pure) ───────────────────────────────────────────────

def test_deltas_zero_positions_all_buys():
    # nav 100k, alloc 0.40, SPY weight 0.5 -> $20k / $100 = 200 sh
    out = compute_trend_deltas(
        {"SPY": 0.5}, {}, {"SPY": 100.0}, nav=100_000,
        trend_allocation_pct=0.40, max_position_pct=0.25,
    )
    assert len(out) == 1
    assert out[0] == {"symbol": "SPY", "side": "buy", "qty": 200,
                      "target_shares": 200, "current_shares": 0, "reason": "rebalance"}


def test_deltas_per_name_cap_binds():
    # weight*alloc*nav = 0.9*0.4*100k = $36k but cap 0.25*100k = $25k -> 250 sh @ $100
    out = compute_trend_deltas(
        {"SPY": 0.9}, {}, {"SPY": 100.0}, nav=100_000,
        trend_allocation_pct=0.40, max_position_pct=0.25,
    )
    assert out[0]["qty"] == 250
    assert out[0]["target_shares"] == 250


def test_deltas_held_not_in_target_full_exit():
    # SPY dropped from target -> sell entire position to 0
    out = compute_trend_deltas(
        {}, {"SPY": 150}, {"SPY": 100.0}, nav=100_000,
        trend_allocation_pct=0.40, max_position_pct=0.25,
    )
    assert out == [{"symbol": "SPY", "side": "sell", "qty": 150,
                    "target_shares": 0, "current_shares": 150,
                    "reason": "exit_not_in_target"}]


def test_deltas_target_equals_current_no_order():
    out = compute_trend_deltas(
        {"SPY": 0.5}, {"SPY": 200}, {"SPY": 100.0}, nav=100_000,
        trend_allocation_pct=0.40, max_position_pct=0.25,
    )
    assert out == []


def test_deltas_sells_before_buys():
    # QQQ to be sold to 0, SPY to be bought from 0 -> sell must come first
    out = compute_trend_deltas(
        {"SPY": 0.5}, {"QQQ": 100}, {"SPY": 100.0, "QQQ": 50.0}, nav=100_000,
        trend_allocation_pct=0.40, max_position_pct=0.25,
    )
    assert [o["side"] for o in out] == ["sell", "buy"]
    assert out[0]["symbol"] == "QQQ" and out[1]["symbol"] == "SPY"


def test_deltas_whole_share_floor():
    # $20k / $99.5 = 201.005 -> floor 201
    out = compute_trend_deltas(
        {"SPY": 0.5}, {}, {"SPY": 99.5}, nav=100_000,
        trend_allocation_pct=0.40, max_position_pct=0.25,
    )
    assert out[0]["qty"] == 201


def test_deltas_dust_buy_skipped_but_full_exit_kept():
    # tiny new buy below min_notional is skipped; a held position dropping to 0 is not
    out = compute_trend_deltas(
        {"SPY": 0.5, "QQQ": 0.00001}, {"TLT": 1}, {"SPY": 100.0, "QQQ": 100.0, "TLT": 5.0},
        nav=100_000, trend_allocation_pct=0.40, max_position_pct=0.25, min_notional=50.0,
    )
    syms = {o["symbol"]: o for o in out}
    assert "QQQ" not in syms          # dust buy skipped
    assert syms["TLT"]["side"] == "sell" and syms["TLT"]["target_shares"] == 0  # full exit kept


def test_deltas_alloc_scales_gross():
    # two equal 0.5 weights, alloc 0.40 -> total target gross <= 0.40 * nav
    out = compute_trend_deltas(
        {"SPY": 0.5, "QQQ": 0.5}, {}, {"SPY": 100.0, "QQQ": 100.0}, nav=100_000,
        trend_allocation_pct=0.40, max_position_pct=0.25,
    )
    gross = sum(o["qty"] * 100.0 for o in out)
    assert gross <= 0.40 * 100_000 + 1e-6


# ── apply_risk_gate (pure) ────────────────────────────────────────────────────

def test_gate_sells_always_approved():
    intents = [{"symbol": "SPY", "side": "sell", "qty": 100,
                "target_shares": 0, "current_shares": 100}]
    approved, blocked = apply_risk_gate(
        intents, total_gross=80_000, nav=100_000, max_position_pct=0.25,
        prices={"SPY": 100.0}, gross_cap=0.80,
    )
    assert len(approved) == 1 and not blocked


def test_gate_blocks_gross_cap_breach():
    # already at 78% gross; a $5k buy -> 83% > 80% cap -> blocked
    intents = [{"symbol": "QQQ", "side": "buy", "qty": 50,
                "target_shares": 50, "current_shares": 0}]
    approved, blocked = apply_risk_gate(
        intents, total_gross=78_000, nav=100_000, max_position_pct=0.25,
        prices={"QQQ": 100.0}, gross_cap=0.80,
    )
    assert not approved
    assert blocked and blocked[0]["block_reason"] == "gross_cap"


def test_gate_sell_frees_room_for_buy():
    # sell SPY (-$10k) then buy QQQ (+$5k): starts at 78%, sell -> 68%, buy -> 73% OK
    intents = [
        {"symbol": "SPY", "side": "sell", "qty": 100, "target_shares": 0, "current_shares": 100},
        {"symbol": "QQQ", "side": "buy", "qty": 50, "target_shares": 50, "current_shares": 0},
    ]
    approved, blocked = apply_risk_gate(
        intents, total_gross=78_000, nav=100_000, max_position_pct=0.25,
        prices={"SPY": 100.0, "QQQ": 100.0}, gross_cap=0.80,
    )
    assert len(approved) == 2 and not blocked


def test_gate_fat_finger_rejected():
    # buy notional > nav -> fat finger
    intents = [{"symbol": "SPY", "side": "buy", "qty": 2000,
                "target_shares": 2000, "current_shares": 0}]
    approved, blocked = apply_risk_gate(
        intents, total_gross=0, nav=100_000, max_position_pct=0.25,
        prices={"SPY": 100.0}, gross_cap=0.80,
    )
    assert not approved and blocked[0]["block_reason"] == "fat_finger"


# ── run_trend_rebalance (I/O, faked) ──────────────────────────────────────────

def _uptrend_prices(symbols, n=300):
    """Synthetic steadily-rising daily closes so TSMOM signals are long."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    data = {}
    for i, s in enumerate(symbols):
        base = 100.0 + i * 10
        data[s] = pd.Series([base * (1.0 + 0.001 * k) for k in range(n)], index=idx)
    return pd.DataFrame(data)


class _FakeAlpaca:
    def __init__(self, prices_df, nav=100_000, positions=None, latest=None,
                 raise_account=False):
        self._prices = prices_df
        self._nav = nav
        self._positions = positions or []
        self._latest = latest or {c: float(prices_df[c].iloc[-1]) for c in prices_df.columns}
        self._raise_account = raise_account
        self.orders = []

    def get_bars_batch(self, universe, timeframe, limit):
        return {s: pd.DataFrame({"close": self._prices[s]})
                for s in universe if s in self._prices.columns}

    def get_account(self):
        if self._raise_account:
            raise RuntimeError("account unavailable")
        return {"equity": self._nav}

    def get_positions(self):
        return list(self._positions)

    def get_latest_price(self, sym):
        return self._latest.get(sym)

    def get_quote(self, sym):
        return None

    def place_market_order(self, sym, qty, side, client_order_id=None, est_price=None):
        self.orders.append({"symbol": sym, "qty": qty, "side": side,
                            "client_order_id": client_order_id})
        return {"order_id": f"oid-{sym}"}


@pytest.fixture
def _patch_env(monkeypatch):
    """Patch config, kill switch, audit, tracker, and DB-touching helpers."""
    cfg = {
        "pm.trend_enabled": "true",
        "pm.trend_shadow": "true",
        "pm.trend_allocation_pct": 0.40,
        "pm.trend_max_position_pct": 0.25,
        "pm.trend_universe": "SPY,QQQ,TLT",
        "pm.trend_rebalance_weekday": 0,
    }
    monkeypatch.setattr("app.database.agent_config.get_agent_config",
                        lambda db, key: cfg.get(key))
    # kill switch off
    from app.live_trading.kill_switch import kill_switch
    monkeypatch.setattr(kill_switch, "_active", False, raising=False)
    # capture audit calls
    audits = []
    monkeypatch.setattr("app.database.decision_audit.write_decision",
                        lambda **kw: audits.append(kw))
    # no-op DB-touching helpers
    monkeypatch.setattr(ts, "_current_trend_positions", lambda db, alpaca: {})
    monkeypatch.setattr(ts, "_sync_trend_trade",
                        lambda db, sym, tgt, price, oid: None)
    monkeypatch.setattr("app.live_trading.trend_tracker.record_daily",
                        lambda **kw: True)
    return cfg, audits


def test_run_shadow_places_no_orders(monkeypatch, _patch_env):
    cfg, audits = _patch_env
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    summary = ts.run_trend_rebalance(db=object())

    assert summary["status"] == "ok" and summary["mode"] == "shadow"
    assert fake.orders == []  # shadow sends nothing
    assert summary["approved"]  # but it computed would-be orders
    shadow_audits = [a for a in audits
                     if a.get("strategy") == "trend" and a.get("block_reason") == "shadow"]
    assert len(shadow_audits) == len(summary["approved"])


def _patch_macro_backwardation(monkeypatch, ratio):
    """Force the crash governor's live VIX/VIX3M read to a fixed ratio (fresh data)."""
    import pandas as pd
    from datetime import date
    idx = pd.bdate_range(end=pd.Timestamp(date.today()), periods=10)
    mdf = pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in idx],
                        "vix": [20.0 * ratio] * 10, "vix3m": [20.0] * 10})
    monkeypatch.setattr("app.data.macro_history.update_macro_history", lambda: None)
    monkeypatch.setattr("app.data.macro_history.load_macro_history", lambda: mdf)


def test_run_crash_governor_halves_exposure(monkeypatch, _patch_env):
    """End-to-end: governor ON + VIX backwardation -> sleeve exposure scaled by derisk_to."""
    cfg, audits = _patch_env
    cfg.update({"pm.crash_governor_enabled": "true", "pm.crash_governor_derisk_to": 0.5,
                "pm.crash_governor_ratio_threshold": 1.0, "pm.crash_governor_confirm_days": 1})
    _patch_macro_backwardation(monkeypatch, ratio=1.2)   # VIX>VIX3M -> de-risk
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    summary = ts.run_trend_rebalance(db=object())
    assert summary["status"] == "ok"
    assert summary["crash_governor_mult"] == 0.5         # exposure dial halved (alloc *= 0.5)
    # the de-risk multiplier is recorded on the per-order decision_audit rows (incident trail)
    enter_audits = [a for a in audits if a.get("final_decision") == "enter"]
    assert enter_audits and all(a.get("size_multiplier") == 0.5 for a in enter_audits)


def test_run_overlays_compose_and_clamp(monkeypatch, _patch_env):
    """Both overlays ON + both stressed -> exposure = max(floor, gov*credit) = max(0.25, 0.25)."""
    import pandas as pd
    from datetime import date
    cfg, audits = _patch_env
    cfg.update({"pm.crash_governor_enabled": "true", "pm.crash_governor_derisk_to": 0.5,
                "pm.crash_governor_ratio_threshold": 1.0, "pm.crash_governor_confirm_days": 1,
                "pm.credit_governor_enabled": "true", "pm.credit_governor_derisk_to": 0.5,
                "pm.credit_governor_lookback": 60, "pm.credit_governor_band": 0.0})
    idx = pd.bdate_range(end=pd.Timestamp(date.today()), periods=200)
    mdf = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in idx],
        "vix": [24.0] * 200, "vix3m": [20.0] * 200,                       # backwardation
        "ief": [100.0] * 200,
        "hyg": [170.0 - 0.3 * i for i in range(200)],    # HY steadily falling -> below its MA
    })
    monkeypatch.setattr("app.data.macro_history.update_macro_history", lambda: None)
    monkeypatch.setattr("app.data.macro_history.load_macro_history", lambda: mdf)
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    summary = ts.run_trend_rebalance(db=object())
    assert summary["crash_governor_mult"] == 0.5
    assert summary["credit_governor_mult"] == 0.5
    assert summary["overlay_mult"] == 0.25          # 0.5*0.5 clamped at the 0.25 floor


def test_credit_governor_off_by_default_overlay_equals_gov(monkeypatch, _patch_env):
    """With credit flag absent (default OFF), overlay_mult == the governor mult alone."""
    cfg, audits = _patch_env
    cfg.update({"pm.crash_governor_enabled": "true", "pm.crash_governor_derisk_to": 0.5,
                "pm.crash_governor_ratio_threshold": 1.0, "pm.crash_governor_confirm_days": 1})
    _patch_macro_backwardation(monkeypatch, ratio=1.2)   # only vix/vix3m; no hyg/ief
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)
    summary = ts.run_trend_rebalance(db=object())
    assert summary["credit_governor_mult"] == 1.0       # off -> inert
    assert summary["overlay_mult"] == 0.5               # == governor alone


def test_run_crash_governor_inert_in_contango(monkeypatch, _patch_env):
    cfg, audits = _patch_env
    cfg.update({"pm.crash_governor_enabled": "true", "pm.crash_governor_derisk_to": 0.5,
                "pm.crash_governor_ratio_threshold": 1.0, "pm.crash_governor_confirm_days": 1})
    _patch_macro_backwardation(monkeypatch, ratio=0.92)  # contango -> full exposure
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    summary = ts.run_trend_rebalance(db=object())
    assert summary["crash_governor_mult"] == 1.0


def test_run_live_places_orders(monkeypatch, _patch_env):
    cfg, audits = _patch_env
    cfg["pm.trend_shadow"] = "false"
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    summary = ts.run_trend_rebalance(db=MagicMock())

    assert summary["status"] == "ok" and summary["mode"] == "live"
    assert len(fake.orders) == len(summary["approved"]) > 0
    assert all(o["client_order_id"].startswith("trend-") for o in fake.orders)


def test_run_live_commits_per_order(monkeypatch, _patch_env):
    """F1: each placed order is committed immediately (one db.commit per order), not a
    single deferred commit after the loop — so a restart mid-loop cannot leave a placed
    ETF with an uncommitted Trade row (which reconciliation would adopt as a swing)."""
    cfg, audits = _patch_env
    cfg["pm.trend_shadow"] = "false"
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)
    db = MagicMock()

    summary = ts.run_trend_rebalance(db=db)

    n = len(fake.orders)
    assert summary["mode"] == "live" and n > 0
    assert db.commit.call_count == n, (
        f"expected one commit per order ({n}), got {db.commit.call_count} "
        "(deferred post-loop commit would be 1)")


# ── H1 reconciliation-before-trade wiring (shadow proceeds / enforce HOLDS) ──────
def _patch_recon_break(monkeypatch):
    from app.live_trading import reconciliation as rec
    monkeypatch.setattr(
        rec, "shadow_reconcile_before_trade",
        lambda *a, **k: rec.ReconciliationResult(
            status=rec.FAIL_CLOSED,
            position_breaks=[rec.PositionBreak("SPY", "ALPACA", 100.0, 0.0, -100.0)]))
    return rec


def test_run_enforce_holds_on_reconciliation_break(monkeypatch, _patch_env):
    cfg, audits = _patch_env
    cfg["pm.reconciliation_mode"] = "enforce"
    cfg["pm.trend_shadow"] = "false"          # live -> proves the HOLD precedes any order
    _patch_recon_break(monkeypatch)
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    summary = ts.run_trend_rebalance(db=MagicMock())

    assert summary["status"] == "blocked" and summary["block_reason"] == "reconciliation"
    assert fake.orders == []                  # nothing placed — held before execution


def test_run_shadow_proceeds_despite_reconciliation_break(monkeypatch, _patch_env):
    cfg, audits = _patch_env
    cfg["pm.reconciliation_mode"] = "shadow"
    rec = _patch_recon_break(monkeypatch)
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    summary = ts.run_trend_rebalance(db=MagicMock())

    assert summary["status"] == "ok"          # shadow: a break does NOT block
    assert summary.get("recon_status") == rec.FAIL_CLOSED   # but it WAS observed


def test_run_enforce_with_uppercase_mode_still_enforces(monkeypatch, _patch_env):
    # mode string is normalized -> "Enforce" must NOT silently behave as shadow (safety footgun)
    cfg, audits = _patch_env
    cfg["pm.reconciliation_mode"] = "ENFORCE"
    cfg["pm.trend_shadow"] = "false"
    _patch_recon_break(monkeypatch)
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    summary = ts.run_trend_rebalance(db=MagicMock())

    assert summary["status"] == "blocked" and summary["block_reason"] == "reconciliation"


# ── CH1 per-name gate wiring (shadow proceeds / enforce HOLDS / off skips) ───────
def _diversified_uptrend_prices(symbols, n=300, seed=0):
    """Independent positive-drift random walks: each name net-UP (TSMOM long) but pairwise
    return-correlation ~0 -> a genuinely diversified book (weighted-avg book corr well below the
    gate threshold). Contrast with `_uptrend_prices` (identical path -> corr 1.0)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    data = {s: pd.Series(100.0 + np.cumsum(rng.normal(0.10, 0.30, n)), index=idx)
            for s in symbols}
    return pd.DataFrame(data)


def test_run_enforce_holds_on_per_name_correlation_breach(monkeypatch, _patch_env):
    # default _uptrend_prices = identical path -> book corr 1.0 -> the "one bet" breach HOLDS.
    cfg, audits = _patch_env
    cfg["pm.per_name_gate_mode"] = "enforce"
    cfg["pm.trend_shadow"] = "false"          # live -> proves the HOLD precedes any order
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    summary = ts.run_trend_rebalance(db=MagicMock())

    assert summary["status"] == "blocked" and summary["block_reason"] == "per_name_gate"
    assert any("book_correlation" in b for b in summary.get("per_name_breaches", []))
    assert fake.orders == []                   # nothing placed — held before execution


def test_run_shadow_proceeds_despite_per_name_breach(monkeypatch, _patch_env):
    cfg, audits = _patch_env
    cfg["pm.per_name_gate_mode"] = "shadow"
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    summary = ts.run_trend_rebalance(db=object())

    assert summary["status"] == "ok"           # shadow: a breach does NOT block
    assert summary.get("per_name_allow") is False              # but it WAS observed
    assert summary.get("per_name_breaches")


def test_run_enforce_uppercase_mode_still_enforces(monkeypatch, _patch_env):
    # normalized -> "ENFORCE" must not silently degrade to shadow
    cfg, audits = _patch_env
    cfg["pm.per_name_gate_mode"] = "ENFORCE"
    cfg["pm.trend_shadow"] = "false"
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    summary = ts.run_trend_rebalance(db=MagicMock())

    assert summary["status"] == "blocked" and summary["block_reason"] == "per_name_gate"


def test_run_off_skips_per_name_gate(monkeypatch, _patch_env):
    cfg, audits = _patch_env
    cfg["pm.per_name_gate_mode"] = "off"
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    summary = ts.run_trend_rebalance(db=object())

    assert summary["status"] == "ok"
    assert "per_name_gate_mode" not in summary    # off -> not evaluated (no telemetry)


def test_run_enforce_clean_book_proceeds(monkeypatch, _patch_env):
    # a genuinely diversified book (pairwise corr ~0) must NOT spuriously HOLD in enforce.
    cfg, audits = _patch_env
    cfg["pm.per_name_gate_mode"] = "enforce"
    cfg["pm.trend_shadow"] = "false"
    fake = _FakeAlpaca(_diversified_uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    summary = ts.run_trend_rebalance(db=MagicMock())

    assert summary["status"] == "ok"                          # not blocked
    assert summary.get("per_name_allow") is True
    assert not any("book_correlation" in b for b in summary.get("per_name_breaches", []))


def test_run_live_earlier_orders_committed_when_later_order_fails(monkeypatch, _patch_env):
    """F1 durability: a mid-loop order failure does not lose earlier orders' commits."""
    cfg, audits = _patch_env
    cfg["pm.trend_shadow"] = "false"
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]))
    _orig = fake.place_market_order
    _n = {"i": 0}

    def _flaky(sym, qty, side, client_order_id=None, est_price=None):
        _n["i"] += 1
        if _n["i"] == 2:
            raise RuntimeError("alpaca transient")
        return _orig(sym, qty, side, client_order_id=client_order_id, est_price=est_price)

    fake.place_market_order = _flaky
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)
    db = MagicMock()

    summary = ts.run_trend_rebalance(db=db)

    # The run survives one bad order, and at least the first order was committed
    # before the failure (durability — not all-or-nothing at the end).
    assert summary["status"] == "ok"
    assert db.commit.call_count >= 1


def test_run_live_commit_failure_rolls_back_and_survives(monkeypatch, _patch_env):
    """F1: a db.commit() failure is rolled back (session left usable) and does NOT abort
    the rebalance run — exercises the new rollback branch."""
    cfg, audits = _patch_env
    cfg["pm.trend_shadow"] = "false"
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)
    db = MagicMock()
    db.commit.side_effect = RuntimeError("db write conflict")

    summary = ts.run_trend_rebalance(db=db)

    assert summary["status"] == "ok"      # run survives commit failures
    assert db.rollback.called             # the new rollback path is exercised
    assert len(fake.orders) > 0           # orders still attempted despite commit issues


def test_run_dormant_when_disabled(monkeypatch, _patch_env):
    cfg, audits = _patch_env
    cfg["pm.trend_enabled"] = "false"
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    summary = ts.run_trend_rebalance(db=object())
    assert summary["status"] == "dormant"
    assert fake.orders == []


def test_run_force_bypasses_dormant(monkeypatch, _patch_env):
    cfg, audits = _patch_env
    cfg["pm.trend_enabled"] = "false"
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    summary = ts.run_trend_rebalance(db=object(), force=True)
    assert summary["status"] == "ok"  # ran despite enabled=false


def test_run_kill_switch_blocks(monkeypatch, _patch_env):
    cfg, audits = _patch_env
    from app.live_trading.kill_switch import kill_switch
    monkeypatch.setattr(kill_switch, "_active", True, raising=False)
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    summary = ts.run_trend_rebalance(db=object())
    assert summary["status"] == "blocked"
    assert summary["block_reason"] == "kill_switch"
    assert fake.orders == []


def test_run_failclosed_on_missing_core_symbol(monkeypatch, _patch_env):
    cfg, audits = _patch_env
    # universe present but SPY (core) has no data -> fail closed
    fake = _FakeAlpaca(_uptrend_prices(["QQQ", "TLT"]))  # no SPY
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    summary = ts.run_trend_rebalance(db=object())
    assert summary["status"] == "failed"
    assert summary["block_reason"] == "data_fetch_failed"
    assert fake.orders == []


def test_run_failclosed_on_nav_failure(monkeypatch, _patch_env):
    cfg, audits = _patch_env
    cfg["pm.trend_shadow"] = "false"
    fake = _FakeAlpaca(_uptrend_prices(["SPY", "QQQ", "TLT"]), raise_account=True)
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    summary = ts.run_trend_rebalance(db=object())
    assert summary["status"] == "failed"
    assert summary["block_reason"] == "nav_unavailable"
    assert fake.orders == []


# ── config registration ───────────────────────────────────────────────────────

def test_trend_config_keys_registered():
    from app.database.agent_config import _DEFAULTS, CONFIG_SCHEMA
    by_key = {s["key"]: s for s in CONFIG_SCHEMA}
    for k in ("pm.trend_enabled", "pm.trend_shadow", "pm.trend_allocation_pct",
              "pm.trend_max_position_pct", "pm.trend_universe", "pm.trend_rebalance_weekday"):
        assert k in by_key, f"missing config key {k}"
    # shadow-safe defaults; allocation reconciled 40%->25% after the H1 DEMOTE
    # made trend the sole live sleeve (Track-B 25% framing); then raised 0.25->0.50
    # by the Alpha-v9 P1-2 Kelly/vol-target analysis (trend is the only live edge).
    assert _DEFAULTS["pm.trend_enabled"] == "false"
    assert _DEFAULTS["pm.trend_shadow"] == "true"
    assert _DEFAULTS["pm.trend_allocation_pct"] == 0.50
    assert _DEFAULTS["pm.trend_max_position_pct"] == 0.25


def test_pead_telemetry_dial_applied():
    from app.database.agent_config import _DEFAULTS
    assert _DEFAULTS["pm.pead_size_mult"] == 1.0
    assert _DEFAULTS["pm.pead_max_position_pct"] == 0.05
