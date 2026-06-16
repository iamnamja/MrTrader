"""P1-4 — tests for the trend intended-vs-actual back-validation instrument."""
import json
import importlib

import pytest


@pytest.fixture()
def bv(tmp_path, monkeypatch):
    """Reload the module with an isolated sqlite DB (no shared-file lock)."""
    monkeypatch.setenv("MRTRADER_BACKVAL_DB", str(tmp_path / "bv.db"))
    import app.live_trading.back_validation as _bv
    importlib.reload(_bv)
    return _bv


def _insert(bv, *, td, nav=100_000.0, prices=None, positions=None,
            intended=None, n_blocked=None, overlay=None):
    with bv._conn() as c:
        c.execute(
            "INSERT INTO trend_backval_daily(trade_date, nav, prices, positions, "
            "intended_weights, n_positions, n_blocked, overlay_mult, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (td, nav, json.dumps(prices or {}), json.dumps(positions or {}),
             (json.dumps(intended) if intended is not None else None),
             len(positions or {}), n_blocked, overlay, 0.0))


# ── daily_pairs: intended-vs-actual, same prices, PIT ─────────────────────────
def test_daily_pairs_zero_friction_when_actual_equals_intended(bv):
    # NAV 100k, SPY $10. Intended 12.5% of NAV in SPY = $12500 = 1250 sh; actual holds 1250.
    # Day2 price $11 (+10%). actual_w = 1250*10/100000 = 0.125; intended_w = 0.125.
    # both returns = 0.125 * 0.10 = 0.0125 -> zero gap.
    snaps = [
        {"trade_date": "2026-06-01", "nav": 100_000.0,
         "prices": json.dumps({"SPY": 10.0}), "positions": json.dumps({"SPY": 1250.0}),
         "intended_weights": json.dumps({"SPY": 0.125})},
        {"trade_date": "2026-06-02", "nav": 100_000.0,
         "prices": json.dumps({"SPY": 11.0}), "positions": json.dumps({"SPY": 1250.0}),
         "intended_weights": None},
    ]
    pairs, diag = bv.daily_pairs(snaps)
    assert len(pairs) == 1
    a, i = pairs[0]
    assert a == pytest.approx(0.0125)
    assert i == pytest.approx(0.0125)
    assert a - i == pytest.approx(0.0)


def test_daily_pairs_rounding_friction_shows_as_gap(bv):
    # Intended 0.125 of NAV in SPY but actual only holds 1000 sh ($10k = 0.10) due to rounding.
    # Day2 +10%: actual_ret = 0.10*0.10 = 0.010; intended_ret = 0.125*0.10 = 0.0125.
    # gap = actual - intended = -0.0025 (under-deployed -> drag).
    snaps = [
        {"trade_date": "2026-06-01", "nav": 100_000.0,
         "prices": json.dumps({"SPY": 10.0}), "positions": json.dumps({"SPY": 1000.0}),
         "intended_weights": json.dumps({"SPY": 0.125})},
        {"trade_date": "2026-06-02", "nav": 100_000.0,
         "prices": json.dumps({"SPY": 11.0}), "positions": json.dumps({"SPY": 1000.0}),
         "intended_weights": None},
    ]
    pairs, _ = bv.daily_pairs(snaps)
    a, i = pairs[0]
    assert a == pytest.approx(0.010)
    assert i == pytest.approx(0.0125)
    assert a - i == pytest.approx(-0.0025)


def test_daily_pairs_carries_intent_forward(bv):
    # Intent set on day1; days 2 and 3 have NO new intent -> carried forward.
    snaps = [
        {"trade_date": "2026-06-01", "nav": 100_000.0, "prices": json.dumps({"SPY": 10.0}),
         "positions": json.dumps({"SPY": 1000.0}), "intended_weights": json.dumps({"SPY": 0.10})},
        {"trade_date": "2026-06-02", "nav": 100_000.0, "prices": json.dumps({"SPY": 11.0}),
         "positions": json.dumps({"SPY": 1000.0}), "intended_weights": None},
        {"trade_date": "2026-06-03", "nav": 100_000.0, "prices": json.dumps({"SPY": 12.0}),
         "positions": json.dumps({"SPY": 1000.0}), "intended_weights": None},
    ]
    pairs, diag = bv.daily_pairs(snaps)
    assert len(pairs) == 2
    assert diag["no_intent_days"] == 0  # intent present from day1


def test_daily_pairs_drops_days_before_first_intent(bv):
    # No intent yet on day1 -> the day1->day2 pair is dropped (counted in diag).
    snaps = [
        {"trade_date": "2026-06-01", "nav": 100_000.0, "prices": json.dumps({"SPY": 10.0}),
         "positions": json.dumps({"SPY": 1000.0}), "intended_weights": None},
        {"trade_date": "2026-06-02", "nav": 100_000.0, "prices": json.dumps({"SPY": 11.0}),
         "positions": json.dumps({"SPY": 1000.0}), "intended_weights": json.dumps({"SPY": 0.10})},
    ]
    pairs, diag = bv.daily_pairs(snaps)
    assert pairs == []
    assert diag["no_intent_days"] == 1


def test_daily_pairs_only_prices_on_both_days_contribute(bv):
    # A symbol newly priced on day2 only (entered) contributes nothing that day (no px0).
    snaps = [
        {"trade_date": "2026-06-01", "nav": 100_000.0, "prices": json.dumps({"SPY": 10.0}),
         "positions": json.dumps({"SPY": 1000.0}), "intended_weights": json.dumps({"SPY": 0.10})},
        {"trade_date": "2026-06-02", "nav": 100_000.0,
         "prices": json.dumps({"SPY": 11.0, "QQQ": 50.0}),
         "positions": json.dumps({"SPY": 1000.0, "QQQ": 100.0}), "intended_weights": None},
    ]
    pairs, _ = bv.daily_pairs(snaps)
    a, i = pairs[0]
    assert a == pytest.approx(0.010)  # only SPY (QQQ has no px0)


# ── _tracking_metrics + _verdict ──────────────────────────────────────────────
def test_tracking_metrics_drift_is_actual_minus_intended(bv):
    pairs = [(0.009, 0.010), (0.009, 0.010), (0.009, 0.010)]  # actual 10bps below intended
    m = bv._tracking_metrics(pairs)
    assert m["drift_ann"] == pytest.approx(-0.001 * 252)
    assert m["te_ann"] == pytest.approx(0.0, abs=1e-9)
    assert m["drag_bps_day"] == pytest.approx(-10.0, abs=1e-6)


def test_verdict_building_below_min_days(bv):
    assert bv._verdict({"n_days": 5, "corr": 0.99, "te_ann": 0.0, "drift_ann": 0.0}) == "BUILDING"


def test_verdict_pass_tight_tracking(bv):
    m = {"n_days": 30, "corr": 0.97, "te_ann": 0.01, "drift_ann": 0.005}
    assert bv._verdict(m) == "PASS"


def test_verdict_pass_flat_window_corr_none(bv):
    # Legitimately flat sleeve: corr undefined but tracking error tiny -> PASS (M2 fix).
    m = {"n_days": 30, "corr": None, "te_ann": 0.001, "drift_ann": 0.0}
    assert bv._verdict(m) == "PASS"


def test_verdict_fail_big_gap(bv):
    m = {"n_days": 30, "corr": 0.99, "te_ann": 0.10, "drift_ann": 0.0}
    assert bv._verdict(m) == "FAIL"


def test_verdict_fail_broken_corr(bv):
    m = {"n_days": 30, "corr": 0.10, "te_ann": 0.015, "drift_ann": 0.005}
    assert bv._verdict(m) == "FAIL"


def test_verdict_watch_middle(bv):
    m = {"n_days": 30, "corr": 0.85, "te_ann": 0.03, "drift_ann": 0.01}
    assert bv._verdict(m) == "WATCH"


# ── compute_report end-to-end (isolated DB) ───────────────────────────────────
def test_compute_report_building_below_min(bv):
    _insert(bv, td="2026-06-01", prices={"SPY": 10.0}, positions={"SPY": 1000.0},
            intended={"SPY": 0.10})
    _insert(bv, td="2026-06-02", prices={"SPY": 11.0}, positions={"SPY": 1000.0})
    rep = bv.compute_report()
    assert rep.verdict == "BUILDING"
    assert rep.n_days == 1


def test_compute_report_counts_governor_and_blocks(bv):
    _insert(bv, td="2026-06-01", prices={"SPY": 10.0}, positions={"SPY": 1000.0},
            intended={"SPY": 0.10}, n_blocked=2, overlay=0.5)
    _insert(bv, td="2026-06-02", prices={"SPY": 11.0}, positions={"SPY": 1000.0})
    rep = bv.compute_report()
    assert rep.governor_days == 1     # overlay 0.5 < 1.0
    assert rep.total_blocked == 2


def test_record_rebalance_intent_skips_dormant(bv):
    # dormant summary carries no intended_weights -> nothing recorded.
    assert bv.record_rebalance_intent({"status": "dormant"}) is False


def test_record_rebalance_intent_skips_shadow_with_real_intent(bv):
    # CRITICAL guard: a SHADOW run DOES carry intended_weights and status='ok', but places
    # no orders (empty actual book). Recording it would make the verdict spuriously FAIL.
    s = {"status": "ok", "mode": "shadow", "intended_weights": {"SPY": 0.1}, "blocked": []}
    assert bv.record_rebalance_intent(s) is False
    assert bv.read_daily() == []  # nothing written


def test_record_rebalance_intent_records_live(bv):
    s = {"status": "ok", "mode": "live", "intended_weights": {"SPY": 0.1}, "blocked": [1, 2]}
    assert bv.record_rebalance_intent(s) is True
    rows = bv.read_daily()
    assert rows and json.loads(rows[-1]["intended_weights"]) == {"SPY": 0.1}
    assert rows[-1]["n_blocked"] == 2
