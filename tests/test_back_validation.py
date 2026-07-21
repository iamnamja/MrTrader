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
            intended=None, n_blocked=None, overlay=None,
            crash=None, credit=None, ladder=None, ungoverned=None):
    with bv._conn() as c:
        c.execute(
            "INSERT INTO trend_backval_daily(trade_date, nav, prices, positions, "
            "intended_weights, n_positions, n_blocked, overlay_mult, crash_mult, credit_mult, "
            "ladder_mult, ungoverned_weights, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (td, nav, json.dumps(prices or {}), json.dumps(positions or {}),
             (json.dumps(intended) if intended is not None else None),
             len(positions or {}), n_blocked, overlay, crash, credit, ladder,
             (json.dumps(ungoverned) if ungoverned is not None else None), 0.0))


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


# ── CH1: per-name-gate shadow soak (enforce-threshold calibration data) ───────
def test_record_intent_persists_per_name_metrics(bv):
    s = {"status": "ok", "mode": "live", "blocked": [], "intended_weights": {"SPY": 0.1},
         "per_name_metrics": {"mode": "shadow", "allow": True, "would_block": False,
                              "breaches": [], "max_name_weight": 0.079,
                              "weighted_avg_book_corr": 0.347, "portfolio_heat_frac": 0.01}}
    assert bv.record_rebalance_intent(s) is True
    m = json.loads(bv.read_daily()[-1]["per_name_metrics"])
    assert m["weighted_avg_book_corr"] == 0.347 and m["would_block"] is False


def test_soak_report_summarizes_corr_and_threshold_blocks(bv):
    # three clean rebalances at book-corr 0.30 / 0.40 / 0.92 → only the 0.92 breaches 0.90
    for td, corr, wb in [("2026-07-06", 0.30, False), ("2026-07-13", 0.40, False),
                         ("2026-07-20", 0.92, True)]:
        bv.record_rebalance_intent({
            "status": "ok", "mode": "live", "blocked": [], "intended_weights": {"SPY": 0.1},
            "per_name_metrics": {"would_block": wb, "max_name_weight": 0.08,
                                 "weighted_avg_book_corr": corr, "portfolio_heat_frac": 0.01}},
            asof=td)
    rep = bv.per_name_soak_report()
    assert rep["n"] == 3
    assert rep["book_corr"]["min"] == 0.30 and rep["book_corr"]["max"] == 0.92
    # only the 0.92 row exceeds the 0.90 candidate; none exceed 0.95
    assert rep["would_block_at"]["0.90"] == 1 and rep["would_block_at"]["0.95"] == 0
    assert rep["would_block_at"]["0.80"] == 1
    assert rep["actual_would_blocks"] == 1


def test_soak_report_empty_when_no_metrics(bv):
    bv.record_rebalance_intent({"status": "ok", "mode": "live", "blocked": [],
                                "intended_weights": {"SPY": 0.1}})   # no per_name_metrics
    rep = bv.per_name_soak_report()
    assert rep["n"] == 0 and rep["book_corr"] == {}


# ── CH0b: per-governor persistence + counterfactual + regime attribution ──────
def test_record_intent_persists_individual_governors_and_ungoverned(bv):
    # crash 0.5, credit 1.0, ladder 1.0 → governed book is half the ungoverned book.
    s = {"status": "ok", "mode": "live", "blocked": [],
         "intended_weights": {"SPY": 0.05, "QQQ": 0.05},
         "ungoverned_weights": {"SPY": 0.10, "QQQ": 0.10},
         "crash_governor_mult": 0.5, "credit_governor_mult": 1.0,
         "drawdown_ladder_mult": 1.0, "overlay_mult": 0.5}
    assert bv.record_rebalance_intent(s) is True
    r = bv.read_daily()[-1]
    assert r["crash_mult"] == 0.5 and r["credit_mult"] == 1.0 and r["ladder_mult"] == 1.0
    assert json.loads(r["ungoverned_weights"]) == {"SPY": 0.10, "QQQ": 0.10}


def test_daily_rows_ungoverned_falls_back_to_intended_when_absent(bv):
    # Pre-CH0b rows (no ungoverned_weights) → ungoverned == intended → governor_pnl 0.
    snaps = [
        {"trade_date": "2026-06-01", "nav": 100_000.0, "prices": json.dumps({"SPY": 10.0}),
         "positions": json.dumps({"SPY": 1000.0}), "intended_weights": json.dumps({"SPY": 0.10})},
        {"trade_date": "2026-06-02", "nav": 100_000.0, "prices": json.dumps({"SPY": 11.0}),
         "positions": json.dumps({"SPY": 1000.0}), "intended_weights": None},
    ]
    rows, _ = bv.daily_rows(snaps)
    assert rows[0]["intended"] == pytest.approx(rows[0]["ungoverned"])
    assert bv.governor_counterfactual(rows)["governor_pnl"] == pytest.approx(0.0)


def test_governor_counterfactual_derisk_costs_in_a_rally(bv):
    # governed = half-size (crash cut), ungoverned = full-size; +10% up move → de-risking
    # LOST money (governed_cum < ungoverned_cum → governor_pnl < 0).
    snaps = [
        {"trade_date": "2026-06-01", "nav": 100_000.0, "prices": json.dumps({"SPY": 10.0}),
         "positions": json.dumps({"SPY": 1000.0}),
         "intended_weights": json.dumps({"SPY": 0.05}),
         "ungoverned_weights": json.dumps({"SPY": 0.10})},
        {"trade_date": "2026-06-02", "nav": 100_000.0, "prices": json.dumps({"SPY": 11.0}),
         "positions": json.dumps({"SPY": 1000.0})},
    ]
    rows, _ = bv.daily_rows(snaps)
    cf = bv.governor_counterfactual(rows)
    assert cf["governed_cum"] == pytest.approx(0.005)     # 0.05 * 0.10
    assert cf["ungoverned_cum"] == pytest.approx(0.010)   # 0.10 * 0.10
    assert cf["governor_pnl"] == pytest.approx(-0.005)    # de-risk cost 50bps in the rally


def test_governor_counterfactual_derisk_helps_in_a_selloff(bv):
    # governed = half-size (crash cut), ungoverned = full-size; -10% down move → de-risking
    # HELPED (governed_cum > ungoverned_cum → governor_pnl > 0, the positive direction).
    snaps = [
        {"trade_date": "2026-06-01", "nav": 100_000.0, "prices": json.dumps({"SPY": 10.0}),
         "positions": json.dumps({"SPY": 500.0}),
         "intended_weights": json.dumps({"SPY": 0.05}),
         "ungoverned_weights": json.dumps({"SPY": 0.10})},
        {"trade_date": "2026-06-02", "nav": 100_000.0, "prices": json.dumps({"SPY": 9.0}),
         "positions": json.dumps({"SPY": 500.0})},
    ]
    rows, _ = bv.daily_rows(snaps)
    cf = bv.governor_counterfactual(rows)
    assert cf["governed_cum"] == pytest.approx(-0.005)     # 0.05 * -0.10
    assert cf["ungoverned_cum"] == pytest.approx(-0.010)   # 0.10 * -0.10
    assert cf["governor_pnl"] == pytest.approx(0.005)      # de-risk SAVED 50bps in the selloff


def test_daily_rows_fresh_intent_without_ungov_does_not_use_stale_ungov(bv):
    # A post-CH0b rebalance (day3) updates intent but — anomalously — carries NO ungoverned book.
    # It must fall back to the FRESH intent (pnl 0), NOT compare against day1's stale ungoverned.
    snaps = [
        {"trade_date": "2026-06-01", "nav": 100_000.0, "prices": json.dumps({"SPY": 10.0}),
         "positions": json.dumps({"SPY": 500.0}),
         "intended_weights": json.dumps({"SPY": 0.05}),
         "ungoverned_weights": json.dumps({"SPY": 0.10})},          # day1: big stale ungov
        {"trade_date": "2026-06-02", "nav": 100_000.0, "prices": json.dumps({"SPY": 10.0}),
         "positions": json.dumps({"SPY": 500.0})},
        {"trade_date": "2026-06-03", "nav": 100_000.0, "prices": json.dumps({"SPY": 10.0}),
         "positions": json.dumps({"SPY": 500.0}),
         "intended_weights": json.dumps({"SPY": 0.20})},            # day3: fresh intent, NO ungov
        {"trade_date": "2026-06-04", "nav": 100_000.0, "prices": json.dumps({"SPY": 11.0}),
         "positions": json.dumps({"SPY": 500.0})},
    ]
    rows, _ = bv.daily_rows(snaps)
    last = rows[-1]  # the day3->day4 pair uses day3's carried books
    assert last["intended"] == pytest.approx(last["ungoverned"])   # fresh intent, not stale 0.10
    assert last["ungoverned"] == pytest.approx(0.20 * (11.0 / 10.0 - 1.0))


def test_regime_slices_group_and_attribute(bv, monkeypatch):
    from scripts.walkforward import regime as _rg
    monkeypatch.setattr(_rg, "load_regime_map",
                        lambda a, b, **k: {__import__("datetime").date(2026, 6, 2): "BULL"})
    rows = [{"date": "2026-06-02", "actual": 0.01, "intended": 0.005, "ungoverned": 0.010}]
    out = bv.regime_slices(rows)
    assert set(out) == {"BULL"}
    assert out["BULL"]["n_days"] == 1
    assert out["BULL"]["governor_pnl"] == pytest.approx(0.005 - 0.010)


def test_regime_slices_empty_when_map_unavailable(bv, monkeypatch):
    from scripts.walkforward import regime as _rg
    monkeypatch.setattr(_rg, "load_regime_map",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no vix")))
    rows = [{"date": "2026-06-02", "actual": 0.01, "intended": 0.005, "ungoverned": 0.010}]
    assert bv.regime_slices(rows) == {}   # swallowed → omitted, not crashed


def test_compute_report_counts_individual_governors(bv):
    _insert(bv, td="2026-06-01", prices={"SPY": 10.0}, positions={"SPY": 1000.0},
            intended={"SPY": 0.05}, ungoverned={"SPY": 0.10},
            crash=0.5, credit=1.0, ladder=1.0, overlay=0.5)
    _insert(bv, td="2026-06-02", prices={"SPY": 11.0}, positions={"SPY": 1000.0})
    rep = bv.compute_report()
    assert rep.crash_governor_days == 1     # crash_mult 0.5 < 1
    assert rep.credit_governor_days == 0    # credit 1.0
    assert rep.ladder_days == 0
    assert rep.governor_pnl == pytest.approx(-0.005)   # de-risk cost in the +10% move


def test_schema_migration_is_idempotent_on_legacy_db(bv):
    # Simulate a legacy DB: drop the CH0b columns, then _conn() must re-add them (no crash).
    with bv._conn() as c:
        c.execute("DROP TABLE trend_backval_daily")
        c.executescript(
            "CREATE TABLE trend_backval_daily (trade_date TEXT PRIMARY KEY, nav REAL, "
            "prices TEXT, positions TEXT, intended_weights TEXT, n_positions INTEGER, "
            "n_blocked INTEGER, overlay_mult REAL, created_at REAL);")
    # first _conn() migrates; a second must be a no-op (idempotent)
    for _ in range(2):
        with bv._conn() as c:
            cols = {r[1] for r in c.execute("PRAGMA table_info(trend_backval_daily)").fetchall()}
    assert {"crash_mult", "credit_mult", "ladder_mult", "ungoverned_weights"} <= cols
