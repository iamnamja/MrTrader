"""Macro Intel Phase 2a — polarity-aware macro event outcomes (fix the Beat/Miss bug).

A macro surprise's market meaning isn't uniform "higher=good": inflation/jobless-claims LOWER is
risk-ON; growth/payrolls HIGHER is risk-ON. classify_outcome() is the single source of truth the
API + UI share, replacing the naive sign(actual-estimate) Beat/Miss.
"""
from __future__ import annotations

from app.news.macro_polarity import classify_outcome, polarity_for


# ── the polarity bug cases the operator flagged ──────────────────────────────────
def test_cooler_inflation_is_risk_on():
    # PCE 0.4 vs est 0.5 — LOWER inflation = good = risk-on (NOT a red "Miss")
    o = classify_outcome("PCE", 0.4, 0.5)
    assert o["market_outcome"] == "risk_on" and o["outcome_label"] == "Cooler"


def test_hotter_inflation_is_risk_off():
    o = classify_outcome("CPI", 0.6, 0.5)
    assert o["market_outcome"] == "risk_off" and o["outcome_label"] == "Hotter"


def test_lower_jobless_claims_is_risk_on():
    # Initial Jobless Claims 215 vs est 225 — FEWER claims = good = risk-on
    o = classify_outcome("JOBLESS_CLAIMS", 215.0, 225.0)
    assert o["market_outcome"] == "risk_on" and o["outcome_label"] == "Cooler"


# ── higher-is-better series ──────────────────────────────────────────────────────
def test_strong_gdp_is_risk_on():
    o = classify_outcome("GDP", 2.1, 1.6)
    assert o["market_outcome"] == "risk_on" and o["outcome_label"] == "Stronger"


def test_weak_payrolls_is_risk_off():
    o = classify_outcome("NFP", 120.0, 180.0)
    assert o["market_outcome"] == "risk_off" and o["outcome_label"] == "Weaker"


# ── in-line / neutral / pending / robustness ─────────────────────────────────────
def test_in_line_within_band():
    o = classify_outcome("CPI", 0.5, 0.5)
    assert o["market_outcome"] == "in_line" and o["outcome_label"] == "In-Line"


def test_neutral_event_reports_direction_only():
    # FOMC / unknown types: no risk-on/off claim, just above/below
    o = classify_outcome("FOMC", 5.5, 5.25)
    assert o["market_outcome"] == "in_line" and o["outcome_label"] == "Above est"
    assert polarity_for("FOMC") == "neutral"
    assert polarity_for("SOMETHING_UNMAPPED") == "neutral"


def test_pending_when_no_actual():
    assert classify_outcome("CPI", None, 0.5)["market_outcome"] == "pending"
    assert classify_outcome("CPI", 0.4, None)["market_outcome"] == "pending"
    assert classify_outcome("CPI", "n/a", 0.5)["market_outcome"] == "pending"   # never raises


def test_polarity_table_directions():
    for t in ("CPI", "PPI", "PCE", "UNEMPLOYMENT", "JOBLESS_CLAIMS"):
        assert polarity_for(t) == "lower_better"
    for t in ("NFP", "GDP", "RETAIL_SALES", "ISM_MFG", "ISM_SVC"):
        assert polarity_for(t) == "higher_better"


# ── /macro endpoint surfaces the outcome on each event ───────────────────────────
def test_macro_endpoint_enriches_events(monkeypatch):
    from app.api import nis_routes
    import types as _t
    from datetime import datetime, timezone

    ev = _t.SimpleNamespace(
        event_type="PCE", event_name="PCE Price Index MoM", event_time="2026-06-25T12:30:00",
        risk_level="HIGH", direction="NEUTRAL", sizing_factor=0.85, block_new_entries=False,
        consensus_summary="", rationale="", already_priced_in=True, actual=0.4, estimate=0.5,
        prior=0.4)
    ctx = _t.SimpleNamespace(as_of=datetime(2026, 6, 25, 13, 30, tzinfo=timezone.utc),
                             overall_risk="HIGH", block_new_entries=False,
                             global_sizing_factor=0.85, rationale="r", events_today=[ev])
    monkeypatch.setattr("app.agents.premarket.premarket_intel",
                        _t.SimpleNamespace(macro_context=ctx))
    out = nis_routes.get_macro_context()
    e0 = out["events_today"][0]
    assert e0["market_outcome"] == "risk_on" and e0["outcome_label"] == "Cooler"
