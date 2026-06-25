"""Macro Intel Phase 2b — macro risk steps DOWN once benign results land, with a deterministic floor.

Fixes "risk stays HIGH all day": the prompt no longer says "HIGH but released → 0.85 forever"; a
benign all-released day → LOW/1.0. A deterministic day-level floor (aggregate_day) is the LLM
fail-safe + clamp: it RAISES risk/block and LOWERS sizing (never less conservative), while the LLM
drives the step-DOWN when the floor permits it.
"""
from __future__ import annotations

from app.news.macro_polarity import aggregate_day, clamp_to_floor


def _evt(et, importance="high", actual=None, estimate=None):
    return {"event_type": et, "importance": importance, "actual": actual, "estimate": estimate}


# ── aggregate_day deterministic floor ────────────────────────────────────────────
def test_floor_unreleased_high_blocks():
    f = aggregate_day([_evt("CPI", actual=None, estimate=0.3)])
    assert f["min_risk"] == "HIGH" and f["block"] is True and f["max_sizing"] == 0.75


def test_floor_all_released_benign_steps_down():
    # cooler PCE (risk-on) + strong GDP (risk-on), all released → LOW / 1.0 (the step-down)
    f = aggregate_day([_evt("PCE", actual=0.3, estimate=0.4), _evt("GDP", actual=2.1, estimate=1.6)])
    assert f["min_risk"] == "LOW" and f["block"] is False and f["max_sizing"] == 1.0
    assert f["all_released"] is True and f["net_lean"] == "BULLISH"


def test_floor_released_adverse_stays_medium():
    # hotter CPI (risk-off) → MEDIUM, no block (outcome known), sizing <=0.85
    f = aggregate_day([_evt("CPI", actual=0.6, estimate=0.4)])
    assert f["min_risk"] == "MEDIUM" and f["block"] is False and f["max_sizing"] == 0.85
    assert f["net_lean"] == "BEARISH"


def test_floor_no_high_impact_is_low():
    f = aggregate_day([_evt("OTHER_HIGH", importance="medium", actual=1, estimate=1)])
    assert f["min_risk"] == "LOW" and f["block"] is False


# ── clamp: floor only raises risk/block, lowers sizing ───────────────────────────
def test_clamp_raises_to_floor():
    # LLM wrongly says LOW/no-block/1.0 but an unreleased high-impact event exists → clamp to HIGH/block
    floor = aggregate_day([_evt("NFP", actual=None, estimate=180)])
    c = clamp_to_floor("LOW", False, 1.0, floor)
    assert c["risk_level"] == "HIGH" and c["block_new_entries"] is True and c["sizing_factor"] == 0.75


def test_clamp_allows_stepdown_when_floor_benign():
    # benign floor (LOW) lets the LLM's LOW/1.0 stand — the step-down is permitted
    floor = aggregate_day([_evt("PCE", actual=0.3, estimate=0.4)])
    c = clamp_to_floor("LOW", False, 1.0, floor)
    assert c["risk_level"] == "LOW" and c["block_new_entries"] is False and c["sizing_factor"] == 1.0


def test_clamp_never_less_conservative():
    # LLM MEDIUM vs floor LOW → keep MEDIUM (max); sizing min
    floor = aggregate_day([_evt("PCE", actual=0.3, estimate=0.4)])
    c = clamp_to_floor("MEDIUM", True, 0.7, floor)
    assert c["risk_level"] == "MEDIUM" and c["block_new_entries"] is True and c["sizing_factor"] == 0.7


# ── _build_macro_context integration ─────────────────────────────────────────────
def _svc():
    from app.news.intelligence_service import NewsIntelligenceService
    return NewsIntelligenceService()


def test_build_context_clamps_stubborn_llm(monkeypatch):
    svc = _svc()
    # unreleased high-impact event; LLM stubbornly returns LOW/no-block → must clamp to HIGH/block
    monkeypatch.setattr("app.news.sources.economic_calendar.fetch_economic_calendar",
                        lambda **k: [_evt("CPI", actual=None, estimate=0.3)])
    monkeypatch.setattr("app.news.llm_scorer.macro_classify", lambda evts: {
        "risk_level": "LOW", "block_new_entries": False, "sizing_factor": 1.0, "rationale": "meh"})
    monkeypatch.setattr(svc, "_persist_macro", lambda *a, **k: None)
    monkeypatch.setattr(svc, "_build_event_signals", lambda evts, res: [])
    ctx = svc._build_macro_context()
    assert ctx.overall_risk == "HIGH" and ctx.block_new_entries is True


def test_build_context_allows_benign_stepdown(monkeypatch):
    svc = _svc()
    # all released benign; LLM returns LOW/1.0 → stands (the step-down works)
    monkeypatch.setattr("app.news.sources.economic_calendar.fetch_economic_calendar",
                        lambda **k: [_evt("PCE", actual=0.3, estimate=0.4)])
    monkeypatch.setattr("app.news.llm_scorer.macro_classify", lambda evts: {
        "risk_level": "LOW", "block_new_entries": False, "sizing_factor": 1.0,
        "rationale": "digested, benign"})
    monkeypatch.setattr(svc, "_persist_macro", lambda *a, **k: None)
    monkeypatch.setattr(svc, "_build_event_signals", lambda evts, res: [])
    ctx = svc._build_macro_context()
    assert ctx.overall_risk == "LOW" and ctx.global_sizing_factor == 1.0


def test_build_context_llm_unavailable_uses_floor(monkeypatch):
    svc = _svc()
    # LLM returns None + an unreleased high-impact event → deterministic floor (HIGH/block), not neutral
    monkeypatch.setattr("app.news.sources.economic_calendar.fetch_economic_calendar",
                        lambda **k: [_evt("NFP", actual=None, estimate=180)])
    monkeypatch.setattr("app.news.llm_scorer.macro_classify", lambda evts: None)
    monkeypatch.setattr(svc, "_persist_macro", lambda *a, **k: None)
    ctx = svc._build_macro_context()
    assert ctx.overall_risk == "HIGH" and ctx.block_new_entries is True


def test_build_context_macro_classify_raise_uses_floor(monkeypatch):
    svc = _svc()
    # macro_classify RAISES (not clean None) + unreleased high-impact → must use the floor (HIGH/block),
    # NOT fall through to flat neutral (the closed invariant gap)
    monkeypatch.setattr("app.news.sources.economic_calendar.fetch_economic_calendar",
                        lambda **k: [_evt("CPI", actual=None, estimate=0.3)])

    def _boom(evts):
        raise RuntimeError("LLM exploded")
    monkeypatch.setattr("app.news.llm_scorer.macro_classify", _boom)
    monkeypatch.setattr(svc, "_persist_macro", lambda *a, **k: None)
    ctx = svc._build_macro_context()
    assert ctx.overall_risk == "HIGH" and ctx.block_new_entries is True


def test_build_context_unavailable_calendar_still_conservative(monkeypatch):
    svc = _svc()
    monkeypatch.setattr("app.news.sources.economic_calendar.fetch_economic_calendar",
                        lambda **k: None)
    ctx = svc._build_macro_context()
    assert ctx.overall_risk == "MEDIUM"  # unchanged Wave-5g behavior
