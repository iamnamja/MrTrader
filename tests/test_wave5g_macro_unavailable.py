"""Alpha-v10 audit Wave 5g — macro calendar UNAVAILABLE must not fail OPEN.

Pin: the economic-calendar chain now distinguishes "FMP unavailable" (None) from a genuine
"no-events day" ([]). When the calendar is truly unavailable, the NIS macro context applies a
CONSERVATIVE stance (size down) instead of the old fail-OPEN ("no events -> trade freely").
"""
from __future__ import annotations


# ── economic_calendar returns None ONLY when genuinely unavailable ───────────────
def test_calendar_fmp_empty_is_authoritative_no_events(monkeypatch):
    from app.news.sources import economic_calendar as ec
    # FMP reachable, returns [] -> a genuine no-events day (NOT None)
    monkeypatch.setattr("app.news.sources.fmp_source.fetch_economic_calendar",
                        lambda **k: [])
    assert ec.fetch_economic_calendar() == []


def test_calendar_unavailable_returns_none(monkeypatch):
    from app.news.sources import economic_calendar as ec
    # FMP unavailable (None) AND Finnhub yields nothing -> UNKNOWN -> None
    monkeypatch.setattr("app.news.sources.fmp_source.fetch_economic_calendar",
                        lambda **k: None)
    monkeypatch.setattr("app.news.sources.finnhub_source.fetch_economic_calendar",
                        lambda **k: [])
    assert ec.fetch_economic_calendar() is None


def test_calendar_finnhub_fallback_used(monkeypatch):
    from app.news.sources import economic_calendar as ec
    evts = [{"id": "x", "event_type": "CPI"}]
    monkeypatch.setattr("app.news.sources.fmp_source.fetch_economic_calendar",
                        lambda **k: None)
    monkeypatch.setattr("app.news.sources.finnhub_source.fetch_economic_calendar",
                        lambda **k: evts)
    assert ec.fetch_economic_calendar() == evts


# ── NIS macro context: unavailable -> conservative (NOT fail-open) ───────────────
def test_macro_context_unavailable_is_conservative(monkeypatch):
    from app.news import intelligence_service as nis_mod
    monkeypatch.setattr(
        "app.news.sources.economic_calendar.fetch_economic_calendar", lambda **k: None)
    svc = nis_mod.NewsIntelligenceService()
    ctx = svc._build_macro_context()
    # NOT the old fail-open (sizing 1.0, block False, risk LOW). Default policy = FAIL CLOSED:
    # block new entries (protects both swing+intraday via the trader gate) + size down intraday.
    assert ctx.global_sizing_factor == nis_mod.MACRO_UNAVAILABLE_SIZING < 1.0
    assert ctx.block_new_entries is True
    assert nis_mod.MACRO_UNAVAILABLE_BLOCK is True
    assert ctx.overall_risk == "MEDIUM"
    assert "unavailable" in ctx.rationale.lower()


def test_macro_context_no_events_still_trades_freely(monkeypatch):
    # a genuine no-events day ([], not None) must remain a no-op (size 1.0, no block) — we only
    # tightened the UNAVAILABLE case, not legitimate quiet days.
    from app.news import intelligence_service as nis_mod
    monkeypatch.setattr(
        "app.news.sources.economic_calendar.fetch_economic_calendar", lambda **k: [])
    monkeypatch.setattr("app.news.llm_scorer.macro_classify", lambda evts: {
        "risk_level": "LOW", "direction": "NEUTRAL", "sizing_factor": 1.0,
        "block_new_entries": False, "rationale": "No significant economic events today"})
    svc = nis_mod.NewsIntelligenceService()
    ctx = svc._build_macro_context()
    assert ctx.global_sizing_factor == 1.0
    assert ctx.block_new_entries is False


# ── the post-event refresh loop tolerates an unavailable (None) calendar ─────────
def test_post_event_refresh_handles_none_calendar(monkeypatch):
    import asyncio
    from datetime import datetime, timezone
    from app.agents.portfolio_manager import PortfolioManager
    pm = PortfolioManager.__new__(PortfolioManager)
    pm.logger = __import__("logging").getLogger("t")
    pm._nis_event_refresh_state = {}
    monkeypatch.setattr(
        "app.news.sources.economic_calendar.fetch_economic_calendar", lambda **k: None)
    # must not raise (was: `for evt in events` over None -> TypeError)
    asyncio.run(pm._maybe_refresh_nis_post_event(datetime.now(timezone.utc)))
