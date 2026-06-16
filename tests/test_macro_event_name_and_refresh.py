"""Tests for the macro-event panel fixes:
  1. event_name is threaded through MacroEventSignal -> snapshot/API (surface the
     real vendor name so OTHER_HIGH rows aren't opaque and same-type rows split).
  2. decide_post_event_refresh: the NIS post-event refresh retries until the vendor
     ACTUAL lands (fixes "Released but Actual = —"), rebuilding at most twice/event.
"""
from app.agents.portfolio_manager import decide_post_event_refresh


# ── Change 2: refresh decision logic ─────────────────────────────────────────
DELAY = 3.0


def test_no_refresh_before_settle_delay():
    should, _, _ = decide_post_event_refresh(
        {}, "e1", elapsed_min=1.0, actual_present=False, delay_min=DELAY)
    assert should is False


def test_first_refresh_fires_after_delay_even_without_actual():
    # preserves the old behavior: a post-release rebuild at +delay (LLM reassessment)
    should, captured, first = decide_post_event_refresh(
        {}, "e1", elapsed_min=4.0, actual_present=False, delay_min=DELAY)
    assert should is True and first is True and captured is False


def test_first_refresh_captures_actual_when_already_present():
    should, captured, first = decide_post_event_refresh(
        {}, "e1", elapsed_min=4.0, actual_present=True, delay_min=DELAY)
    assert should is True and first is True and captured is True


def test_retries_until_lagging_actual_lands():
    state = {}
    # +4min: first refresh, no actual yet -> fires, not captured
    should, captured, _ = decide_post_event_refresh(state, "e1", 4.0, False, DELAY)
    assert should is True and captured is False
    state["e1"] = captured
    # +20min: still no actual -> do NOT rebuild again, keep waiting
    should, _, _ = decide_post_event_refresh(state, "e1", 20.0, False, DELAY)
    assert should is False
    # +45min: actual finally lands -> rebuild once more, mark captured
    should, captured, first = decide_post_event_refresh(state, "e1", 45.0, True, DELAY)
    assert should is True and captured is True and first is False
    state["e1"] = captured
    # +50min: already captured -> never rebuild again (cap = 2 rebuilds/event)
    should, _, _ = decide_post_event_refresh(state, "e1", 50.0, True, DELAY)
    assert should is False


def test_no_rebuild_once_actual_captured():
    state = {"e1": True}
    should, _, _ = decide_post_event_refresh(state, "e1", 30.0, True, DELAY)
    assert should is False


# ── Change 1: event_name threading ───────────────────────────────────────────
def test_macro_event_signal_carries_event_name():
    from app.news.signal import MacroEventSignal
    from datetime import datetime, timezone
    s = MacroEventSignal(
        event_type="RETAIL_SALES", event_time="08:30 ET", risk_level="HIGH",
        direction="NEUTRAL", sizing_factor=1.0, block_new_entries=False,
        consensus_summary="", rationale="", scorer_tier="haiku",
        evaluated_at=datetime.now(timezone.utc),
        event_name="Retail Sales Ex Autos MoM (May)",
    )
    assert s.event_name == "Retail Sales Ex Autos MoM (May)"


def test_macro_event_signal_event_name_defaults_empty():
    from app.news.signal import MacroEventSignal
    from datetime import datetime, timezone
    s = MacroEventSignal(
        event_type="CPI", event_time="08:30 ET", risk_level="HIGH",
        direction="NEUTRAL", sizing_factor=1.0, block_new_entries=False,
        consensus_summary="", rationale="", scorer_tier="haiku",
        evaluated_at=datetime.now(timezone.utc),
    )
    assert s.event_name == ""


def test_event_uid_distinct_for_same_type_same_time_different_name():
    """Paired same-type same-time releases (Retail Sales MoM vs Ex-Autos) and multiple
    OTHER_HIGH at 08:30 must get DISTINCT ids, else one starves the refresh retry."""
    from app.news.sources.finnhub_source import event_uid
    t = "2026-06-16 12:30:00"
    a = event_uid("RETAIL_SALES", "Retail Sales MoM (May)", t)
    b = event_uid("RETAIL_SALES", "Retail Sales Ex Autos MoM (May)", t)
    c = event_uid("OTHER_HIGH", "Empire State Manufacturing Index", t)
    d = event_uid("OTHER_HIGH", "Import Prices MoM", t)
    assert len({a, b, c, d}) == 4              # all distinct
    assert event_uid("CPI", "CPI MoM", t) == event_uid("CPI", "CPI MoM", t)  # stable


def test_snapshot_persister_includes_event_name():
    """persist_nis_macro_snapshot must serialize event_name into events_json so the
    DB-snapshot fallback path (API/UI after a restart) also has the real name."""
    import inspect
    from app.database import decision_audit
    src = inspect.getsource(decision_audit.persist_nis_macro_snapshot)
    assert '"event_name"' in src
