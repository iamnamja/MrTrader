"""
Tests for scripts/walkforward/registry_enforcement.py — the shared
--hypothesis-id contract the nine run_*_cpcv scripts use.

Coverage:
  - begin_run with a registered + preregistered confirmatory id returns a
    HypothesisRun carrying the run-START timestamp;
  - an unregistered id FAILS FAST (RegistryEnforcementError, before any fetch);
  - a confirmatory-but-not-preregistered id fails fast (R2 enforced pre-run);
  - an already-recorded id (R4) fails fast (the re-run could never record);
  - a run starting at/before the prereg instant fails fast (R2 ordering pre-run);
  - an exploratory-labeled id needs no pre-registration;
  - no id + exploratory=True -> None (unlimited exploratory runs);
  - no id before GRACE_UNTIL -> warn-only (printed) + None;
  - no id on/after GRACE_UNTIL -> raises (grace window over);
  - HypothesisRun.record is BEST-EFFORT: an R4 already-recorded second call
    prints a warning and returns False, never raises.

All tests isolate the registry via MRTRADER_RESEARCH_REGISTRY_DB (the
conftest pattern); the real data/research_registry.db is never touched.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from app.research.registry import ResearchRegistry
from scripts.walkforward.registry_enforcement import (
    GRACE_UNTIL,
    HypothesisRun,
    RegistryEnforcementError,
    begin_run,
)

PREREG = "2026-06-10T12:00:00+00:00"
CRITERIA = {"track_a_t": 2.0, "min_fold_sharpe": -0.3}
RUN_AT = datetime(2026, 6, 11, 3, 0, tzinfo=timezone.utc)


@pytest.fixture()
def reg(tmp_path, monkeypatch):
    """Isolated registry; begin_run/record construct ResearchRegistry() with
    no explicit path, so the env var is what they resolve."""
    monkeypatch.setenv("MRTRADER_RESEARCH_REGISTRY_DB", str(tmp_path / "reg.db"))
    return ResearchRegistry()


def _confirmatory_preregistered(reg, hid="HYP-OK"):
    reg.register(hid, label="confirmatory", family="pead")
    reg.preregister(hid, acceptance_criteria=CRITERIA, preregistered_at=PREREG)
    return hid


# ─────────────────────────────────────────────────── begin_run: with an id

def test_begin_run_confirmatory_preregistered_returns_run(reg):
    hid = _confirmatory_preregistered(reg)
    run = begin_run(hid, run_at=RUN_AT)
    assert isinstance(run, HypothesisRun)
    assert run.hypothesis_id == hid
    assert run.run_at == RUN_AT


def test_begin_run_unregistered_id_fails_fast(reg):
    with pytest.raises(RegistryEnforcementError, match="not registered"):
        begin_run("HYP-GHOST")


def test_begin_run_confirmatory_without_preregistration_fails_fast(reg):
    reg.register("HYP-NOPRE", label="confirmatory", family="pead")
    with pytest.raises(RegistryEnforcementError, match="never preregistered"):
        begin_run("HYP-NOPRE")


def test_begin_run_live_confirm_without_preregistration_fails_fast(reg):
    reg.register("HYP-LC", label="live_confirm")
    with pytest.raises(RegistryEnforcementError, match="never preregistered"):
        begin_run("HYP-LC")


def test_begin_run_exploratory_label_needs_no_preregistration(reg):
    reg.register("HYP-EXP", label="exploratory", family="pead")
    run = begin_run("HYP-EXP", run_at=RUN_AT)
    assert isinstance(run, HypothesisRun)
    assert run.hypothesis_id == "HYP-EXP"


def test_begin_run_defaults_run_at_to_call_time(reg):
    hid = _confirmatory_preregistered(reg, "HYP-NOW")
    before = datetime.now(timezone.utc)
    run = begin_run(hid)
    after = datetime.now(timezone.utc)
    assert before <= run.run_at <= after


def test_begin_run_already_recorded_fails_fast(reg):
    # R4 one-shot: a re-run of a hypothesis that already has a result must
    # refuse to START — its result could never be recorded, so the expensive
    # CPCV would be burned for nothing (the swallowed record() would warn).
    hid = _confirmatory_preregistered(reg, "HYP-DONE")
    begin_run(hid, run_at=RUN_AT).record({"mean_sharpe": 0.3})
    with pytest.raises(RegistryEnforcementError, match="already has a recorded result"):
        begin_run(hid, run_at=RUN_AT + timedelta(hours=1))


def test_begin_run_run_before_preregistration_fails_fast(reg):
    # R2 ordering: a run that STARTS at/before the criteria-freeze instant can
    # never be recorded; refuse it up front rather than discover it at record().
    hid = _confirmatory_preregistered(reg, "HYP-EARLY")
    early = datetime.fromisoformat(PREREG) - timedelta(seconds=1)
    with pytest.raises(RegistryEnforcementError, match="not strictly after"):
        begin_run(hid, run_at=early)


# ───────────────────────────────────────────────── begin_run: without an id

def test_begin_run_none_exploratory_returns_none(reg, capsys):
    assert begin_run(None, exploratory=True) is None
    # Silent: exploratory runs are unlimited, nothing to warn about.
    assert "[registry]" not in capsys.readouterr().out


def test_begin_run_none_before_grace_warns_and_returns_none(reg, capsys):
    as_of = GRACE_UNTIL - timedelta(days=1)
    assert begin_run(None, as_of=as_of) is None
    out = capsys.readouterr().out
    assert "[registry] WARNING" in out
    assert GRACE_UNTIL.isoformat() in out


@pytest.mark.parametrize("as_of", [GRACE_UNTIL, GRACE_UNTIL + timedelta(days=30)])
def test_begin_run_none_on_or_after_grace_raises(reg, as_of):
    with pytest.raises(RegistryEnforcementError, match="required"):
        begin_run(None, as_of=as_of)


def test_grace_until_matches_blueprint_two_week_window():
    # Blueprint line: "warn-only for 2 weeks" from the 2026-06-11 ship date.
    assert GRACE_UNTIL == date(2026, 6, 25)


# ─────────────────────────────────────────── HypothesisRun.record best-effort

def test_record_writes_result_and_run_at(reg):
    hid = _confirmatory_preregistered(reg, "HYP-REC")
    run = begin_run(hid, run_at=RUN_AT)
    assert run.record({"mean_sharpe": 0.42, "gate_ok": False}) is True
    row = reg.get(hid)
    assert row["run_at"] == RUN_AT.isoformat()
    assert row["result_json"] == {"mean_sharpe": 0.42, "gate_ok": False}
    assert row["decision"] is None  # promotion is owner-gated, never auto


def test_record_second_shot_r4_is_swallowed_with_warning(reg, capsys):
    hid = _confirmatory_preregistered(reg, "HYP-R4")
    run = begin_run(hid, run_at=RUN_AT)
    assert run.record({"mean_sharpe": 0.42}) is True
    # Second record on the same hypothesis violates R4 — must NOT raise (a
    # completed multi-hour CPCV is never lost to a registry hiccup).
    assert run.record({"mean_sharpe": 0.99}) is False
    out = capsys.readouterr().out
    assert "WARNING" in out and "NOT recorded" in out
    # First result untouched.
    assert reg.get(hid)["result_json"] == {"mean_sharpe": 0.42}


def test_record_on_nonexistent_hypothesis_is_swallowed(reg, capsys):
    # Constructed directly (bypassing begin_run) — record must still not raise.
    run = HypothesisRun("HYP-NEVER-REGISTERED", RUN_AT)
    assert run.record({"mean_sharpe": 0.1}) is False
    assert "NOT recorded" in capsys.readouterr().out


def test_record_coerces_numpy_and_tuples_to_jsonable(reg):
    np = pytest.importorskip("numpy")
    hid = _confirmatory_preregistered(reg, "HYP-NP")
    run = begin_run(hid, run_at=RUN_AT)
    assert run.record({
        "mean_sharpe": np.float64(0.5),
        "n_paths": np.int64(28),
        "gate_detail": {"tstat": (np.float64(1.2), np.bool_(False))},
    }) is True
    got = reg.get(hid)["result_json"]
    assert got["mean_sharpe"] == 0.5
    assert got["n_paths"] == 28
    assert got["gate_detail"]["tstat"] == [1.2, False]
