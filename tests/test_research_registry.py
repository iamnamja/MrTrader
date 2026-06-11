"""
Tests for app/research/registry.py — the pre-registration ledger.

Every integrity rule (R1-R5) has a dedicated test. All tests use a temp-dir
sqlite file (or the conftest-isolated env var); the real
data/research_registry.db is never touched.
"""
from __future__ import annotations

import sqlite3

import pytest

from app.research.registry import RegistryIntegrityError, ResearchRegistry

PREREG = "2026-06-09T12:00:00+00:00"
RUN = "2026-06-10T03:00:00+00:00"
CRITERIA = {"track_a_t": 2.0, "min_fold_sharpe": -0.3}


@pytest.fixture()
def reg(tmp_path):
    return ResearchRegistry(db_path=str(tmp_path / "reg.db"))


def _confirmatory(reg, hid="HYP-C1", **kw):
    return reg.register(hid, label="confirmatory", family="pead", **kw)


def _raw_insert(reg, hypothesis_id, **cols):
    """Insert a row bypassing the API — simulates schema drift / manual edits."""
    cols = {"hypothesis_id": hypothesis_id, **cols}
    names = ", ".join(cols)
    ph = ", ".join("?" for _ in cols)
    c = reg._conn()  # ensures the schema exists on a fresh temp DB
    try:
        with c:
            c.execute(f"INSERT INTO experiments ({names}) VALUES ({ph})",
                      tuple(cols.values()))
    finally:
        c.close()


# --------------------------------------------------------------------- basics

def test_register_and_get_roundtrip(reg):
    row = reg.register(
        "HYP-1",
        label="exploratory",
        family="trend",
        features=["mom_12m", "vol_20d"],
        params={"lookback": 252, "vol_target": 0.1},
        universe="sp500",
        window="2019-2025",
        folds="8",
        cost_model="5bps",
        code_commit="abc1234",
        data_hash="deadbeef",
        mechanism="time-series momentum persists",
    )
    assert row["hypothesis_id"] == "HYP-1"
    assert row["label"] == "exploratory"
    assert row["run_at"] is None and row["decision"] is None
    got = reg.get("HYP-1")
    # JSON columns round-trip as Python objects.
    assert got["params"] == {"lookback": 252, "vol_target": 0.1}
    assert got["features"] == ["mom_12m", "vol_20d"]
    assert got["created_at"] is not None


def test_get_unknown_returns_none(reg):
    assert reg.get("nope") is None


def test_invalid_label_raises(reg):
    with pytest.raises(RegistryIntegrityError, match="label"):
        reg.register("HYP-X", label="speculative")


# ----------------------------------------------------------- R1: duplicate id

def test_duplicate_hypothesis_id_raises(reg):
    reg.register("HYP-DUP", label="exploratory")
    with pytest.raises(RegistryIntegrityError, match=r"R1.*already.*registered"):
        reg.register("HYP-DUP", label="exploratory")


# --------------------------------------------- R2: confirmatory pre-reg order

def test_confirmatory_happy_path_promotes(reg):
    _confirmatory(reg)
    reg.preregister("HYP-C1", acceptance_criteria=CRITERIA, preregistered_at=PREREG)
    row = reg.record_result(
        "HYP-C1", run_at=RUN, result={"sharpe": 0.62}, decision="promote_paper"
    )
    assert row["decision"] == "promote_paper"
    assert row["result_json"] == {"sharpe": 0.62}
    assert row["acceptance_criteria"] == CRITERIA
    assert row["preregistered_at"] == PREREG


def test_confirmatory_without_preregistration_raises(reg):
    _confirmatory(reg, "HYP-C2")
    with pytest.raises(RegistryIntegrityError, match=r"R2.*never preregistered"):
        reg.record_result("HYP-C2", run_at=RUN, decision="kill")


def test_confirmatory_preregistered_after_run_raises(reg):
    _confirmatory(reg, "HYP-C3")
    reg.preregister(
        "HYP-C3", acceptance_criteria=CRITERIA,
        preregistered_at="2026-06-11T00:00:00+00:00",  # AFTER the run
    )
    with pytest.raises(RegistryIntegrityError, match=r"R2.*not strictly before"):
        reg.record_result("HYP-C3", run_at=RUN, decision="kill")


def test_confirmatory_preregistered_equal_to_run_raises(reg):
    _confirmatory(reg, "HYP-C4")
    reg.preregister("HYP-C4", acceptance_criteria=CRITERIA, preregistered_at=RUN)
    with pytest.raises(RegistryIntegrityError, match=r"R2"):
        reg.record_result("HYP-C4", run_at=RUN, decision="kill")


def test_live_confirm_also_requires_preregistration(reg):
    reg.register("HYP-LC", label="live_confirm")
    with pytest.raises(RegistryIntegrityError, match=r"R2"):
        reg.record_result("HYP-LC", run_at=RUN, decision="live")


def test_tz_naive_prereg_vs_aware_run_compares_correctly(reg):
    # Naive is treated as UTC: 2026-06-09T12:00 (naive) < RUN (aware) -> OK.
    _confirmatory(reg, "HYP-TZ")
    reg.preregister(
        "HYP-TZ", acceptance_criteria=CRITERIA,
        preregistered_at="2026-06-09T12:00:00",  # tz-naive
    )
    row = reg.record_result("HYP-TZ", run_at=RUN, decision="kill")
    assert row["decision"] == "kill"


# ------------------------------------------- R3: exploratory cannot promote

@pytest.mark.parametrize("bad", ["promote_paper", "live"])
def test_exploratory_cannot_promote(reg, bad):
    reg.register("HYP-E1", label="exploratory")
    with pytest.raises(RegistryIntegrityError, match=r"R3.*cannot promote"):
        reg.record_result("HYP-E1", run_at=RUN, decision=bad)
    # The blocked attempt must not have consumed the one shot.
    assert reg.get("HYP-E1")["run_at"] is None


@pytest.mark.parametrize("ok", ["kill", "park", "exploratory_only"])
def test_exploratory_allowed_decisions(reg, ok):
    reg.register(f"HYP-E-{ok}", label="exploratory")
    row = reg.record_result(f"HYP-E-{ok}", run_at=RUN, decision=ok)
    assert row["decision"] == ok


def test_invalid_decision_raises(reg):
    reg.register("HYP-E2", label="exploratory")
    with pytest.raises(RegistryIntegrityError, match="decision"):
        reg.record_result("HYP-E2", run_at=RUN, decision="ship_it")


# --------------------------------------------------- R4: one shot + re-tests

def test_second_result_on_same_hypothesis_raises(reg):
    _confirmatory(reg, "HYP-ONE")
    reg.preregister("HYP-ONE", acceptance_criteria=CRITERIA, preregistered_at=PREREG)
    reg.record_result("HYP-ONE", run_at=RUN, decision="kill")
    with pytest.raises(RegistryIntegrityError, match=r"R4 one shot"):
        reg.record_result("HYP-ONE", run_at="2026-06-12T00:00:00+00:00",
                          decision="promote_paper")


def test_retest_with_past_cooling_off_is_allowed(reg):
    _confirmatory(reg, "HYP-P")
    reg.preregister("HYP-P", acceptance_criteria=CRITERIA, preregistered_at=PREREG)
    reg.record_result("HYP-P", run_at=RUN, decision="park")
    # Sanctioned re-test: NEW id, parent_id set, cooling-off in the past.
    reg.register(
        "HYP-P-RT1", label="confirmatory", parent_id="HYP-P",
        cooling_off_until="2026-07-01T00:00:00+00:00",
    )
    reg.preregister("HYP-P-RT1", acceptance_criteria=CRITERIA,
                    preregistered_at="2026-07-05T00:00:00+00:00")
    row = reg.record_result(
        "HYP-P-RT1", run_at="2026-07-10T00:00:00+00:00", decision="promote_paper",
    )
    assert row["decision"] == "promote_paper"
    assert row["parent_id"] == "HYP-P"


def test_retest_run_before_cooling_off_raises(reg):
    reg.register("HYP-P", label="exploratory")
    reg.register(
        "HYP-RT-EARLY", label="confirmatory", parent_id="HYP-P",
        cooling_off_until="2026-08-01T00:00:00+00:00",
    )
    reg.preregister("HYP-RT-EARLY", acceptance_criteria=CRITERIA,
                    preregistered_at="2026-07-01T00:00:00+00:00")
    with pytest.raises(RegistryIntegrityError,
                       match=r"R4 re-test run must execute after the cooling-off"):
        reg.record_result("HYP-RT-EARLY", run_at="2026-07-10T00:00:00+00:00",
                          decision="kill")  # run executed before cooling-off end


def test_retest_cooling_off_cannot_be_bypassed_with_late_now(reg):
    """F3: the old `now` escape hatch is gone — a run that EXECUTED inside the
    cooling-off window cannot be recorded by claiming a later recording time."""
    reg.register("HYP-P", label="exploratory")
    reg.register(
        "HYP-RT-NOW", label="confirmatory", parent_id="HYP-P",
        cooling_off_until="2026-08-01T00:00:00+00:00",
    )
    reg.preregister("HYP-RT-NOW", acceptance_criteria=CRITERIA,
                    preregistered_at="2026-07-01T00:00:00+00:00")
    # The `now` kwarg no longer exists (it was the bypass).
    with pytest.raises(TypeError):
        reg.record_result("HYP-RT-NOW", run_at="2026-07-10T00:00:00+00:00",
                          decision="kill", now="2026-09-01T00:00:00+00:00")
    # And the check itself is against run_at, regardless of wall-clock.
    with pytest.raises(RegistryIntegrityError,
                       match=r"R4 re-test run must execute after the cooling-off"):
        reg.record_result("HYP-RT-NOW", run_at="2026-07-10T00:00:00+00:00",
                          decision="kill")


def test_retest_without_cooling_off_raises(reg):
    reg.register("HYP-P", label="exploratory")
    reg.register("HYP-RT-NOCOOL", label="exploratory", parent_id="HYP-P")
    with pytest.raises(RegistryIntegrityError, match=r"R4 re-test.*no cooling_off"):
        reg.record_result("HYP-RT-NOCOOL", run_at=RUN, decision="kill")


# ----------------------------------------------------- R5: preregister rules

def test_preregister_after_result_raises(reg):
    reg.register("HYP-PH", label="exploratory")
    reg.record_result("HYP-PH", run_at=RUN, decision="kill")
    with pytest.raises(RegistryIntegrityError, match=r"R5 post-hoc"):
        reg.preregister("HYP-PH", acceptance_criteria=CRITERIA,
                        preregistered_at="2026-06-12T00:00:00+00:00")


def test_repreregistration_raises(reg):
    _confirmatory(reg, "HYP-IMM")
    reg.preregister("HYP-IMM", acceptance_criteria=CRITERIA, preregistered_at=PREREG)
    with pytest.raises(RegistryIntegrityError, match=r"R5 criteria immutable"):
        reg.preregister("HYP-IMM", acceptance_criteria={"track_a_t": 1.0},
                        preregistered_at=PREREG)


def test_preregister_unknown_hypothesis_raises(reg):
    with pytest.raises(RegistryIntegrityError, match="not registered"):
        reg.preregister("ghost", acceptance_criteria=CRITERIA, preregistered_at=PREREG)


def test_record_result_unknown_hypothesis_raises(reg):
    with pytest.raises(RegistryIntegrityError, match="not registered"):
        reg.record_result("ghost", run_at=RUN, decision="kill")


def test_bad_timestamp_raises(reg):
    reg.register("HYP-TS", label="exploratory")
    with pytest.raises(RegistryIntegrityError, match="ISO-8601"):
        reg.record_result("HYP-TS", run_at="next tuesday", decision="kill")


# ------------------------------------------------------- list() and summary()

def test_list_filters(reg):
    reg.register("A", label="exploratory", family="pead")
    reg.register("B", label="confirmatory", family="pead")
    reg.register("C", label="exploratory", family="trend")
    assert {r["hypothesis_id"] for r in reg.list(label="exploratory")} == {"A", "C"}
    assert {r["hypothesis_id"] for r in reg.list(family="pead")} == {"A", "B"}
    assert [r["hypothesis_id"] for r in reg.list(label="exploratory", family="pead")] == ["A"]
    assert len(reg.list()) == 3


def test_summary_counts(reg):
    reg.register("S1", label="exploratory")
    reg.register("S2", label="exploratory")
    _confirmatory(reg, "S3")
    reg.record_result("S1", run_at=RUN, decision="kill")
    reg.record_result("S2", run_at=RUN, decision="exploratory_only")
    s = reg.summary()
    assert s["total"] == 3
    assert s["by_label"] == {"exploratory": 2, "confirmatory": 1}
    assert s["by_decision"] == {"kill": 1, "exploratory_only": 1, "pending": 1}


# ------------------------------------------------------------- env isolation

def test_env_var_override_respected(tmp_path, monkeypatch):
    db = tmp_path / "env_override.db"
    monkeypatch.setenv("MRTRADER_RESEARCH_REGISTRY_DB", str(db))
    r = ResearchRegistry()  # no explicit path -> env var wins
    assert r.db_path == db
    r.register("HYP-ENV", label="exploratory")
    assert db.exists()
    assert r.get("HYP-ENV")["label"] == "exploratory"


def test_explicit_path_beats_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MRTRADER_RESEARCH_REGISTRY_DB", str(tmp_path / "env.db"))
    explicit = tmp_path / "explicit.db"
    r = ResearchRegistry(db_path=str(explicit))
    assert r.db_path == explicit


# -------------------------- F2: pre-registration requires non-empty criteria

def test_register_confirmatory_preregistered_without_criteria_raises(reg):
    with pytest.raises(RegistryIntegrityError,
                       match=r"R2 pre-registration requires acceptance criteria"):
        reg.register("HYP-NC1", label="confirmatory", preregistered_at=PREREG)


def test_register_confirmatory_preregistered_with_empty_criteria_raises(reg):
    with pytest.raises(RegistryIntegrityError,
                       match=r"R2 pre-registration requires acceptance criteria"):
        reg.register("HYP-NC2", label="confirmatory", preregistered_at=PREREG,
                     acceptance_criteria={})


def test_register_live_confirm_preregistered_without_criteria_raises(reg):
    with pytest.raises(RegistryIntegrityError,
                       match=r"R2 pre-registration requires acceptance criteria"):
        reg.register("HYP-NC3", label="live_confirm", preregistered_at=PREREG)


def test_register_confirmatory_preregistered_with_criteria_ok(reg):
    row = reg.register("HYP-OK1", label="confirmatory", preregistered_at=PREREG,
                       acceptance_criteria=CRITERIA)
    assert row["acceptance_criteria"] == CRITERIA
    out = reg.record_result("HYP-OK1", run_at=RUN, decision="promote_paper")
    assert out["decision"] == "promote_paper"


@pytest.mark.parametrize("empty", [None, {}, ""])
def test_preregister_with_empty_criteria_raises(reg, empty):
    _confirmatory(reg, "HYP-NC4")
    with pytest.raises(RegistryIntegrityError,
                       match=r"R2 pre-registration requires acceptance criteria"):
        reg.preregister("HYP-NC4", acceptance_criteria=empty, preregistered_at=PREREG)
    # The blocked attempt must not have preregistered the row.
    assert reg.get("HYP-NC4")["preregistered_at"] is None


def test_record_result_defense_raises_on_stored_empty_criteria(reg):
    # Unreachable via the public API — simulate a manual DB edit.
    _raw_insert(reg, "HYP-NC5", label="confirmatory", preregistered_at=PREREG,
                acceptance_criteria=None)
    with pytest.raises(RegistryIntegrityError,
                       match=r"R2 pre-registration requires acceptance criteria"):
        reg.record_result("HYP-NC5", run_at=RUN, decision="promote_paper")
    assert reg.get("HYP-NC5")["run_at"] is None  # nothing committed


# ------------------------------- F4: guarded conditional UPDATE (TOCTOU race)

def test_record_result_concurrent_writer_hits_rowcount_guard(reg, monkeypatch):
    """Simulate a stale second writer: pre-checks see a clean row (stale read)
    but the DB already has a result — the conditional UPDATE must match 0 rows
    and raise, never overwrite."""
    _confirmatory(reg, "HYP-RACE")
    reg.preregister("HYP-RACE", acceptance_criteria=CRITERIA, preregistered_at=PREREG)
    clean_row = reg.get("HYP-RACE")  # snapshot BEFORE the first writer commits
    reg.record_result("HYP-RACE", run_at=RUN, result={"sharpe": 0.5}, decision="kill")
    monkeypatch.setattr(ResearchRegistry, "_fetch_or_raise",
                        lambda self, c, hid: dict(clean_row))
    with pytest.raises(RegistryIntegrityError, match=r"R4 one shot.*concurrent"):
        reg.record_result("HYP-RACE", run_at="2026-06-12T00:00:00+00:00",
                          decision="promote_paper")
    # First writer's result is untouched.
    final = reg.get("HYP-RACE")
    assert final["run_at"] == RUN and final["decision"] == "kill"


def test_preregister_concurrent_writer_hits_rowcount_guard(reg, monkeypatch):
    _confirmatory(reg, "HYP-RACE2")
    clean_row = reg.get("HYP-RACE2")
    reg.preregister("HYP-RACE2", acceptance_criteria=CRITERIA, preregistered_at=PREREG)
    monkeypatch.setattr(ResearchRegistry, "_fetch_or_raise",
                        lambda self, c, hid: dict(clean_row))
    with pytest.raises(RegistryIntegrityError, match=r"R5 criteria immutable.*concurrent"):
        reg.preregister("HYP-RACE2", acceptance_criteria={"track_a_t": 0.1},
                        preregistered_at="2026-06-09T13:00:00+00:00")
    # Original pre-registration is untouched (criteria not moved).
    final = reg.get("HYP-RACE2")
    assert final["acceptance_criteria"] == CRITERIA
    assert final["preregistered_at"] == PREREG


# ------------------------------------------ F5: unknown label fails CLOSED

def test_unknown_label_row_fails_closed_at_record_result(reg):
    _raw_insert(reg, "HYP-DRIFT", label="speculative")  # manual edit / drift
    with pytest.raises(RegistryIntegrityError, match=r"\[label\] unknown label"):
        reg.record_result("HYP-DRIFT", run_at=RUN, decision="promote_paper")
    assert reg.get("HYP-DRIFT")["run_at"] is None


# ------------------------------------------------ F6: parent_id must exist

def test_register_with_nonexistent_parent_raises(reg):
    with pytest.raises(RegistryIntegrityError, match=r"\[parent\].*not a registered"):
        reg.register("HYP-ORPHAN", label="confirmatory", parent_id="HYP-GHOST",
                     cooling_off_until="2026-07-01T00:00:00+00:00")
    assert reg.get("HYP-ORPHAN") is None  # nothing inserted


def test_register_with_existing_parent_ok(reg):
    reg.register("HYP-PARENT", label="exploratory")
    row = reg.register("HYP-CHILD", label="exploratory", parent_id="HYP-PARENT",
                       cooling_off_until="2026-01-01T00:00:00+00:00")
    assert row["parent_id"] == "HYP-PARENT"


# ----------------------------------------- F7: JSON columns round-trip types

def test_json_string_value_round_trips_as_string(reg):
    reg.register("HYP-STR", label="exploratory", features="123",
                 params={"k": "v"})
    got = reg.get("HYP-STR")
    assert got["features"] == "123"          # string stays a string, not int 123
    assert isinstance(got["features"], str)
    assert got["params"] == {"k": "v"}


def test_none_json_columns_stay_none_and_null_in_storage(reg):
    reg.register("HYP-NONE", label="exploratory")
    got = reg.get("HYP-NONE")
    assert got["features"] is None
    assert got["params"] is None
    assert got["acceptance_criteria"] is None
    assert got["result_json"] is None
    # Stored as SQL NULL (not the string 'null') so IS NULL integrity checks work.
    c = sqlite3.connect(str(reg.db_path))
    try:
        n = c.execute(
            "SELECT COUNT(*) FROM experiments WHERE hypothesis_id='HYP-NONE' "
            "AND features IS NULL AND params IS NULL "
            "AND acceptance_criteria IS NULL AND result_json IS NULL"
        ).fetchone()[0]
    finally:
        c.close()
    assert n == 1
