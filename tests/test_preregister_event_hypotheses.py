"""
Tests for scripts/preregister_event_hypotheses.py — the H1/H2/H3 frozen
pre-registrations (Phase-3 scientific contract).

Coverage (isolated temp registry via MRTRADER_RESEARCH_REGISTRY_DB; the real
data/research_registry.db is never touched):
  - all three hypotheses register + preregister cleanly with the frozen
    criteria and PREREGISTERED_AT;
  - the script is idempotent (second run skips R1/R5, criteria unchanged);
  - R2 ordering holds: record_result requires run_at strictly AFTER
    2026-06-11T12:00:00+00:00, and accepts a later run_at.
"""
from __future__ import annotations

import pytest

from app.research.registry import RegistryIntegrityError, ResearchRegistry
from scripts.preregister_event_hypotheses import (
    HYPOTHESES,
    PREREGISTERED_AT,
    preregister_all,
)

EXPECTED_IDS = {
    "H1-PEAD-EVENTLEVEL-20260611",
    "H2-IMPLIEDMOVE-CONTINUOUS-20260611",
    "H3-PEADV2-SCORECARD-20260611",
}


@pytest.fixture()
def reg(tmp_path, monkeypatch):
    monkeypatch.setenv("MRTRADER_RESEARCH_REGISTRY_DB", str(tmp_path / "reg.db"))
    return ResearchRegistry()


def test_all_three_preregister_cleanly(reg):
    verified = preregister_all(reg)
    assert set(verified) == EXPECTED_IDS
    for h in HYPOTHESES:
        row = reg.get(h["hypothesis_id"])
        assert row is not None
        assert row["label"] == "confirmatory"
        assert row["preregistered_at"] == PREREGISTERED_AT
        assert row["acceptance_criteria"] == h["acceptance_criteria"]
        assert row["params"] == h["params"]
        assert row["run_at"] is None and row["decision"] is None


def test_idempotent_rerun_skips_and_preserves_criteria(reg):
    preregister_all(reg)
    snapshot = {hid: reg.get(hid) for hid in EXPECTED_IDS}
    verified = preregister_all(reg)  # must not raise (R1/R5 swallowed)
    assert set(verified) == EXPECTED_IDS
    for hid in EXPECTED_IDS:
        again = reg.get(hid)
        assert again["acceptance_criteria"] == snapshot[hid]["acceptance_criteria"]
        assert again["preregistered_at"] == snapshot[hid]["preregistered_at"]


@pytest.mark.parametrize("bad_run_at", [
    "2026-06-11T11:59:00+00:00",   # before the freeze
    PREREGISTERED_AT,              # same instant — R2 requires STRICTLY after
])
def test_record_result_requires_run_after_freeze(reg, bad_run_at):
    preregister_all(reg)
    for hid in EXPECTED_IDS:
        with pytest.raises(RegistryIntegrityError, match="R2"):
            reg.record_result(hid, run_at=bad_run_at, result={"p": 0.04})
        assert reg.get(hid)["run_at"] is None  # nothing committed


def test_record_result_after_freeze_is_one_shot(reg):
    preregister_all(reg)
    hid = "H1-PEAD-EVENTLEVEL-20260611"
    row = reg.record_result(hid, run_at="2026-06-20T03:00:00+00:00",
                            result={"p_10d": 0.03})
    assert row["result_json"] == {"p_10d": 0.03}
    with pytest.raises(RegistryIntegrityError, match="R4 one shot"):
        reg.record_result(hid, run_at="2026-06-21T03:00:00+00:00",
                          result={"p_10d": 0.5})
