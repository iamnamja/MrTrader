"""
Tests for scripts/preregister_options_xs_features.py — the H4a-H4e frozen
cross-sectional options-signal pre-registrations (Phase-4 scientific contract).

Coverage (isolated temp registry via MRTRADER_RESEARCH_REGISTRY_DB; the real
data/research_registry.db is never touched):
  - all five hypotheses register + preregister cleanly with the frozen criteria
    and PREREGISTERED_AT; directions/features match the spec;
  - the script is idempotent (second run skips R1/R5, criteria unchanged);
  - R2 ordering holds: record_result requires run_at strictly AFTER
    2026-06-12T12:00:00+00:00.
"""
from __future__ import annotations

import pytest

from app.research.registry import RegistryIntegrityError, ResearchRegistry
from scripts.preregister_options_xs_features import (
    HYPOTHESES,
    PREREGISTERED_AT,
    _acceptance_criteria,
    preregister_all,
)

EXPECTED_IDS = {
    "H4a-OPTIONS-CPIV-20260612",
    "H4b-OPTIONS-SKEW-20260612",
    "H4c-OPTIONS-OSRATIO-20260612",
    "H4d-OPTIONS-TERMSLOPE-20260612",
    "H4e-OPTIONS-IVRV-20260612",
}

# (hypothesis_id -> (feature, sign-keyword expected in the frozen direction))
EXPECTED = {
    "H4a-OPTIONS-CPIV-20260612": ("cpiv_matched_delta", "POSITIVE"),
    "H4b-OPTIONS-SKEW-20260612": ("skew_25d_put", "NEGATIVE"),
    "H4c-OPTIONS-OSRATIO-20260612": ("opt_share_volume_ratio", "NEGATIVE"),
    "H4d-OPTIONS-TERMSLOPE-20260612": ("term_slope_30_60", "POSITIVE"),
    "H4e-OPTIONS-IVRV-20260612": ("iv_rv_20d_ratio", "NEGATIVE"),
}


@pytest.fixture()
def reg(tmp_path, monkeypatch):
    monkeypatch.setenv("MRTRADER_RESEARCH_REGISTRY_DB", str(tmp_path / "reg.db"))
    return ResearchRegistry()


def test_all_five_preregister_cleanly(reg):
    verified = preregister_all(reg)
    assert set(verified) == EXPECTED_IDS
    for h in HYPOTHESES:
        row = reg.get(h["hypothesis_id"])
        assert row is not None
        assert row["label"] == "confirmatory"
        assert row["family"] == "options_signal"
        assert row["preregistered_at"] == PREREGISTERED_AT
        assert row["acceptance_criteria"] == _acceptance_criteria(h)
        assert row["params"] == h["params"]
        assert row["run_at"] is None and row["decision"] is None


def test_features_and_directions_match_spec(reg):
    preregister_all(reg)
    for hid, (feature, sign_kw) in EXPECTED.items():
        row = reg.get(hid)
        assert row["params"]["feature"] == feature
        crit = row["acceptance_criteria"]
        assert crit["feature"] == feature
        assert sign_kw in crit["direction"]
        # Shared template fields are present + identical across all five.
        assert "WEEKS as clusters" in crit["instrument"]
        assert crit["decision_rule"]["pass"].startswith("week-clustered t>=2")
        assert "NOT a revival of the dead XS-ML" in crit["decision_rule"]["kill"]


def test_shared_template_is_identical_across_hypotheses(reg):
    preregister_all(reg)
    instruments = {reg.get(h["hypothesis_id"])["acceptance_criteria"]["instrument"]
                   for h in HYPOTHESES}
    decision_rules = {tuple(sorted(
        reg.get(h["hypothesis_id"])["acceptance_criteria"]["decision_rule"].items()))
        for h in HYPOTHESES}
    assert len(instruments) == 1
    assert len(decision_rules) == 1


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
    "2026-06-12T11:59:00+00:00",   # before the freeze
    PREREGISTERED_AT,              # same instant — R2 requires STRICTLY after
])
def test_record_result_requires_run_after_freeze(reg, bad_run_at):
    preregister_all(reg)
    for hid in EXPECTED_IDS:
        with pytest.raises(RegistryIntegrityError, match="R2"):
            reg.record_result(hid, run_at=bad_run_at, result={"t": 2.5})
        assert reg.get(hid)["run_at"] is None  # nothing committed


def test_record_result_after_freeze_is_one_shot(reg):
    preregister_all(reg)
    hid = "H4a-OPTIONS-CPIV-20260612"
    row = reg.record_result(hid, run_at="2026-06-20T03:00:00+00:00",
                            result={"t": 2.4}, decision="park")
    assert row["result_json"] == {"t": 2.4}
    with pytest.raises(RegistryIntegrityError, match="R4 one shot"):
        reg.record_result(hid, run_at="2026-06-21T03:00:00+00:00",
                          result={"t": 0.1})
