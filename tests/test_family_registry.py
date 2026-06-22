"""Alpha-v10 P0.5 — strategy-family registry + family-level trial counting.

Pins the auditable N_TRIALS (replacing the GL-0 hardcoded ~20), registry integrity, and the
wiring into null_zoo's deflated-Sharpe cross-check (more trials → harsher deflation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.research import family_registry as fr
from app.research.null_zoo import NullZooResult, deflated_sharpe


# ── registry integrity ────────────────────────────────────────────────────────
def test_no_duplicate_ids():
    ids = [f.id for f in fr.FAMILIES]
    assert len(ids) == len(set(ids))


def test_all_statuses_valid():
    for f in fr.FAMILIES:
        assert f.status in fr._VALID_STATUS, f"{f.id} has bad status {f.status}"


def test_every_family_has_a_doc_ref_and_verdict():
    for f in fr.FAMILIES:
        assert f.doc_ref and f.verdict, f"{f.id} missing doc_ref/verdict"


def test_count_by_status_sums_to_total():
    assert sum(fr.count_by_status().values()) == len(fr.FAMILIES)


# ── the trial count ───────────────────────────────────────────────────────────
def test_trial_count_matches_counted_families():
    assert fr.family_trial_count() == sum(1 for f in fr.FAMILIES if f.counts_as_trial)


def test_trial_count_exceeds_the_old_hardcoded_estimate():
    # the whole point of P0.5: the enumerated burden is REAL and was under-counted at ~20
    n = fr.family_trial_count()
    assert 20 <= n <= 40              # sane range; enumerated count, not a guess
    assert n >= 20                    # at least the prior estimate


def test_degrees_of_freedom_log_present():
    assert len(fr.DEGREES_OF_FREEDOM) >= 5


def test_excluded_entries_are_auditable_and_not_counted():
    # cash (infra) + futures_book (ensemble) are IN the registry for auditability but
    # excluded from the trial count — the exclusion is explicit, not by silent omission.
    excluded = {f.id for f in fr.FAMILIES if not f.counts_as_trial}
    assert excluded == {"cash_sleeve", "futures_book"}
    assert fr.family_trial_count() == len(fr.FAMILIES) - len(excluded)


def test_known_live_and_killed_families_present():
    by_id = {f.id: f for f in fr.FAMILIES}
    assert by_id["etf_trend"].status == fr.LIVE
    assert by_id["futures_carry"].status == fr.PAPER
    assert by_id["pead"].status == fr.KILLED
    assert by_id["curve_momentum"].status == fr.KILLED


# ── null_zoo wiring ───────────────────────────────────────────────────────────
def test_nullzoo_result_carries_family_fields():
    fields = NullZooResult.__dataclass_fields__
    assert "n_families" in fields and "dsr_family" in fields


def test_deflated_sharpe_is_monotone_decreasing_in_trials():
    # more trials searched → HARSHER deflation → lower DSR. This is why under-counting N inflates
    # the verdict, and why the enumerated (larger) family count is the conservative, honest choice.
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0004, 0.01, 2500))
    d20 = deflated_sharpe(r, 20, 1e-4)
    d_fam = deflated_sharpe(r, fr.family_trial_count(), 1e-4)
    assert d_fam <= d20              # N=25 ≥ 20 → DSR no higher than the old N=20 cross-check
