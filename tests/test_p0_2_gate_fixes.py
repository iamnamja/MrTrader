"""Alpha-v9 P0-2 — the two diversifier-killing gate flaws, fixed.

(Ⓑ) Powered stability test replaces the binary "positive in both halves" guard.
(Ⓒ) Track-B standalone-SR floor is report-only for declared diversifiers/risk_premia.
(Ⓓ) Track-B P(ΔSR>0) is judged against a lowered probation bar for those types,
    yielding a PAPER-only, live-paper-ratified small-size admit.
"""
import numpy as np
import pandas as pd

from app.research.stability import stability_test, MIN_HALF_OBS
from scripts.walkforward.track_b_appraisal import appraise_track_b, TrackBAppraisalCriteria


_IDX = pd.bdate_range("2010-01-01", periods=2520)   # ~10y daily


def _series(mu1, mu2, sig=0.01, seed=0):
    rng = np.random.default_rng(seed)
    n = len(_IDX)
    h = n // 2
    r = np.concatenate([rng.normal(mu1, sig, h), rng.normal(mu2, sig, n - h)])
    return pd.Series(r, index=_IDX)


def _mu_for_sr(sr, sig=0.01):
    return sr * sig / np.sqrt(252)


# ── Ⓑ powered stability test ─────────────────────────────────────────────────
def test_powered_stability_far_fewer_false_negatives_than_binary_on_true_edge():
    """A TRUE stable SR-0.5 edge: the old binary both-halves guard false-negatives
    ~24%; the powered test should be far lower (it tests for a structural BREAK,
    not luck)."""
    m = _mu_for_sr(0.5)
    N = 200
    # apples-to-apples: the OLD `both_halves_positive` vs the NEW stability verdict
    # (`halves_indistinguishable` — the drop-in replacement; edge significance is
    # handled separately by Track-A/Track-B in the real gate).
    res = [stability_test(_series(m, m, seed=s), n_boot=400, seed=s) for s in range(N)]
    old_fn = sum(not r.both_halves_positive for r in res)
    new_fn = sum(not r.halves_indistinguishable for r in res)
    assert old_fn / N > 0.15        # the unpowered guard kills a real edge often
    assert new_fn / N < 0.10        # the powered test rarely does
    assert new_fn < old_fn


def test_powered_stability_catches_a_real_break_on_average():
    """A genuine +1.0/−1.0 break is flagged UNSTABLE (halves distinguishable) in the
    large majority of draws where the realized break is significant."""
    caught = sum(
        not stability_test(_series(_mu_for_sr(1.0), _mu_for_sr(-1.0), seed=s),
                           n_boot=800, seed=s).halves_indistinguishable
        for s in range(20))
    assert caught >= 14             # >=70% of draws flagged unstable


def test_passed_rejects_no_edge_noise_even_though_halves_indistinguishable():
    """C1 guard: a PURE-NOISE sleeve (both halves ~SR 0) trivially has
    indistinguishable halves — but `passed` must be False because the pooled edge is
    not significant. (Stability alone must never gate a promotion.)"""
    noise = _series(0.0, 0.0, seed=42)
    r = stability_test(noise, n_boot=800, seed=42)
    assert r.halves_indistinguishable is True     # no break (both halves ~0)
    assert r.pooled_significant is False          # ...but there's no edge
    assert r.passed is False                      # so the SAFE verdict rejects it


def test_passed_accepts_a_significant_stable_edge():
    """A strong, stable edge (SR ~0.9 both halves over 10y) is both indistinguishable
    AND pooled-significant -> `passed` True."""
    m = _mu_for_sr(0.9)
    r = stability_test(_series(m, m, seed=1), n_boot=1000, seed=1)
    assert r.halves_indistinguishable is True
    assert r.pooled_significant is True
    assert r.passed is True


def test_weakly_powered_flag_surfaces_wide_ci():
    """The half-Sharpe difference CI is inherently wide on short halves; the flag makes
    a wide-CI 'no break' explicit rather than silently equal to a tight-CI one."""
    short = _series(_mu_for_sr(0.5), _mu_for_sr(0.5), sig=0.01, seed=2)
    short = short.iloc[:600]        # ~2.4y total -> ~1.2y halves -> very wide diff CI
    r = stability_test(short, n_boot=800, seed=2)
    assert r.weakly_powered is True


def test_powered_stability_fails_closed_on_a_too_short_half():
    short = pd.Series(np.random.default_rng(0).normal(0.0003, 0.01, 2 * MIN_HALF_OBS - 4),
                      index=pd.bdate_range("2020-01-01", periods=2 * MIN_HALF_OBS - 4))
    r = stability_test(short)
    assert r.passed is False and "too short" in r.reason


def test_stability_difference_ci_brackets_the_point_estimate():
    r = stability_test(_series(_mu_for_sr(0.8), _mu_for_sr(0.2), seed=1), n_boot=1000, seed=1)
    assert r.diff_ci_low <= r.sr_diff <= r.diff_ci_high


# ── Ⓒ/Ⓓ diversifier-aware Track-B ────────────────────────────────────────────
def _book_and_weak_candidate(seed=5, cand_mu=0.00045):
    idx = pd.bdate_range("2008-01-01", periods=3000)
    rng = np.random.default_rng(seed)
    base = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    cand = pd.Series(rng.normal(cand_mu, 0.012, len(idx)), index=idx)   # low corr, weak
    return base, cand


def test_probation_lowers_the_pdsr_bar_only_for_declared_diversifiers():
    base, cand = _book_and_weak_candidate()
    crit = TrackBAppraisalCriteria.from_retrain_config()
    rp = appraise_track_b(base, cand, component_type="risk_premium", criteria=crit,
                          probation=True, worst_regime_sharpe=-0.2, n_boot=800, seed=1)
    rf = appraise_track_b(base, cand, component_type="risk_premium", criteria=crit,
                          probation=False, worst_regime_sharpe=-0.2, n_boot=800, seed=1)
    ra = appraise_track_b(base, cand, component_type="alpha", criteria=crit,
                          probation=True, worst_regime_sharpe=-0.2, n_boot=800, seed=1)
    assert rp.probation_applied is True and rp.effective_min_pdsr == crit.probation_min_pdsr
    assert rf.probation_applied is False and rf.effective_min_pdsr == crit.min_pdsr
    # probation has NO effect on an `alpha` component — it keeps the full bar
    assert ra.probation_applied is False and ra.effective_min_pdsr == crit.min_pdsr


def test_full_pass_implies_probation_pass_monotonicity():
    """The probation bar is weakly more lenient than the full bar (only P(ΔSR>0) and
    the standalone floor differ, both in the lenient direction), so a full PASS must
    imply a probation PASS for the same diversifier series."""
    crit = TrackBAppraisalCriteria.from_retrain_config()
    for seed in range(6):
        base, cand = _book_and_weak_candidate(seed=seed, cand_mu=0.0004 + 0.0001 * seed)
        rf = appraise_track_b(base, cand, component_type="risk_premium", criteria=crit,
                              probation=False, worst_regime_sharpe=-0.2, n_boot=600, seed=1)
        rp = appraise_track_b(base, cand, component_type="risk_premium", criteria=crit,
                              probation=True, worst_regime_sharpe=-0.2, n_boot=600, seed=1)
        if rf.passed:
            assert rp.passed, f"seed {seed}: full passed but probation did not"


def test_standalone_floor_is_report_only_for_diversifiers_but_gates_alpha():
    """A weak candidate with standalone SR < the floor: 'standalone_vt_sharpe' GATES an
    alpha (appears in failed_criteria) but is REPORT-ONLY for a diversifier."""
    # negative-drift candidate -> reliably-negative standalone SR (a genuine diversifier
    # can lose money standalone; that's exactly the case the floor wrongly killed).
    base, cand = _book_and_weak_candidate(cand_mu=-0.0003)
    crit = TrackBAppraisalCriteria.from_retrain_config()
    rdiv = appraise_track_b(base, cand, component_type="diversifier", criteria=crit,
                            worst_regime_sharpe=-0.2, n_boot=400, seed=1)
    ralpha = appraise_track_b(base, cand, component_type="alpha", criteria=crit,
                              worst_regime_sharpe=-0.2, n_boot=400, seed=1)
    assert rdiv.standalone_vt_sharpe < crit.min_standalone_sr     # genuinely below the floor
    assert "standalone_vt_sharpe" not in rdiv.failed_criteria     # waived (report-only)
    assert rdiv.standalone_floor_waived is True
    assert "standalone_vt_sharpe" in ralpha.failed_criteria       # still gates alpha
    assert ralpha.standalone_floor_waived is False


def test_probation_pass_flags_live_paper_ratification():
    """When a diversifier PASSES on the probation bar, the result must flag that it's a
    PAPER-only admit requiring live-paper ratification before any capital."""
    base, cand = _book_and_weak_candidate(seed=11, cand_mu=0.00045)
    crit = TrackBAppraisalCriteria.from_retrain_config()
    rp = appraise_track_b(base, cand, component_type="risk_premium", criteria=crit,
                          probation=True, worst_regime_sharpe=-0.2, n_boot=1500, seed=1)
    if rp.passed:                                  # this construction passes; guard anyway
        assert rp.requires_live_paper_ratification is True
        assert rp.verdict == "PASS (probation)"
    # a non-passing result never claims ratification
    if not rp.passed:
        assert rp.requires_live_paper_ratification is False
