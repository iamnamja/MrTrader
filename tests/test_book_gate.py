"""
Tests for the Track B (book-delta) acceptance gate (scripts/walkforward/book_gate.py).

Coverage (deterministic seeded synthetic daily-return series; NO network, NO DB):
  1. A genuine crisis-diversifier (low/negative corr, positive standalone SR,
     worst days disjoint from the book's) -> TRACK_B_PASS on every criterion.
  2. A candidate highly correlated to the book -> FAIL on corr_to_book.
  3. A candidate that improves book Sharpe but deepens book maxDD (its own
     crash inside the base book's drawdown window) -> FAIL on max_dd_delta
     while sharpe_delta still passes.
  4. TAIL-OVERLAP TEST (the registered criterion): an independent diversifier
     PASSES it; a co-crasher (worst days coincide with the book's) FAILS it;
     and the MAJOR-2 masking candidate (crashes on 13/14 of the book's tail
     days with one big positive day that flips the MEAN positive) FAILS it -
     the old mean-on-tail-days test would have ADMITTED that one.
  5. A negative-drift candidate -> FAIL on the standalone vol-targeted SR floor.
  6. BookGateCriteria is frozen/immutable.
  7. book_delta_gate is PURE: deterministic across calls, does not mutate its
     input series, and does not mutate app/ml/retrain_config constants.
  8. BookGateCriteria.from_retrain_config reads the live TRACKB_* constants
     (incl. TRACKB_MAX_TAIL_OVERLAP and SHARPE_IMPLAUSIBILITY_CEILING).
  9. Target-invariance after the leverage-cap removal: doubling the base's
     vol-target level does NOT change corr / standalone-SR / overlap verdicts
     when the 2% vol floor is not binding.
 10. Guards: non-DatetimeIndex -> TypeError; zero-variance candidate -> clean
     degenerate REJECT with no RuntimeWarnings; near-constant positive
     candidate -> FAIL on the Sharpe implausibility ceiling (+ vol-floor
     warning); n_tail floor of 3 on short histories; vol_floor_bind_frac
     diagnostic present.
  Plus: risk-budget validation (out-of-budget raises; smaller budget honored),
  insufficient-history ValueError, to_dict round-trip / report formatting.
"""
import dataclasses
import warnings as _warnings

import numpy as np
import pandas as pd
import pytest

from scripts.walkforward.book_gate import (
    BookGateCriteria,
    BookGateResult,
    book_delta_gate,
    format_report,
)

N_DAYS = 1500
CRASH = slice(700, 710)          # the base book's 10-day crisis (-2%/day)


def _criteria() -> BookGateCriteria:
    """Fixed criteria for pure tests (match the registered production values by
    design, but constructed locally so tests do not depend on live config).
    max_risk_budget = 0.25 per the owner-approved REGISTERED amendment of
    2026-06-11 (TRACKB_MAX_RISK_BUDGET 0.10 -> 0.25)."""
    return BookGateCriteria(
        min_sharpe_delta=0.10, min_calmar_delta=0.0, max_dd_delta=0.0,
        max_corr=0.30, min_standalone_sr=0.20, max_risk_budget=0.25,
        joint_tail_pctl=0.01, max_tail_overlap=0.30,
        sharpe_implausibility_ceiling=3.0,
    )


def _idx(n: int = N_DAYS) -> pd.DatetimeIndex:
    return pd.bdate_range(end="2026-06-01", periods=n)


def _base_book() -> pd.Series:
    """The existing combined book: SR ~0.7 with a 10-day -2%/day crisis."""
    rng = np.random.default_rng(11)
    vals = rng.normal(0.0005, 0.008, N_DAYS)
    vals[CRASH] = -0.02
    return pd.Series(vals, index=_idx(), name="base")


def _diversifier() -> pd.Series:
    """Genuine crisis-diversifier: positive drift, mild NEGATIVE corr to the
    book (hedge term), so it is up on the book's worst days."""
    base = _base_book().to_numpy()
    rng = np.random.default_rng(12)
    vals = 0.0008 - 0.20 * (base - 0.0005) + rng.normal(0.0, 0.0058, N_DAYS)
    return pd.Series(vals, index=_idx(), name="cand")


def _independent() -> pd.Series:
    """Genuinely independent candidate (no base term at all): under
    independence the tail-overlap fraction is ~n_tail/n_days (~1%) << 0.30."""
    rng = np.random.default_rng(21)
    return pd.Series(rng.normal(0.0008, 0.006, N_DAYS), index=_idx(), name="cand")


def _correlated() -> pd.Series:
    """Candidate that is essentially the book again (corr ~0.95)."""
    base = _base_book().to_numpy()
    rng = np.random.default_rng(13)
    vals = 0.0005 + 0.9 * (base - 0.0005) + rng.normal(0.0, 0.002, N_DAYS)
    return pd.Series(vals, index=_idx(), name="cand")


def _dd_worsener() -> pd.Series:
    """High-Sharpe candidate that cushions the book's crash days (tail-overlap
    OK) but has its OWN crash inside the base's drawdown window (days 715-740),
    deepening the combined book's max drawdown."""
    rng = np.random.default_rng(14)
    vals = rng.normal(0.0012, 0.006, N_DAYS)
    vals[CRASH] = 0.008          # cushions the book's tail days
    vals[715:741] = -0.02        # own crash during the base's recovery
    return pd.Series(vals, index=_idx(), name="cand")


def _tail_crasher() -> pd.Series:
    """Candidate that crashes WITH the book on the book's worst days: whenever
    the base has an extreme down day (its crash window AND its worst noise
    days), the candidate crashes too - the canonical tail-amplifier."""
    base = _base_book().to_numpy()
    rng = np.random.default_rng(15)
    vals = rng.normal(0.0008, 0.006, N_DAYS)
    vals[base < -0.018] = -0.03
    return pd.Series(vals, index=_idx(), name="cand")


def _low_sr() -> pd.Series:
    """Negative-drift candidate: fails the standalone vol-targeted SR floor."""
    rng = np.random.default_rng(16)
    return pd.Series(rng.normal(-0.0002, 0.006, N_DAYS), index=_idx(), name="cand")


# --- MAJOR-2 masking pair: base whose worst 14 days are EXACTLY its 15-day
# --- crash window (minus one), candidate that crashes on 13 of those days but
# --- posts one +8% day that flips the tail MEAN positive.
MASK_CRASH = slice(700, 715)     # 15-day crisis at -2.5%/day


def _masking_base() -> pd.Series:
    """Low-noise base (std 0.006) so the -2.5% crash days are unambiguously its
    worst days (no noise day reaches -2.5%)."""
    rng = np.random.default_rng(31)
    vals = rng.normal(0.0005, 0.006, N_DAYS)
    vals[MASK_CRASH] = -0.025
    return pd.Series(vals, index=_idx(), name="base")


def _masking_candidate() -> pd.Series:
    """The MAJOR-2 false-ADMIT case for the OLD mean test: crashes -0.5% on 13
    of the base's 14 worst days (its own worst days, vs 0.1% noise), plus ONE
    +8% day on the 14th -> tail MEAN positive (would have passed mean >= 0)
    while the candidate is plainly a tail-amplifier (overlap 13/14)."""
    rng = np.random.default_rng(32)
    vals = rng.normal(0.0001, 0.001, N_DAYS)
    vals[700:713] = -0.005       # crashes on 13 of the base's worst-14 days
    vals[713] = 0.08             # the one lucky day that masks the mean
    return pd.Series(vals, index=_idx(), name="cand")


# ------------------------------------------------------------------ case 1
def test_genuine_diversifier_passes():
    res = book_delta_gate(_base_book(), _diversifier(), criteria=_criteria(),
                          candidate_label="diversifier")
    assert isinstance(res, BookGateResult)
    assert res.failed_criteria == []
    assert res.passed is True
    assert res.verdict.startswith("TRACK_B_PASS")
    # Every individual criterion holds.
    assert all(ok for _v, ok in res.checks.values())
    # The economics are as constructed: book Sharpe up, corr low/negative,
    # tails disjoint, drawdown no deeper.
    assert res.sharpe_delta >= 0.10
    assert res.corr_to_book < 0.30
    assert res.tail_overlap_fraction <= 0.30
    assert res.with_book["max_drawdown"] >= res.without_book["max_drawdown"]
    assert res.standalone_vt_sharpe > 0.20
    # A ~9%-vol candidate never touches the 2% floor; no warnings.
    assert res.vol_floor_bind_frac == 0.0
    assert res.warnings == []
    # Track B scope statement is part of the verdict contract.
    assert "CAPITAL" in res.verdict


# ------------------------------------------------------------------ case 2
def test_high_correlation_fails_corr():
    res = book_delta_gate(_base_book(), _correlated(), criteria=_criteria())
    assert res.passed is False
    assert "corr_to_book" in res.failed_criteria
    assert res.corr_to_book >= 0.30
    assert res.verdict.startswith("TRACK_B_FAIL")


# ------------------------------------------------------------------ case 3
def test_sharpe_up_but_deeper_drawdown_fails_maxdd():
    res = book_delta_gate(_base_book(), _dd_worsener(), criteria=_criteria())
    assert res.passed is False
    assert "max_dd_delta" in res.failed_criteria
    # It DOES improve Sharpe and DOES keep its tail off the book's - maxDD is
    # the failure (its own crash sits in the base's RECOVERY, not its tail).
    assert res.checks["sharpe_delta"][1] is True
    assert res.checks["tail_overlap"][1] is True
    # Sign convention: maxDD <= 0, with-book deeper => more negative => delta < 0.
    assert res.with_book["max_drawdown"] < res.without_book["max_drawdown"]
    assert res.max_dd_delta < 0


# ------------------------------------------------------------------ case 4
def test_independent_diversifier_passes_tail_overlap():
    """An independent candidate must not be tail-rejected: under independence
    the expected overlap is ~n_tail/n_days (~1%) << 0.30 (this was the ~43%
    false-REJECT failure mode of the old mean test)."""
    res = book_delta_gate(_base_book(), _independent(), criteria=_criteria())
    assert res.checks["tail_overlap"][1] is True
    assert res.tail_overlap_fraction <= 0.30
    # Bookkeeping: registered tail size on the evaluated index, floored at 3.
    assert res.n_tail == max(3, int(np.floor(0.01 * res.n_days)))
    assert len(res.base_worst_dates) == res.n_tail


def test_crashes_with_book_fails_tail_overlap():
    res = book_delta_gate(_base_book(), _tail_crasher(), criteria=_criteria())
    assert res.passed is False
    assert "tail_overlap" in res.failed_criteria
    assert res.tail_overlap_fraction > 0.30
    assert res.tail_overlap == round(res.tail_overlap_fraction * res.n_tail)
    assert res.verdict.startswith("TRACK_B_FAIL")


def test_one_lucky_day_masking_fails_tail_overlap():
    """MAJOR-2: the candidate crashes on 13/14 of the book's worst days; one
    +8% day flips the tail MEAN positive, so the OLD (unregistered) mean test
    would have ADMITTED it. The registered overlap test FAILS it."""
    base, cand = _masking_base(), _masking_candidate()
    res = book_delta_gate(base, cand, criteria=_criteria())
    assert res.passed is False
    assert "tail_overlap" in res.failed_criteria
    assert res.tail_overlap_fraction > 0.30
    # Demonstrate the mask: the raw candidate MEAN on the book's worst days is
    # POSITIVE (the old criterion's exact blind spot).
    tail_dates = pd.to_datetime(res.base_worst_dates)
    assert float(cand.loc[tail_dates].mean()) > 0.0
    # ... while it crashed on at least 12 of those days.
    assert int((cand.loc[tail_dates] <= -0.005).sum()) >= 12


# ------------------------------------------------------------------ case 5
def test_low_standalone_sr_fails_floor():
    res = book_delta_gate(_base_book(), _low_sr(), criteria=_criteria())
    assert res.passed is False
    assert "standalone_vt_sharpe" in res.failed_criteria
    assert res.standalone_vt_sharpe <= 0.20


# ------------------------------------------------------------------ case 6
def test_criteria_frozen():
    crit = _criteria()
    with pytest.raises(dataclasses.FrozenInstanceError):
        crit.min_sharpe_delta = 0.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        crit.max_tail_overlap = 1.0  # type: ignore[misc]


# ------------------------------------------------------------------ case 7
def test_gate_is_pure_and_deterministic():
    import app.ml.retrain_config as rc
    snapshot = {k: getattr(rc, k) for k in dir(rc) if k.startswith("TRACKB_")}
    assert snapshot, "TRACKB_* constants must exist in retrain_config"
    assert "TRACKB_MAX_TAIL_OVERLAP" in snapshot

    base, cand = _base_book(), _diversifier()
    base_copy, cand_copy = base.copy(deep=True), cand.copy(deep=True)

    r1 = book_delta_gate(base, cand, criteria=_criteria())
    r2 = book_delta_gate(base, cand, criteria=_criteria())
    assert r1.to_dict() == r2.to_dict()          # deterministic

    pd.testing.assert_series_equal(base, base_copy)   # inputs not mutated
    pd.testing.assert_series_equal(cand, cand_copy)

    after = {k: getattr(rc, k) for k in dir(rc) if k.startswith("TRACKB_")}
    assert after == snapshot                     # config not mutated


# ------------------------------------------------------------------ case 8
def test_from_retrain_config_reads_live_constants():
    import app.ml.retrain_config as rc
    crit = BookGateCriteria.from_retrain_config()
    assert crit.min_sharpe_delta == rc.TRACKB_MIN_SHARPE_DELTA
    assert crit.min_calmar_delta == rc.TRACKB_MIN_CALMAR_DELTA
    assert crit.max_dd_delta == rc.TRACKB_MAX_DD_DELTA
    assert crit.max_corr == rc.TRACKB_MAX_CORR
    assert crit.min_standalone_sr == rc.TRACKB_MIN_STANDALONE_SR
    assert crit.max_risk_budget == rc.TRACKB_MAX_RISK_BUDGET
    assert crit.joint_tail_pctl == rc.TRACKB_JOINT_TAIL_PCTL
    assert crit.max_tail_overlap == rc.TRACKB_MAX_TAIL_OVERLAP
    assert crit.sharpe_implausibility_ceiling == rc.SHARPE_IMPLAUSIBILITY_CEILING
    # And the registered values themselves are the pre-registered ones.
    assert crit == _criteria()


# ------------------------------------------------------------------ case 9
def test_target_invariance_after_cap_removal():
    """MAJOR-1: doubling the base series doubles the (full-sample) vol-target
    level. With the leverage cap removed and the 2% floor not binding, the
    vol-targeted candidate is an exact scalar multiple of a PIT series, so
    corr / standalone SR / overlap verdicts must be invariant to the target."""
    base, cand = _base_book(), _diversifier()
    r1 = book_delta_gate(base, cand, criteria=_criteria())
    r2 = book_delta_gate(base * 2.0, cand, criteria=_criteria())
    assert r1.vol_floor_bind_frac == 0.0 and r2.vol_floor_bind_frac == 0.0
    assert r2.corr_to_book == pytest.approx(r1.corr_to_book, rel=1e-9)
    assert r2.standalone_vt_sharpe == pytest.approx(r1.standalone_vt_sharpe, rel=1e-9)
    assert r2.n_tail == r1.n_tail
    assert r2.tail_overlap == r1.tail_overlap
    assert r2.base_worst_dates == r1.base_worst_dates
    for k in ("corr_to_book", "standalone_vt_sharpe", "tail_overlap"):
        assert r2.checks[k][1] == r1.checks[k][1]


# ------------------------------------------------------------------ case 10
def test_non_datetime_index_raises_typeerror():
    rng = np.random.default_rng(41)
    int_indexed = pd.Series(rng.normal(0.0005, 0.006, N_DAYS),
                            index=np.arange(N_DAYS))
    with pytest.raises(TypeError, match="DatetimeIndex"):
        book_delta_gate(int_indexed, _diversifier(), criteria=_criteria())
    with pytest.raises(TypeError, match="DatetimeIndex"):
        book_delta_gate(_base_book(), int_indexed, criteria=_criteria())


def test_zero_variance_candidate_clean_reject():
    """NIT-11: a constant candidate must short-circuit to a clean degenerate
    REJECT - no RuntimeWarnings, no NaN prints, report still renders."""
    const = pd.Series(0.001, index=_idx(), name="cand")
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")          # any warning -> test failure
        res = book_delta_gate(_base_book(), const, criteria=_criteria())
        report = format_report(res)
    assert res.passed is False
    assert "degenerate candidate (zero variance)" in res.verdict
    assert res.failed_criteria == ["candidate_variance"]
    assert res.without_book == {} and res.with_book == {}
    assert "VERDICT: FAIL" in report
    report.encode("ascii")


def test_implausible_standalone_sharpe_fails_ceiling():
    """NIT-12: a near-constant positive candidate (huge but finite SR) must
    FAIL the SHARPE_IMPLAUSIBILITY_CEILING check (Track A's CRITICAL-1 mirror),
    and its ~0.16% ann vol pins the 2% floor -> diagnostic + warning."""
    rng = np.random.default_rng(42)
    near_const = pd.Series(0.0005 + rng.normal(0.0, 1e-4, N_DAYS),
                           index=_idx(), name="cand")
    res = book_delta_gate(_base_book(), near_const, criteria=_criteria())
    assert res.passed is False
    assert "standalone_sr_plausible" in res.failed_criteria
    assert res.standalone_vt_sharpe > 3.0
    assert "implausible standalone Sharpe" in res.verdict
    assert res.vol_floor_bind_frac > 0.10
    assert any("vol-target unreliable" in w for w in res.warnings)


def test_n_tail_floor_of_three_on_short_history():
    """MINOR-6: on a short history floor(1% * n_days) would be 1; the tail is
    floored at 3 days so the overlap test stays defined."""
    n = 160                                       # 130 post-warmup, 129 evaluated
    rng = np.random.default_rng(43)
    base = pd.Series(rng.normal(0.0005, 0.008, n), index=_idx(n), name="base")
    cand = pd.Series(rng.normal(0.0008, 0.006, n), index=_idx(n), name="cand")
    res = book_delta_gate(base, cand, criteria=_criteria())
    assert res.n_tail == 3
    assert len(res.base_worst_dates) == 3
    assert int(np.floor(0.01 * res.n_days)) < 3   # the floor actually engaged


def test_n_days_matches_book_metrics_index():
    """MINOR-7: n_days and the tail are computed on the post-combine index
    (combine() drops day 0), so the reported counts and the book metrics refer
    to one and the same window."""
    res = book_delta_gate(_base_book(), _diversifier(), criteria=_criteria())
    assert res.n_days == res.without_book["n_days"] == res.with_book["n_days"]


# ------------------------------------------------------------------ extras
def test_risk_budget_validation():
    base, cand = _base_book(), _diversifier()
    # 0.30 is above the amended 0.25 cap -> out of the pre-registered budget.
    with pytest.raises(ValueError):
        book_delta_gate(base, cand, criteria=_criteria(), candidate_risk_budget=0.3)
    with pytest.raises(ValueError):
        book_delta_gate(base, cand, criteria=_criteria(), candidate_risk_budget=0.0)
    res = book_delta_gate(base, cand, criteria=_criteria(), candidate_risk_budget=0.05)
    assert res.risk_budget == 0.05
    assert res.checks["risk_budget"][1] is True


def test_insufficient_history_raises():
    short = _base_book().iloc[:100]
    with pytest.raises(ValueError):
        book_delta_gate(short, _diversifier().iloc[:100], criteria=_criteria())


def test_to_dict_and_report_ascii():
    res = book_delta_gate(_base_book(), _diversifier(), criteria=_criteria(),
                          candidate_label="diversifier")
    d = res.to_dict()
    assert d["passed"] is True
    assert d["criteria"]["max_risk_budget"] == 0.25
    assert d["criteria"]["max_tail_overlap"] == 0.30
    assert set(d["checks"]) == {
        "sharpe_delta", "calmar_delta", "max_dd_delta", "corr_to_book",
        "standalone_vt_sharpe", "standalone_sr_plausible", "risk_budget",
        "tail_overlap",
    }
    assert d["vol_floor_bind_frac"] == 0.0
    report = format_report(res)
    assert "TRACK B (BOOK-DELTA) ACCEPTANCE GATE" in report
    assert "tail overlap" in report
    assert "VERDICT: PASS" in report
    assert "NEVER" in report                      # capital scope note
    report.encode("ascii")                        # ASCII-only contract
