"""
Tests for scripts/walkforward/event_inference.py — the event-LEVEL inference
instrument (CGM two-way clustered OLS) that H1/H2/H3 decide on.

THE REFERENCE PIN — Petersen (2009) test dataset
------------------------------------------------
tests/fixtures/petersen_test_data.txt is the canonical published test panel
(Mitchell Petersen, "Estimating Standard Errors in Finance Panel Data Sets",
RFS 2009; 500 firms x 10 years = 5000 obs, columns firmid/year/x/y, vendored
from kellogg.northwestern.edu/faculty/petersen/htm/papers/se/test_data.txt).
The published reference numbers for the regression y = a + b*x:

    beta                 = 1.034833
    SE (OLS, iid)        = 0.028583
    SE (cluster by firm) = 0.050596
    SE (cluster by year) = 0.033389
    SE (two-way CGM)     = 0.053561  (0.0535580 under the convention below)

SMALL-SAMPLE CONVENTION pinned here (must match the module): the Stata-style
factor  c_d = G_d/(G_d-1) * (N-1)/(N-K)  applied to EACH component (firm,
year, AND the firm x year intersection) separately, V = c_a*V_a + c_b*V_b -
c_ab*V_ab. This is the convention of Stata's cluster2/cgmreg used by Petersen
and reproduces all four SEs above (verified to <= 5e-4 here, beta to 1e-5).

Plus the identity/behavioral suite:
  - one-way CGM (clusters=(g,g)) == bread-wrapped cluster_robust_cov on g;
  - intercept-only on iid data ~= pead_significance.newey_west_tstat's t_ols;
  - inflating within-date correlation WIDENS the two-way SE vs White;
  - the non-PSD eigenvalue-clip guard (psd_fixed flag);
  - decile_report monotonicity detection;
  - loco_robustness row counts (quarter / sector / top10).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.walkforward.event_inference import (
    InferenceResult,
    _psd_clip,
    _stata_factor,
    cluster_robust_cov,
    decile_report,
    loco_robustness,
    twoway_cluster_ols,
)

FIXTURE = Path(__file__).parent / "fixtures" / "petersen_test_data.txt"

# Published Petersen (2009) reference values for y = a + b*x.
PIN_BETA = 1.034833
PIN_SE_OLS = 0.028583
PIN_SE_FIRM = 0.050596
PIN_SE_TIME = 0.033389
PIN_SE_TWOWAY = 0.053561
BETA_TOL = 1e-5
SE_TOL = 5e-4


@pytest.fixture(scope="module")
def petersen() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE, sep=r"\s+", header=None,
                     names=["firmid", "year", "x", "y"])
    assert len(df) == 5000, "vendored Petersen dataset must be 500 firms x 10 years"
    return df


# ─────────────────────────────────────────────── Petersen reference pins

def test_petersen_beta_and_ols_se(petersen):
    res = twoway_cluster_ols(petersen, y="y", X=["x"], clusters=("firmid", "year"))
    assert res.names == ("const", "x")
    assert res.coef[1] == pytest.approx(PIN_BETA, abs=BETA_TOL)
    assert res.se_ols[1] == pytest.approx(PIN_SE_OLS, abs=SE_TOL)


def test_petersen_one_way_firm_se(petersen):
    # One-way clustering expressed through the CGM identity: with both
    # dimensions equal, V = V_g + V_g - V_(g^g) = V_g (intersection == g).
    res = twoway_cluster_ols(petersen, y="y", X=["x"], clusters=("firmid", "firmid"))
    assert res.se[1] == pytest.approx(PIN_SE_FIRM, abs=SE_TOL)
    assert res.n_clusters["firmid"] == 500


def test_petersen_one_way_time_se(petersen):
    res = twoway_cluster_ols(petersen, y="y", X=["x"], clusters=("year", "year"))
    assert res.se[1] == pytest.approx(PIN_SE_TIME, abs=SE_TOL)
    assert res.n_clusters["year"] == 10


def test_petersen_twoway_se(petersen):
    res = twoway_cluster_ols(petersen, y="y", X=["x"], clusters=("firmid", "year"))
    assert res.se[1] == pytest.approx(PIN_SE_TWOWAY, abs=SE_TOL)
    assert res.n_clusters == {"firmid": 500, "year": 10, "intersection": 5000}
    # df convention: min(G_a, G_b) - 1 = 9 (the conservative CGM/Stata choice).
    assert res.df == 9
    assert res.psd_fixed is False
    assert res.n_obs == 5000


def test_petersen_clustered_se_wider_than_ols(petersen):
    # The dataset has a firm random effect by construction — firm clustering
    # must widen the SE vs iid OLS (the whole point of the estimator).
    res = twoway_cluster_ols(petersen, y="y", X=["x"], clusters=("firmid", "year"))
    assert res.se[1] > res.se_ols[1] * 1.5


# ───────────────────────────────────── identity: one-way CGM == hand wrap

def test_one_way_identity_matches_cluster_robust_cov(petersen):
    sub = petersen
    n = len(sub)
    design = np.column_stack([np.ones(n), sub["x"].to_numpy()])
    bread = np.linalg.inv(design.T @ design)
    coef = bread @ design.T @ sub["y"].to_numpy()
    resid = sub["y"].to_numpy() - design @ coef
    meat = cluster_robust_cov(design, resid, sub["firmid"])
    g = sub["firmid"].nunique()
    v_hand = _stata_factor(g, n, 2) * (bread @ meat @ bread)
    se_hand = np.sqrt(np.diag(v_hand))

    res = twoway_cluster_ols(sub, y="y", X=["x"], clusters=("firmid", "firmid"))
    np.testing.assert_allclose(res.se, se_hand, rtol=1e-10)


def test_cluster_robust_cov_rejects_missing_groups():
    X = np.ones((4, 1))
    resid = np.array([0.1, -0.2, 0.3, 0.0])
    groups = pd.Series(["a", None, "b", "b"])
    with pytest.raises(ValueError, match="missing"):
        cluster_robust_cov(X, resid, groups)


# ─────────────────────── intercept-only iid ~= newey_west_tstat's t_ols

def test_intercept_only_iid_matches_ols_t():
    from scripts.pead_significance import newey_west_tstat

    rng = np.random.default_rng(1303)
    n = 5000
    y = rng.normal(0.001, 0.02, n)
    # Singleton clusters in both dimensions -> CGM collapses to (ss-scaled)
    # White, which on homoskedastic iid data matches the plain OLS t closely.
    panel = pd.DataFrame({
        "ret": y,
        "announce_date": np.arange(n),   # every obs its own day
        "symbol": np.arange(n),          # ... and its own firm
    })
    res = twoway_cluster_ols(panel, y="ret")
    nw = newey_west_tstat(list(y), lag=0)
    assert res.coef[0] == pytest.approx(float(np.mean(y)), rel=1e-9)
    assert res.tstat[0] == pytest.approx(nw["t_ols"], rel=0.02)


# ──────────────────── within-date correlation must WIDEN the two-way SE

def test_within_date_correlation_widens_se_vs_white():
    rng = np.random.default_rng(7)
    n_dates, n_firms = 60, 50
    date_shock = rng.normal(0, 0.03, n_dates)
    rows = []
    for d in range(n_dates):
        for f in range(n_firms):
            rows.append({
                "announce_date": d,
                "symbol": f,
                "ret": date_shock[d] + rng.normal(0, 0.005),
                "rowid": d * n_firms + f,
            })
    panel = pd.DataFrame(rows)
    res_two = twoway_cluster_ols(panel, y="ret", clusters=("announce_date", "symbol"))
    # White(ish) baseline via singleton clusters (the CGM degenerate case).
    res_white = twoway_cluster_ols(panel, y="ret", clusters=("rowid", "rowid"))
    assert res_two.se[0] > 3.0 * res_white.se[0], (
        "two-way SE must blow out vs White when returns are correlated within "
        "announcement days — the exact failure mode of per-event iid t-stats"
    )


# ─────────────────────────────────────────────────────────── PSD guard

def test_psd_clip_fixes_and_flags_non_psd():
    bad = np.array([[1.0, 0.0], [0.0, -0.5]])
    fixed, flag = _psd_clip(bad)
    assert flag is True
    eig = np.linalg.eigvalsh(fixed)
    assert (eig >= -1e-12).all()
    # The positive subspace is untouched.
    assert fixed[0, 0] == pytest.approx(1.0)
    assert fixed[1, 1] == pytest.approx(0.0, abs=1e-12)


def test_psd_clip_leaves_psd_untouched():
    good = np.array([[2.0, 0.3], [0.3, 1.0]])
    fixed, flag = _psd_clip(good)
    assert flag is False
    np.testing.assert_allclose(fixed, good, atol=1e-12)


# ──────────────────────────────────────────────────────── decile_report

def _decile_panel(monotone: bool, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    feat = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.001, n)
    y = (0.01 * feat + noise) if monotone else rng.permutation(0.01 * feat) + noise
    return pd.DataFrame({
        "score": feat,
        "ret": y,
        "announce_date": rng.integers(0, 40, n),
        "symbol": rng.integers(0, 50, n),
    })


def test_decile_report_detects_monotone_signal():
    rep = decile_report(_decile_panel(monotone=True), "score", "ret", n=10)
    assert rep["is_monotone"] is True
    assert rep["direction"] == "increasing"
    assert rep["spearman_rho"] == pytest.approx(1.0)
    assert rep["top_minus_bottom"] > 0
    assert len(rep["rows"]) == 10
    assert sum(r["n_obs"] for r in rep["rows"]) == 400


def test_decile_report_rejects_shuffled_signal():
    rep = decile_report(_decile_panel(monotone=False), "score", "ret", n=10)
    assert rep["is_monotone"] is False
    assert abs(rep["spearman_rho"]) < 0.9


# ───────────────────────────────────────────────────── loco_robustness

def _loco_panel() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 240
    quarters = pd.to_datetime(
        rng.choice(["2024-02-01", "2024-05-01", "2024-08-01", "2024-11-01"], n)
    )
    return pd.DataFrame({
        "announce_date": quarters,
        "symbol": rng.integers(0, 30, n),
        "sector": rng.choice(["Technology", "Energy", "Healthcare"], n),
        "ret": rng.normal(0.004, 0.02, n),
    })


def _infer(sub: pd.DataFrame) -> InferenceResult:
    return twoway_cluster_ols(sub, y="ret", clusters=("announce_date", "symbol"))


def test_loco_quarter_row_counts():
    panel = _loco_panel()
    rows = loco_robustness(panel, "ret", "quarter", _infer)
    assert len(rows) == 4  # 4 distinct calendar quarters
    from scripts.pead_significance import cluster_key
    keys = panel["announce_date"].map(lambda d: cluster_key(d.date()))
    for r in rows:
        assert r["n_obs"] == int((keys != r["left_out"]).sum())
        assert r["p_one_sided"] is not None


def test_loco_sector_row_counts():
    panel = _loco_panel()
    rows = loco_robustness(panel, "ret", "sector", _infer)
    assert len(rows) == 3
    for r in rows:
        assert r["n_obs"] == int((panel["sector"] != r["left_out"]).sum())


def test_loco_top10_drops_exactly_ten_largest():
    panel = _loco_panel()
    rows = loco_robustness(panel, "ret", "top10", _infer)
    assert len(rows) == 1
    assert rows[0]["left_out"] == "top10_abs_y_dropped"
    assert rows[0]["n_obs"] == len(panel) - 10
    # The 10 dropped really are the largest |ret|: re-run by hand.
    kept = panel.drop(index=panel["ret"].abs().nlargest(10).index)
    assert rows[0]["coef"] == pytest.approx(float(kept["ret"].mean()), rel=1e-9)


def test_loco_unknown_unit_raises():
    with pytest.raises(ValueError, match="unknown LOCO unit"):
        loco_robustness(_loco_panel(), "ret", "month", _infer)
