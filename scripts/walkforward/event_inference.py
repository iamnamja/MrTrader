"""
event_inference.py — event-LEVEL inference for the earnings-event panel (Alpha-v6 Phase 3).

THE significance instrument for event/rules strategies (blueprint
NEXT_PHASE_BLUEPRINT_2026-06.md Phase 3, consensus C3): OLS on the panel of
hedged forward event returns with **two-way cluster-robust standard errors**,
clusters = (announce_date, symbol), per Cameron-Gelbach-Miller (2011).
The independence unit is the announcement-day cluster, not the CPCV fold —
the P0 calibration showed the 8-fold path t-stat cannot separate PEAD from
noise (3/5 true nulls cleared t>=2.0), so this module replaces it for event
strategies. CPCV remains the robustness/coverage leg.

Pure math: numpy/pandas/scipy only. statsmodels is NOT installed in this
environment and must not be added — the CGM estimator is hand-rolled and
unit-tested against the published Petersen (2009) reference dataset
(tests/test_event_inference.py pins beta and all four SEs).

SMALL-SAMPLE CONVENTION (the one that reproduces Petersen's published numbers):
    Stata-style factor applied to EACH covariance component separately,
        c_d = (G_d / (G_d - 1)) * ((N - 1) / (N - K)),
    where G_d is that component's own cluster count — including the
    intersection component (G_ab). The two-way covariance is
        V = c_a*V_a + c_b*V_b - c_ab*V_ab          (CGM identity)
    with each V_d = bread @ M_d @ bread and M_d the Liang-Zeger meat.
    t is referred to a Student-t with df = min(G_a, G_b) - 1 (the standard
    conservative CGM/Stata convention for two-way clustering).

NON-PSD GUARD: V_a + V_b - V_ab is not guaranteed positive semi-definite.
Following CGM (2011) section 2.3 we eigendecompose, clip negative eigenvalues
to zero and reconstruct; `InferenceResult.psd_fixed` flags when this fired so
a confirmatory run can report it (it should be rare on real panels).

Public API:
    twoway_cluster_ols(panel, y, X=None, clusters=("announce_date","symbol"))
        -> InferenceResult                      # X=None -> intercept-only mean test (H1)
    cluster_robust_cov(X, resid, groups)        # one-way Liang-Zeger meat (K x K)
    decile_report(panel, feature, y, n=10)      # per-decile means + monotonicity
    loco_robustness(panel, y, unit, infer_fn)   # unit in {"quarter","sector","top10"}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Eigenvalues below -PSD_TOL (relative to the largest eigenvalue) count as a
# genuine non-PSD defect (clipped + flagged); tiny negative values inside the
# tolerance are numerical noise (clipped silently, NOT flagged).
PSD_TOL = 1e-10


@dataclass(frozen=True)
class InferenceResult:
    """Result of a (two-way) cluster-robust OLS. Arrays are aligned to `names`
    (index 0 is always the intercept "const")."""

    names: Tuple[str, ...]
    coef: np.ndarray
    se: np.ndarray                 # two-way cluster-robust SE
    se_ols: np.ndarray             # naive iid OLS SE (the optimism comparison)
    tstat: np.ndarray              # coef / se (cluster-robust)
    p_one_sided: np.ndarray        # H0: coef <= 0 vs H1: coef > 0 (Student-t, df)
    p_two_sided: np.ndarray
    n_obs: int
    n_clusters: Dict[str, int]     # {dim_a: G_a, dim_b: G_b, "intersection": G_ab}
    df: int                        # min(G_a, G_b) - 1
    cov: np.ndarray = field(repr=False, default=None)  # K x K two-way covariance
    psd_fixed: bool = False        # True iff the non-PSD eigenvalue clip fired

    def summary_dict(self) -> dict:
        """Compact JSON-able summary (artifact/registry friendly)."""
        return {
            "names": list(self.names),
            "coef": [float(v) for v in self.coef],
            "se": [float(v) for v in self.se],
            "se_ols": [float(v) for v in self.se_ols],
            "tstat": [float(v) for v in self.tstat],
            "p_one_sided": [float(v) for v in self.p_one_sided],
            "p_two_sided": [float(v) for v in self.p_two_sided],
            "n_obs": int(self.n_obs),
            "n_clusters": {k: int(v) for k, v in self.n_clusters.items()},
            "df": int(self.df),
            "psd_fixed": bool(self.psd_fixed),
        }


def cluster_robust_cov(X: np.ndarray, resid: np.ndarray, groups) -> np.ndarray:
    """One-way Liang-Zeger cluster 'meat': M = sum_g (X_g' u_g)(X_g' u_g)'.

    Returns the raw K x K meat matrix — NO bread wrapping and NO small-sample
    factor (the caller applies both; see the module-docstring convention).
    `groups` is any 1-D label array/Series aligned to the rows of X.
    """
    X = np.asarray(X, dtype=float)
    resid = np.asarray(resid, dtype=float)
    if X.ndim != 2 or len(X) != len(resid):
        raise ValueError("X must be (N,K) with resid aligned (N,)")
    codes, uniques = pd.factorize(np.asarray(groups))
    if (codes < 0).any():
        raise ValueError("groups contains missing values — drop them before inference")
    scores = X * resid[:, None]                       # N x K score contributions
    sums = np.zeros((len(uniques), X.shape[1]))
    np.add.at(sums, codes, scores)                    # per-cluster score sums
    return sums.T @ sums


def _psd_clip(cov: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Eigenvalue-clip a symmetric matrix to PSD (CGM 2011 sec 2.3).

    Returns (psd_cov, fixed) — `fixed` is True only when a MATERIALLY negative
    eigenvalue (beyond numerical noise, see PSD_TOL) was clipped.
    """
    sym = 0.5 * (cov + cov.T)
    eigval, eigvec = np.linalg.eigh(sym)
    scale = max(float(np.max(np.abs(eigval))), 1.0)
    fixed = bool(np.any(eigval < -PSD_TOL * scale))
    if np.any(eigval < 0.0):
        eigval = np.clip(eigval, 0.0, None)
        sym = eigvec @ np.diag(eigval) @ eigvec.T
    return sym, fixed


def _stata_factor(g: int, n: int, k: int) -> float:
    """Stata-style small-sample factor c = G/(G-1) * (N-1)/(N-K)."""
    return (g / (g - 1.0)) * ((n - 1.0) / (n - k))


def twoway_cluster_ols(
    panel: pd.DataFrame,
    y: str,
    X: Optional[Sequence[str]] = None,
    clusters: Tuple[str, str] = ("announce_date", "symbol"),
) -> InferenceResult:
    """OLS of panel[y] on [1, panel[X]] with Cameron-Gelbach-Miller TWO-WAY
    cluster-robust covariance, V = V_a + V_b - V_ab (V_ab clusters on the
    (a, b) intersection), each component carrying its own Stata small-sample
    factor (see module docstring — the convention that reproduces the
    published Petersen 2009 reference SEs).

    X=None -> intercept-only: the H1 mean test (H0: mean hedged event return
    <= 0; p_one_sided is the pre-registered decision number).

    Rows with NaN in y, any X column, or either cluster column are dropped.
    t ~ Student-t with df = min(G_a, G_b) - 1.
    """
    x_cols = list(X) if X else []
    dim_a, dim_b = clusters
    # dict.fromkeys dedupes while preserving order — the degenerate one-way
    # case clusters=(g, g) (used for the Petersen one-way pins) repeats a column.
    needed = list(dict.fromkeys([y] + x_cols + [dim_a, dim_b]))
    missing = [c for c in needed if c not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing columns {missing}")
    sub = panel[needed].dropna()
    n = len(sub)
    k = 1 + len(x_cols)
    if n <= k + 1:
        raise ValueError(f"too few complete observations (n={n}, k={k})")

    yv = sub[y].to_numpy(dtype=float)
    design = np.column_stack([np.ones(n)] + [sub[c].to_numpy(dtype=float) for c in x_cols])
    names = tuple(["const"] + x_cols)

    xtx = design.T @ design
    bread = np.linalg.inv(xtx)
    coef = bread @ design.T @ yv
    resid = yv - design @ coef

    # Naive iid OLS SE (reported for the optimism comparison).
    s2 = float(resid @ resid) / (n - k)
    se_ols = np.sqrt(np.diag(s2 * bread))

    ga = sub[dim_a].nunique()
    gb = sub[dim_b].nunique()
    if ga < 2 or gb < 2:
        raise ValueError(
            f"need >= 2 clusters per dimension (G_{dim_a}={ga}, G_{dim_b}={gb})"
        )
    inter = sub[dim_a].astype(str) + "\x1f" + sub[dim_b].astype(str)
    gab = inter.nunique()

    def _component(groups, g: int) -> np.ndarray:
        meat = cluster_robust_cov(design, resid, groups)
        return _stata_factor(g, n, k) * (bread @ meat @ bread)

    v_a = _component(sub[dim_a], ga)
    v_b = _component(sub[dim_b], gb)
    v_ab = _component(inter, gab)
    cov, psd_fixed = _psd_clip(v_a + v_b - v_ab)

    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, coef / np.where(se > 0, se, 1.0), 0.0)

    df = max(min(ga, gb) - 1, 1)
    from scipy import stats as _st
    p_one = _st.t.sf(tstat, df=df)          # H1: coef > 0
    p_two = 2.0 * _st.t.sf(np.abs(tstat), df=df)

    return InferenceResult(
        names=names,
        coef=coef,
        se=se,
        se_ols=se_ols,
        tstat=tstat,
        p_one_sided=p_one,
        p_two_sided=np.minimum(p_two, 1.0),
        n_obs=n,
        n_clusters={dim_a: int(ga), dim_b: int(gb), "intersection": int(gab)},
        df=df,
        cov=cov,
        psd_fixed=psd_fixed,
    )


def decile_report(
    panel: pd.DataFrame,
    feature: str,
    y: str,
    n: int = 10,
    clusters: Tuple[str, str] = ("announce_date", "symbol"),
) -> dict:
    """Per-decile report of `y` across `feature` deciles (REPORTED robustness,
    never the deciding number).

    Deciles are formed on rank(method='first') so ties cannot collapse bins.
    Per decile: n_obs, mean feature, mean y, and the two-way intercept-only t
    (None when the decile has too few clusters to infer on). Monotonicity:
    Spearman rho between decile index and decile mean y, plus a strict
    is_monotone flag (every consecutive decile-mean step in one direction).
    """
    sub = panel[[feature, y, *clusters]].dropna()
    if len(sub) < n:
        raise ValueError(f"too few complete rows ({len(sub)}) for {n} deciles")
    ranks = sub[feature].rank(method="first")
    sub = sub.assign(_decile=pd.qcut(ranks, n, labels=False))

    rows: List[dict] = []
    means: List[float] = []
    for d in range(n):
        grp = sub[sub["_decile"] == d]
        mean_y = float(grp[y].mean())
        means.append(mean_y)
        try:
            res = twoway_cluster_ols(grp, y=y, clusters=clusters)
            t_val: Optional[float] = float(res.tstat[0])
        except ValueError:
            t_val = None
        rows.append({
            "decile": d + 1,
            "n_obs": int(len(grp)),
            "mean_feature": float(grp[feature].mean()),
            "mean_y": mean_y,
            "twoway_t": t_val,
        })

    from scipy import stats as _st
    rho, _ = _st.spearmanr(np.arange(n), np.asarray(means))
    diffs = np.diff(np.asarray(means))
    if np.all(diffs > 0):
        direction: Optional[str] = "increasing"
    elif np.all(diffs < 0):
        direction = "decreasing"
    else:
        direction = None
    return {
        "feature": feature,
        "y": y,
        "n_deciles": n,
        "rows": rows,
        "spearman_rho": float(rho),
        "is_monotone": direction is not None,
        "direction": direction,
        "top_minus_bottom": float(means[-1] - means[0]),
    }


def loco_robustness(
    panel: pd.DataFrame,
    y: str,
    unit: str,
    infer_fn: Callable[[pd.DataFrame], InferenceResult],
) -> List[dict]:
    """Leave-one-cluster-out robustness (REPORTED, never re-deciding).

    unit:
      "quarter" — leave each earnings season (calendar quarter of
                  announce_date, via scripts.pead_significance.cluster_key) out;
      "sector"  — leave each GICS sector out;
      "top10"   — ONE run dropping the 10 largest |y| events (gap-monster cut).

    infer_fn(sub_panel) -> InferenceResult runs the same registered inference
    on each reduced panel (so LOCO can never drift from the primary spec).
    Returns one dict per left-out unit: left_out / n_obs / coef / tstat /
    p_one_sided (None metrics when the reduced panel is too thin to infer on).
    """
    def _row(label: str, sub: pd.DataFrame) -> dict:
        try:
            res = infer_fn(sub)
            return {
                "left_out": label,
                "n_obs": int(res.n_obs),
                "coef": float(res.coef[0]),
                "tstat": float(res.tstat[0]),
                "p_one_sided": float(res.p_one_sided[0]),
            }
        except ValueError as exc:
            return {"left_out": label, "n_obs": int(len(sub)), "coef": None,
                    "tstat": None, "p_one_sided": None, "error": str(exc)}

    if unit == "top10":
        order = panel[y].abs().sort_values(ascending=False, na_position="last")
        sub = panel.drop(index=order.index[:10])
        return [_row("top10_abs_y_dropped", sub)]

    if unit == "quarter":
        from scripts.pead_significance import cluster_key
        keys = panel["announce_date"].map(
            lambda d: cluster_key(pd.Timestamp(d).date())
        )
    elif unit == "sector":
        keys = panel["sector"]
    else:
        raise ValueError(f"unknown LOCO unit {unit!r} (quarter|sector|top10)")

    out: List[dict] = []
    for k in sorted(keys.dropna().unique()):
        out.append(_row(str(k), panel.loc[keys != k]))
    return out
