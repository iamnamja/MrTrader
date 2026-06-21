"""
tail_diagnostics.py — Alpha-v10 GL-1: the TAIL / co-crash diagnostics for the multi-premia book.

The Go-Live panel was unanimous that the four premia (trend, carry, xs-momentum, VIX-VRP) are
**net short-crisis / "effectively one bet"** and that the unconditional correlations (~0.49
post-2015) are "dangerously misleading" — what matters is the TAIL correlation. This module
operationalises the panel's requested tests so the verdict (VRP in/out, defensive-sleeve go/no-go)
rests on measured tail behaviour, not assertion:

  1. EXCEEDANCE (stress-conditional) CORRELATION — pairwise correlation among the sleeves restricted
     to the worst-q% days of a conditioner (equity = SPY, or the book itself). Diversified books show
     flat/declining lower-tail correlation; a hidden one-bet shows a rising "smile". "One bet" if the
     avg off-diagonal stress correlation exceeds ~0.6.
  2. DOWN-vs-UP BETA (Claude's "the tell") — regress each sleeve (and the book) on SPY separately on
     down-market vs up-market days. down_beta > up_beta = NEGATIVE CONVEXITY / a hidden long-risk
     bet (the book loses more when equities fall than it gains when they rise).
  3. CRISIS-WINDOW REPLAY — each sleeve's cumulative return through the named crises (2008, Aug-2011,
     Aug-2015, Feb-2018, Q4-2018, COVID-2020, 2022, Mar-2023): who offsets vs who compounds.

VRP verdict: compare the inverse-vol book WITH vs WITHOUT VRP on (down-beta asymmetry, stress
correlation, crisis cumret). If VRP worsens the tail (more negative down-up asymmetry, higher stress
corr, deeper crisis losses) it does not belong as a naked sleeve.
Defensive-sleeve verdict: if the book's down_beta materially exceeds its up_beta (negative
convexity), a deliberately convex/defensive sleeve is needed.

PURE + deterministic. Inputs are daily return series; the book weights are inverse-vol (full-sample,
a historical-characterisation choice — flagged; NOT a live-sizing decision)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ANN = 252

# Named crisis windows (ISO date ranges) — the equity/vol stress episodes in the sleeve overlap.
DEFAULT_CRISES: Dict[str, Tuple[str, str]] = {
    "GFC_2008": ("2008-09-01", "2009-03-31"),
    "EuroAug_2011": ("2011-08-01", "2011-10-31"),
    "Aug_2015": ("2015-08-01", "2015-09-30"),
    "Volmageddon_Feb2018": ("2018-02-01", "2018-02-28"),
    "Q4_2018": ("2018-10-01", "2018-12-31"),
    "COVID_2020": ("2020-02-15", "2020-04-30"),
    "Bear_2022": ("2022-01-01", "2022-10-31"),
    "BankStress_Mar2023": ("2023-03-01", "2023-03-31"),
}


# ---------------------------------------------------------------- core diagnostics
def exceedance_correlation(sleeves: pd.DataFrame, conditioner: pd.Series,
                           q: float) -> Dict[str, object]:
    """Pairwise correlation among `sleeves` columns on the worst-q-quantile days of `conditioner`.
    Returns {matrix, avg_offdiag, max_offdiag, n_days, threshold}."""
    aligned = sleeves.join(conditioner.rename("_cond"), how="inner").dropna()
    if len(aligned) < 50:
        return {"matrix": pd.DataFrame(), "avg_offdiag": float("nan"),
                "max_offdiag": float("nan"), "n_days": len(aligned), "threshold": float("nan")}
    thr = float(aligned["_cond"].quantile(q))
    sub = aligned[aligned["_cond"] <= thr].drop(columns="_cond")
    corr = sub.corr()
    mask = ~np.eye(len(corr), dtype=bool)
    od = corr.values[mask]
    od = od[np.isfinite(od)]
    return {"matrix": corr, "avg_offdiag": float(od.mean()) if od.size else float("nan"),
            "max_offdiag": float(od.max()) if od.size else float("nan"),
            "n_days": int(len(sub)), "threshold": thr}


def down_up_beta(sleeve: pd.Series, market: pd.Series) -> Dict[str, float]:
    """OLS beta of `sleeve` on `market` separately on down-market (market<0) vs up-market days.
    down_beta > up_beta => negative convexity (the tell of a hidden long-risk bet)."""
    a = pd.concat([sleeve.rename("s"), market.rename("m")], axis=1, join="inner").dropna()

    def _beta(df: pd.DataFrame) -> float:
        v = float(df["m"].var())
        return float(df["s"].cov(df["m"]) / v) if v > 0 else float("nan")

    down, up = a[a["m"] < 0], a[a["m"] > 0]
    db, ub = _beta(down), _beta(up)
    return {"down_beta": db, "up_beta": ub,
            "asymmetry": (db - ub) if (np.isfinite(db) and np.isfinite(ub)) else float("nan"),
            "n_down": int(len(down)), "n_up": int(len(up))}


def crisis_replay(sleeves: pd.DataFrame,
                  windows: Optional[Dict[str, Tuple[str, str]]] = None) -> pd.DataFrame:
    """Per-crisis cumulative return of each sleeve column (compounded over the window)."""
    windows = windows or DEFAULT_CRISES
    rows = []
    for name, (s, e) in windows.items():
        w = sleeves.loc[(sleeves.index >= pd.Timestamp(s)) & (sleeves.index <= pd.Timestamp(e))]
        w = w.dropna(how="all")
        if len(w) < 3:
            continue
        row = {"crisis": name, "days": int(len(w))}
        for c in sleeves.columns:
            col = w[c].dropna()
            row[c] = float((1.0 + col).prod() - 1.0) if len(col) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("crisis") if rows else pd.DataFrame()


def inverse_vol_book(sleeves: pd.DataFrame) -> pd.Series:
    """Equal-risk (inverse-full-sample-vol) combine of the sleeve columns on their common dates.
    A historical-characterisation choice for the tail diagnostics — NOT a live-sizing decision."""
    s = sleeves.dropna(how="any")
    vol = s.std()
    w = (1.0 / vol.replace(0.0, np.nan))
    w = (w / w.sum()).fillna(0.0)
    return (s * w).sum(axis=1)


# ---------------------------------------------------------------- assembled verdict
def bootstrap_asymmetry_ci(book: pd.Series, market: pd.Series, *,
                           n_boot: int = 1000, seed: int = 0) -> Tuple[float, float, float]:
    """(point, 5th, 95th) percentile CI of the down-up beta ASYMMETRY via an iid bootstrap of the
    (book, market) day pairs. (iid, not block — a diagnostic CI; mild daily autocorrelation makes
    this slightly tight, which is conservative for an "is it clearly fine" call.)"""
    a = pd.concat([book.rename("s"), market.rename("m")], axis=1, join="inner").dropna()
    n = len(a)
    if n < 100:
        b = down_up_beta(a["s"], a["m"])
        return b["asymmetry"], float("nan"), float("nan")
    arr = a.to_numpy()
    rng = np.random.default_rng(seed)
    asyms = []
    for _ in range(n_boot):
        d = arr[rng.integers(0, n, n)]
        m = d[:, 1]
        down, up = d[m < 0], d[m > 0]
        vd, vu = down[:, 1].var(), up[:, 1].var()
        if vd > 0 and vu > 0:
            db = np.cov(down[:, 0], down[:, 1])[0, 1] / vd
            ub = np.cov(up[:, 0], up[:, 1])[0, 1] / vu
            asyms.append(db - ub)
    point = down_up_beta(a["s"], a["m"])["asymmetry"]
    if not asyms:
        return point, float("nan"), float("nan")
    lo, hi = (float(x) for x in np.percentile(asyms, [5, 95]))
    return point, lo, hi


@dataclass
class TailDiagnosticsResult:
    sleeves: List[str]
    uncond_corr_avg: float
    # exceedance corr (avg off-diagonal) conditioned on SPY, full-history AND post-2015
    spy_exceedance: Dict[float, float]
    spy_exceedance_post2015: Dict[float, float]
    # book-conditioned exceedance — REPORT-ONLY and BIASED LOW (collider: a sum's worst days induce
    # negative dependence among its addends); NOT used in the one_bet verdict.
    book_exceedance: Dict[float, float]
    # down/up beta of each sleeve + the books vs SPY
    sleeve_beta: Dict[str, Dict[str, float]]
    book_with_vrp_beta: Dict[str, float]
    book_without_vrp_beta: Dict[str, float]
    core_asymmetry_ci: Tuple[float, float, float]      # (point, 5th, 95th) ex-VRP book
    crisis_table: pd.DataFrame
    vrp_crises_worse: int                              # #crises where book-with-VRP < book-without
    vrp_crises_total: int
    one_bet: bool
    vrp_worsens_tail: bool
    defensive_sleeve_needed: bool
    verdict_notes: List[str] = field(default_factory=list)


def run_tail_diagnostics(sleeves: pd.DataFrame, spy: pd.Series, *,
                         vrp_col: str = "vrp",
                         one_bet_corr: float = 0.60,
                         windows: Optional[Dict[str, Tuple[str, str]]] = None
                         ) -> TailDiagnosticsResult:
    """Full GL-1 tail diagnostics. `sleeves` columns are daily returns (expects trend/carry/xsmom
    and optionally `vrp_col`); `spy` is the equity conditioner/market for beta."""
    cols = list(sleeves.columns)
    spy_ret = spy.copy()

    uncond = sleeves.corr()
    od = uncond.values[~np.eye(len(uncond), dtype=bool)]
    uncond_avg = float(np.nanmean(od)) if od.size else float("nan")

    # SPY-conditioned exceedance correlation — full-history AND post-2015 (the panel flagged the
    # modern regime as more correlated; the WORSE window drives the verdict).
    spy_exc = {q: exceedance_correlation(sleeves, spy_ret, q)["avg_offdiag"] for q in (0.10, 0.05)}
    post = sleeves[sleeves.index >= pd.Timestamp("2015-01-01")]
    spy_post = spy_ret[spy_ret.index >= pd.Timestamp("2015-01-01")]
    spy_exc_post = {q: exceedance_correlation(post, spy_post, q)["avg_offdiag"] for q in (0.10, 0.05)}
    # book-conditioned (REPORT-ONLY, biased low — collider on the aggregate)
    book_exc = {q: exceedance_correlation(sleeves, inverse_vol_book(sleeves), q)["avg_offdiag"]
                for q in (0.10, 0.05)}

    sleeve_beta = {c: down_up_beta(sleeves[c], spy_ret) for c in cols}

    # with/without-VRP books on the COMMON 4-sleeve date index (so the comparison is same-sample)
    common = sleeves.dropna(how="any")
    cols_no_vrp = [c for c in cols if c != vrp_col]
    with_vrp = inverse_vol_book(common)
    without_vrp = inverse_vol_book(common[cols_no_vrp]) if cols_no_vrp else with_vrp
    b_with = down_up_beta(with_vrp, spy_ret)
    b_without = down_up_beta(without_vrp, spy_ret)
    core_ci = bootstrap_asymmetry_ci(without_vrp, spy_ret)

    crisis = crisis_replay(sleeves, windows)
    # book-delta crisis comparison: does adding VRP DEEPEN the book's loss in a majority of crises?
    book_crises = crisis_replay(pd.DataFrame({"with_vrp": with_vrp, "without_vrp": without_vrp}),
                                windows)
    vrp_worse_n = vrp_total = 0
    if not book_crises.empty:
        vrp_total = int(len(book_crises))
        vrp_worse_n = int((book_crises["with_vrp"] < book_crises["without_vrp"] - 1e-9).sum())

    # --- verdicts ---
    vrp_present = vrp_col in cols
    # one_bet: the WORSE of full-history / post-2015 SPY-worst-5% stress correlation.
    stress_full = spy_exc.get(0.05, float("nan"))
    stress_post = spy_exc_post.get(0.05, float("nan"))
    stress_worst = float(np.nanmax([stress_full, stress_post]))
    one_bet = bool(np.isfinite(stress_worst) and stress_worst > one_bet_corr)

    # VRP worsens the tail — PRIMARY: deepens the book's loss in a majority of crises (book-delta,
    # weight-honest on-point test). CORROBORATION: VRP loses standalone in a majority of crises.
    crisis_worse = bool(vrp_present and vrp_total and vrp_worse_n / vrp_total > 0.5)
    standalone_worse = bool(vrp_present and not crisis.empty and vrp_col in crisis.columns
                            and (crisis[vrp_col] < 0).mean() > 0.5)
    vrp_worsens_tail = bool(crisis_worse or standalone_worse)

    # Defensive sleeve: judge the core (ex-VRP) book asymmetry against the bar with a bootstrap CI.
    # needed if even the CI LOWER bound exceeds the bar; clearly-fine if the CI UPPER bound is below
    # it; otherwise BORDERLINE (recommended) — the +0.08 point is net negatively convex but noisy.
    point, ci_lo, ci_hi = core_ci
    if np.isfinite(ci_lo) and ci_lo > 0.10:
        defensive_needed, defensive_label = True, "NEEDED (CI lower bound > 0.10)"
    elif np.isfinite(ci_hi) and ci_hi < 0.10:
        defensive_needed, defensive_label = False, "not needed (CI upper bound < 0.10)"
    else:
        defensive_needed, defensive_label = False, "BORDERLINE — recommended (net negatively convex, CI spans 0.10)"

    notes = [
        f"unconditional avg pairwise corr = {uncond_avg:.2f}; SPY-worst-5% stress corr = "
        f"{stress_full:.2f} full / {stress_post:.2f} post-2015 -> "
        f"{'ONE BET' if one_bet else 'diversified'} (worse {stress_worst:.2f} vs bar {one_bet_corr}).",
        f"core book (ex-VRP) down-beta {b_without['down_beta']:+.2f} vs up-beta "
        f"{b_without['up_beta']:+.2f}, asymmetry {point:+.2f} [90% CI {ci_lo:+.2f}, {ci_hi:+.2f}] "
        f"-> defensive sleeve {defensive_label}.",
        (f"VRP: deepens the book's crisis loss in {vrp_worse_n}/{vrp_total} crises; standalone "
         f"with-VRP asymmetry {b_with['asymmetry']:+.2f} vs without {b_without['asymmetry']:+.2f} "
         f"(confounded — inverse-vol over-weights low-vol VRP). -> VRP "
         f"{'WORSENS the tail (drop / pair with long-vol)' if vrp_worsens_tail else 'does not clearly worsen the tail'}."
         if vrp_present else "VRP not in the supplied sleeve set."),
    ]

    return TailDiagnosticsResult(
        sleeves=cols, uncond_corr_avg=uncond_avg,
        spy_exceedance=spy_exc, spy_exceedance_post2015=spy_exc_post, book_exceedance=book_exc,
        sleeve_beta=sleeve_beta, book_with_vrp_beta=b_with, book_without_vrp_beta=b_without,
        core_asymmetry_ci=core_ci, crisis_table=crisis,
        vrp_crises_worse=vrp_worse_n, vrp_crises_total=vrp_total,
        one_bet=one_bet, vrp_worsens_tail=vrp_worsens_tail,
        defensive_sleeve_needed=defensive_needed, verdict_notes=notes)
