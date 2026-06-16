"""
stability.py — Alpha-v9 P0-2 (Ⓑ): a POWERED sub-period stability test that replaces
the binary "positive Sharpe in both halves" guard.

WHY the binary guard was wrong
------------------------------
The old guard (`SR(H1) > 0 AND SR(H2) > 0`) has NO power accounting. A genuine edge of
annual SR 0.5 split into two ~5-year halves has a per-half SR estimation SE ≈ 1/√5 ≈
0.45, so P(positive in one half) ≈ Φ(0.5/0.45) ≈ 0.87 and P(positive in BOTH) ≈ 0.76 —
i.e. a ~24% false-negative rate on a REAL SR-0.5 edge, from this guard alone. It cannot
tell "unstable/regime-fluke" (what we want to catch) from "real but noisy" (what we want
to keep) — it just rejects edges unlucky enough to dip below zero in one half. It killed
the carry sleeve (H1 +0.69 / H2 −0.10) even though the two halves are not statistically
distinguishable given only ~9 years each.

WHAT "stable" should mean
-------------------------
"No evidence of a STRUCTURAL BREAK" — the two half-Sharpes are not significantly
different — NOT "lucky enough to be positive twice". We test that via a stationary
(block) bootstrap of the half-Sharpe DIFFERENCE: resample each half independently,
recompute SR(H1)−SR(H2), and check whether the bootstrap CI of that difference overlaps
zero. Overlap ⇒ indistinguishable ⇒ no detectable break ⇒ STABLE.

Two outputs for two use-cases:
  • `halves_indistinguishable` — the pure stability verdict (the drop-in replacement
    for the old "both halves positive"). Use this when a SEPARATE test already
    establishes the edge exists (e.g. Track-B book-delta admits a diversifier on its
    appraisal-IR / P(ΔSR>0), so only the no-break check is needed on top).
  • `passed` — the SAFE standalone verdict = no-break AND the pooled HAC-SR is
    significant. A pure-noise series has trivially-indistinguishable halves, so
    stability ALONE must never be a gate; `passed` adds the edge requirement so a
    naive caller can't admit noise.

PURE: reads its argument, mutates nothing, no I/O. Reuses the project's stationary
bootstrap + HAC-Sharpe from app.research.inference.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

from app.research.inference import (
    ANN, hac_sharpe, _stationary_bootstrap_index_matrix, _auto_block_len,
)

# A half with fewer than this many observations is too short to estimate a Sharpe the
# bootstrap can trust; the test fails closed (cannot certify stability).
MIN_HALF_OBS = 60
# Significance level for the pooled HAC-SR leg (one-sided).
POOLED_ALPHA = 0.05
# A half-Sharpe-difference 95% CI wider than this (annualized SR units) means the "no
# break" conclusion is absence-of-evidence (under-powered), not evidence-of-stability.
# Report-only — surfaced via `weakly_powered` so a wide-CI pass is never silently
# equated with a tight-CI pass.
WIDE_DIFF_CI = 1.5


@dataclass
class StabilityResult:
    n: int
    n_h1: int
    n_h2: int
    sr_h1: float
    sr_h2: float
    sr_diff: float                  # SR(H1) - SR(H2), point estimate
    diff_ci_low: float              # bootstrap 95% CI of the half-Sharpe difference
    diff_ci_high: float
    halves_indistinguishable: bool  # CI overlaps 0 -> no detectable structural break
    pooled_sr: float
    pooled_hac_t: float
    pooled_hac_p_one_sided: float
    pooled_significant: bool        # pooled HAC-SR significant at POOLED_ALPHA (1-sided)
    both_halves_positive: bool      # the OLD binary criterion — REPORT-ONLY, for comparison
    weakly_powered: bool            # the difference CI is so wide the "no break" is absence-of-
    #                                 evidence, not evidence-of-stability (transparency flag)
    block_len: float
    n_boot: int
    split_label: str
    # SAFE standalone gate: STABLE (no detectable break) AND the pooled edge is significant.
    # A caller using a SEPARATE edge test (e.g. Track-B book-delta) should instead read
    # `halves_indistinguishable` directly — significance is then established by that test, and
    # requiring pooled significance here would double-gate (and wrongly reject a book-delta
    # diversifier like carry, whose standalone pooled HAC deliberately misses).
    passed: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _ann_sr(r: np.ndarray) -> float:
    sd = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
    return float(np.mean(r) / sd * np.sqrt(ANN)) if sd > 0 else 0.0


def _sr_rows(mat: np.ndarray) -> np.ndarray:
    """Annualized Sharpe per row of an (n_boot, n) matrix; constant row -> 0.0."""
    mu = mat.mean(axis=1)
    sd = mat.std(axis=1, ddof=1)
    return np.where(sd > 0, mu / sd * np.sqrt(ANN), 0.0)


def stability_test(
    returns: pd.Series,
    *,
    split: Optional[pd.Timestamp] = None,
    n_boot: int = 2000,
    block_len: Optional[float] = None,
    seed: int = 0,
    hac_lag: int = 5,
) -> StabilityResult:
    """Powered sub-period stability test on a daily return series.

    split: optional date boundary; default = midpoint by observation count.
    Returns a StabilityResult with TWO verdicts (see the module docstring):
      • `halves_indistinguishable` — the pure no-break check (drop-in for the old
        "both halves positive"); use this WITH a separate edge test.
      • `passed` — the SAFE standalone verdict = no-break AND pooled HAC-SR
        significant, so a pure-noise series (indistinguishable but edge-less) is
        rejected.
    Fails closed (both False) when either half is shorter than MIN_HALF_OBS.
    """
    r = returns.dropna()
    if not isinstance(r.index, pd.DatetimeIndex):
        raise TypeError("stability_test requires a pd.DatetimeIndex on `returns`")
    n = int(len(r))

    if split is not None:
        h1 = r[r.index < split]
        h2 = r[r.index >= split]
        split_label = f"split@{pd.Timestamp(split).date()}"
    else:
        mid = n // 2
        h1, h2 = r.iloc[:mid], r.iloc[mid:]
        split_label = "midpoint"

    a1 = h1.to_numpy(dtype=float)
    a2 = h2.to_numpy(dtype=float)
    n1, n2 = int(a1.size), int(a2.size)
    sr1, sr2 = _ann_sr(a1), _ann_sr(a2)
    sr_diff = float(sr1 - sr2)
    both_pos = bool(sr1 > 0 and sr2 > 0)

    # Pooled HAC-SR significance (does the edge exist over the full window?).
    hac = hac_sharpe(r.to_numpy(dtype=float), hac_lag=hac_lag)
    pooled_significant = bool(hac.gating and hac.p_one_sided < POOLED_ALPHA)

    if n1 < MIN_HALF_OBS or n2 < MIN_HALF_OBS:
        return StabilityResult(
            n=n, n_h1=n1, n_h2=n2, sr_h1=sr1, sr_h2=sr2, sr_diff=sr_diff,
            diff_ci_low=float("nan"), diff_ci_high=float("nan"),
            halves_indistinguishable=False,
            pooled_sr=float(hac.sr_ann), pooled_hac_t=float(hac.t_stat),
            pooled_hac_p_one_sided=float(hac.p_one_sided),
            pooled_significant=pooled_significant, both_halves_positive=both_pos,
            weakly_powered=True,
            block_len=1.0, n_boot=0, split_label=split_label,
            passed=False,
            reason=f"a half is too short for a powered test (n_h1={n1}, n_h2={n2}, "
                   f"min={MIN_HALF_OBS}) — failing closed")

    # Bootstrap the half-Sharpe DIFFERENCE: resample each half independently with the
    # stationary block bootstrap (preserves within-half autocorrelation), recompute
    # SR(H1*) - SR(H2*). The CI of that distribution is the powered stability instrument.
    blk = float(block_len) if block_len is not None else max(
        1.0, 0.5 * (_auto_block_len(a1) + _auto_block_len(a2)))
    rng = np.random.default_rng(seed)
    idx1 = _stationary_bootstrap_index_matrix(n1, 1.0 / blk, int(n_boot), rng)
    idx2 = _stationary_bootstrap_index_matrix(n2, 1.0 / blk, int(n_boot), rng)
    diff_bs = _sr_rows(a1[idx1]) - _sr_rows(a2[idx2])
    lo, hi = (float(x) for x in np.percentile(diff_bs, [2.5, 97.5]))
    indistinguishable = bool(lo <= 0.0 <= hi)
    weakly_powered = bool((hi - lo) > WIDE_DIFF_CI)
    # `passed` is the SAFE standalone verdict: no detectable break AND the pooled edge
    # is significant — so a pure-noise series (halves trivially indistinguishable but no
    # edge) can NOT pass. Callers with a separate edge test read `halves_indistinguishable`.
    passed = bool(indistinguishable and pooled_significant)

    return StabilityResult(
        n=n, n_h1=n1, n_h2=n2, sr_h1=sr1, sr_h2=sr2, sr_diff=sr_diff,
        diff_ci_low=lo, diff_ci_high=hi, halves_indistinguishable=indistinguishable,
        pooled_sr=float(hac.sr_ann), pooled_hac_t=float(hac.t_stat),
        pooled_hac_p_one_sided=float(hac.p_one_sided),
        pooled_significant=pooled_significant, both_halves_positive=both_pos,
        weakly_powered=weakly_powered,
        block_len=blk, n_boot=int(n_boot), split_label=split_label,
        passed=passed,
        reason=(("stable + significant" if passed else
                 "stable but pooled edge NOT significant" if indistinguishable else
                 "UNSTABLE — half-Sharpe difference CI excludes 0 (structural break)")
                + f"; diff CI [{lo:+.2f}, {hi:+.2f}]"
                + (" [weakly powered]" if weakly_powered else "")))
