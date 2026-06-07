"""
Fold-coverage report — Alpha-v4 P0 validation integrity.

The #1 finding of the 2026-06-06 5-LLM review was that a fold set skewed toward
later (bull) regimes can make a strategy look clean when it was never tested in a
crisis. This module answers, for every CPCV run, "which calendar years and market
regimes did our out-of-sample evaluations actually touch?" — and flags a run whose
fold evaluations don't span enough years / regimes BEFORE its performance is read.

It is deliberately dependency-light and PURE: it consumes the per-fold test windows
already iterated in run_cpcv plus the strategy's pre-built regime map
(strategy._global_regime_map from regime.load_regime_map). No new VIX/SPY fetch.

Under the default REGIME_SCHEME='coarse3' the labels are BULL/BEAR/NEUTRAL (the
VIX+trend fusion). Under 'legacy16' they decompose to '<vix_quartile><trend><mom>'
(e.g. '4DN'), so the same year×regime breakdown also yields year×VIX×trend coverage.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

# Require fold evaluations to span at least this many distinct calendar years.
MIN_DISTINCT_YEARS = 3
# Require at least this many distinct (non-UNK) regime labels among the folds.
MIN_DISTINCT_REGIMES = 2


def _is_stress_label(label: str) -> bool:
    """True for a crisis/high-stress regime in either scheme.

    coarse3:  'BEAR'.  legacy16: top VIX quartile ('4...') or a downtrend ('?D?').
    """
    if not label or label == "UNK":
        return False
    if label == "BEAR":
        return True
    # legacy16 '<vix_quartile><trend><momentum>', e.g. '4DN'
    if label[0] == "4":
        return True
    if len(label) >= 2 and label[1] == "D":
        return True
    return False


def _majority_regime(te_start: date, te_end: date,
                     regime_map: Optional[Dict[date, str]]) -> str:
    """Most-common regime label over the calendar days in [te_start, te_end].

    Returns 'UNK' when no day in the window is labelled (or no map is available).
    """
    if not regime_map:
        return "UNK"
    counts: Counter = Counter()
    d = te_start
    while d <= te_end:
        lbl = regime_map.get(d)
        if lbl:
            counts[lbl] += 1
        d += timedelta(days=1)
    if not counts:
        return "UNK"
    return counts.most_common(1)[0][0]


@dataclass
class CoverageReport:
    """Year × regime coverage of the CPCV fold evaluations + a pass/flag verdict."""
    n_folds_evaluated: int
    by_year: Dict[int, int] = field(default_factory=dict)
    by_regime: Dict[str, int] = field(default_factory=dict)
    n_distinct_years: int = 0
    n_distinct_regimes: int = 0
    has_stress_fold: bool = False
    coverage_ok: bool = True
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_folds_evaluated": self.n_folds_evaluated,
            "by_year": {str(k): v for k, v in sorted(self.by_year.items())},
            "by_regime": dict(sorted(self.by_regime.items())),
            "n_distinct_years": self.n_distinct_years,
            "n_distinct_regimes": self.n_distinct_regimes,
            "has_stress_fold": self.has_stress_fold,
            "coverage_ok": self.coverage_ok,
            "warnings": list(self.warnings),
        }

    def render(self) -> str:
        """Human-readable table — printed BEFORE performance in CPCVResult.print()."""
        head = "✅ FOLD COVERAGE OK" if self.coverage_ok else "⚠️  LOW FOLD COVERAGE"
        lines = [
            f"{head} — {self.n_folds_evaluated} fold eval(s), "
            f"{self.n_distinct_years} year(s), {self.n_distinct_regimes} regime(s)",
        ]
        if self.by_year:
            yr = "  ".join(f"{y}:{n}" for y, n in sorted(self.by_year.items()))
            lines.append(f"  by year:   {yr}")
        if self.by_regime:
            rg = "  ".join(f"{r}:{n}" for r, n in sorted(self.by_regime.items()))
            lines.append(f"  by regime: {rg}")
        for w in self.warnings:
            lines.append(f"  ⚠️  {w}")
        return "\n".join(lines)


def build_fold_coverage(
    fold_windows: List[Tuple[date, date]],
    regime_map: Optional[Dict[date, str]] = None,
    *,
    min_years: int = MIN_DISTINCT_YEARS,
    min_regimes: int = MIN_DISTINCT_REGIMES,
) -> CoverageReport:
    """Bucket evaluated CPCV test windows by year and regime; flag thin coverage.

    fold_windows: (test_start, test_end) per fold ACTUALLY EVALUATED (deduped — one
        entry per distinct test fold, not per CPCV combination).
    regime_map: {date: label} from the strategy's pre-built global regime map.

    Soft gate: `coverage_ok` is False (with explicit `warnings`) when the fold set
    spans < min_years distinct years, < min_regimes distinct regimes, or contains no
    stress (BEAR / high-VIX) fold. Soft by design — the caller reports + flags for
    human review (it does not hard-block legacy runs), mirroring requires_human_review.
    """
    by_year: Counter = Counter()
    by_regime: Counter = Counter()
    for (te_start, te_end) in fold_windows:
        by_year[te_start.year] += 1
        by_regime[_majority_regime(te_start, te_end, regime_map)] += 1

    non_unk = {r for r in by_regime if r != "UNK"}
    has_stress = any(_is_stress_label(r) for r in by_regime)

    warnings: List[str] = []
    if len(by_year) < min_years:
        warnings.append(
            f"fold evaluations span only {len(by_year)} distinct year(s) (want >= {min_years})"
        )
    if not regime_map:
        warnings.append("no regime map available — regime coverage UNKNOWN")
    else:
        if len(non_unk) < min_regimes:
            warnings.append(
                f"fold evaluations cover only {len(non_unk)} distinct regime(s) (want >= {min_regimes})"
            )
        if not has_stress:
            warnings.append("no stress-regime (BEAR / high-VIX) fold evaluated")

    return CoverageReport(
        n_folds_evaluated=len(fold_windows),
        by_year=dict(by_year),
        by_regime=dict(by_regime),
        n_distinct_years=len(by_year),
        n_distinct_regimes=len(non_unk),
        has_stress_fold=has_stress,
        coverage_ok=len(warnings) == 0,
        warnings=warnings,
    )
