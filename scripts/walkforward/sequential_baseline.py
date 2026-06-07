"""
Purged sequential walk-forward baseline — Alpha-v4 P0 validation integrity.

For TRAINED models, CPCV-with-rolling-retrain is inherently low-coverage: the
overlap guard legitimately skips folds whose train window spans a prior test fold
(in-sample leakage). The honest complement is a purged SEQUENTIAL walk-forward —
expanding/rolling `[start, t]` train → `[t + purge, t + 1]` test, sliding forward,
EVERY out-of-sample block evaluated, zero holes — reported ALONGSIDE CPCV as a
sanity baseline (the 2026-06-06 5-LLM review's recommendation for trained models).

This is a thin wrapper over the existing, already-tested `FoldEngine.run`
(scripts/walkforward/engine.py) — which is exactly that purged sequential machine
and honours `strategy.per_fold_retrain`. It is a BASELINE, not a promotion path:
under `GATE_MODE='significance'` a `WalkForwardReport` has no path t-stat, so its
gate is INCONCLUSIVE by design. We surface the numbers, not a verdict.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def run_sequential_baseline(
    strategy,
    *,
    n_folds: int,
    purge_days: int,
    total_years: Optional[int] = None,
    total_days: Optional[int] = None,
    embargo_days: Optional[int] = None,
    train_years: Optional[int] = None,
    regime_map: Optional[dict] = None,
    allow_sacred_holdout: bool = False,
):
    """Run a purged sequential walk-forward and return its WalkForwardReport.

    Reuses the strategy's pre-built regime map (for the per-fold diversity log) when
    one isn't passed. Exactly one of total_years / total_days must be given (same
    contract as FoldEngine.run).
    """
    from scripts.walkforward.engine import FoldEngine

    _rm = regime_map if regime_map is not None else getattr(strategy, "_global_regime_map", None)
    engine = FoldEngine(
        strategy,
        purge_days=purge_days,
        embargo_days=embargo_days,
        regime_map=_rm,
    )
    logger.info(
        "Sequential-WF baseline: n_folds=%d purge=%d train_years=%s (sanity baseline, not a gate)",
        n_folds, purge_days, train_years,
    )
    return engine.run(
        n_folds=n_folds,
        total_years=total_years,
        total_days=total_days,
        train_years=train_years,
        allow_sacred_holdout=allow_sacred_holdout,
    )


def print_baseline_vs_cpcv(wf_report, cpcv_result) -> None:
    """Print the sequential-WF baseline next to the CPCV result (sanity comparison).

    Both are defensively read via getattr so a partially-populated result never raises.
    The baseline is a point estimate (no path distribution) — divergence between the
    two is the signal: a CPCV number far above the sequential baseline suggests the
    CPCV mean is riding favourable-fold selection / low coverage.
    """
    def _g(obj, name, default=None):
        return getattr(obj, name, default)

    seq_avg = _g(wf_report, "avg_sharpe")
    seq_min = _g(wf_report, "min_sharpe")
    seq_wr = _g(wf_report, "worst_regime_sharpe")
    seq_n = len(_g(wf_report, "folds", []) or [])

    cpcv_mean = _g(cpcv_result, "mean_sharpe")
    cpcv_p5 = _g(cpcv_result, "p5_sharpe")
    cpcv_wr = _g(cpcv_result, "worst_regime_sharpe")
    cpcv_n = _g(cpcv_result, "n_folds")

    def _f(x):
        return f"{x:+.3f}" if isinstance(x, (int, float)) else "n/a"

    print("\n  ── Sequential-WF baseline vs CPCV (sanity; baseline is NOT a gate) ──")
    print(f"  {'metric':<22}{'CPCV':>12}{'Sequential-WF':>16}")
    print(f"  {'avg/mean Sharpe':<22}{_f(cpcv_mean):>12}{_f(seq_avg):>16}")
    print(f"  {'P5 / min fold Sharpe':<22}{_f(cpcv_p5):>12}{_f(seq_min):>16}")
    print(f"  {'worst-regime Sharpe':<22}{_f(cpcv_wr):>12}{_f(seq_wr):>16}")
    print(f"  {'folds':<22}{str(cpcv_n or 'n/a'):>12}{str(seq_n or 'n/a'):>16}")
    # Flag a large optimistic gap (CPCV >> baseline) — the low-coverage tell.
    try:
        if isinstance(cpcv_mean, (int, float)) and isinstance(seq_avg, (int, float)):
            gap = cpcv_mean - seq_avg
            if gap > 0.30:
                print(f"  ⚠️  CPCV mean exceeds the sequential baseline by {gap:+.2f} — "
                      f"check fold coverage (CPCV may be riding favourable folds).")
    except Exception:
        pass
