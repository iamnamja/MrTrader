#!/usr/bin/env python
"""
smoke_test_per_fold.py — REAL-DATA integration smoke test for the per-fold-retrain CPCV pipeline.

═══════════════════════════════════════════════════════════════════════════════
PURPOSE
───────────────────────────────────────────────────────────────────────────────
Run a MINIMAL but REAL per-fold Combinatorial Purged CV (CPCV) for swing and/or
intraday and assert the pipeline actually produces surviving paths. This is the
end-to-end "does the real data flow through build_train_matrix_for_window → fit →
simulate and yield ≥1 path?" check that the unit suite cannot give.

This is NOT a pytest test and does NOT run in the default CI suite. It needs real
data (yfinance / Polygon cache / network) and is too slow/heavy for CI. Run it
NIGHTLY, or manually BEFORE merging any PR that touches the per-fold/CPCV/training
pipeline (cpcv.py, retrainers.py, strategies/*, training.py, intraday_training.py).

───────────────────────────────────────────────────────────────────────────────
WHY THE UNIT TESTS MISSED TWO PRODUCTION BUGS
───────────────────────────────────────────────────────────────────────────────
Two per-fold "empty training matrix → 0 surviving paths" bugs shipped to prod and
only surfaced on real ~8-minute runs, because the unit tests fed MOCKED/synthetic
frames that ALWAYS have full coverage and asserted len(X)>0 only on those synthetic
frames — so they passed VACUOUSLY against the real-data failure modes:

  • PR #339 / #PERFOLD2 (swing): the regime LABEL map ({date: "BULL"}) was passed
    where a numeric regime SCORE map was expected. The worker did float("BULL") →
    ValueError, swallowed by a bare `except Exception: continue`, so EVERY training
    row was dropped → empty matrix → 0 paths. Synthetic tests mocked the spine and
    never exercised the string-label map, so len(X)>0 passed vacuously.

  • PR #342 / #C14-1 (intraday): the fold day-axis (all_days_sorted) and the
    per-fold matrix-builder day-axis (derived from the loaded 5-min bars) desynced.
    When all_days_sorted spanned EARLIER than the actual bars, the first folds got
    train windows BEFORE any data existed → train_days={} → empty matrix → 0 paths.
    Synthetic tests always had aligned coverage, so this never triggered.

In BOTH cases the symptom was identical and machine-checkable: the run produced
ZERO surviving CPCV paths (n_combinations == 0). That is exactly the non-vacuous
invariant this smoke test asserts against REAL data.

───────────────────────────────────────────────────────────────────────────────
WHAT IT ASSERTS (the invariants the unit tests could not)
───────────────────────────────────────────────────────────────────────────────
For each model run:
  1. result.n_combinations > 0           — ≥1 surviving path (catches BOTH bugs).
  2. ≥1 fold actually fit a model        — at least one window built a non-empty
                                            matrix and trained (NOT every fold
                                            raised "no training samples"). Some
                                            skips are fine; ALL-skipped is the bug.
  3. result.is_true_walkforward is True   — genuine per-fold OOS, not frozen mode.
  4. per-fold trained_through == fold tr_end for ≥1 fold — proves real per-fold
                                            training happened (the fresh fold model
                                            pinned its cutoff to its training-window
                                            upper bound, not a frozen model cutoff).
  5. (intraday only) the shallow daily-bar coverage WARNING did NOT fire — confirms
     PR #343's yfinance-daily fix kept 52w/vol features at full coverage.

Exit code 0 on PASS, 1 on FAIL — suitable for a nightly cron / pre-merge hook.

───────────────────────────────────────────────────────────────────────────────
HOW TO RUN
───────────────────────────────────────────────────────────────────────────────
    python scripts/smoke_test_per_fold.py --model both       # swing + intraday
    python scripts/smoke_test_per_fold.py --model swing
    python scripts/smoke_test_per_fold.py --model intraday
    python scripts/smoke_test_per_fold.py --model both --dry-run   # wiring only, NO data fetch

Config is intentionally MINIMAL (30 symbols, cpcv-k=4, cpcv-paths=2, short window,
--as-of 2026-05-29) — integration coverage, not a real performance number. Target
runtime: a few minutes, not an hour.

IMPORTANT: importing this module triggers NO data fetch. Data is fetched only when
run() is invoked (i.e. only when the smoke test is actually executed, not on import),
so a future CI `--dry-run`/import-check stays fast.

See docs/living/PIPELINE_ARCHITECTURE.md §12 (Known Limitations / Testing).
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Make the repo root importable so `scripts.walkforward_tier3` and
# `scripts.walkforward.*` resolve regardless of the invoking cwd (mirrors
# walkforward_tier3.py's own sys.path bootstrap). No data fetch — path setup only.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Minimal real config (small + fast: integration coverage, not perf) ─────────
DEFAULT_AS_OF = "2026-05-29"
CPCV_K = 4
CPCV_PATHS = 2
SWING_YEARS = 3
INTRADAY_DAYS = 504
INTRADAY_TOP_N = 30
N_SMOKE_SYMBOLS = 30

# Real Russell-1000 large-caps — liquid names that exist across the whole window
# for both the daily (swing) and 5-min + daily (intraday) paths. A small fixed
# subset keeps the run to a few minutes.
SMOKE_SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "HD", "DIS", "BAC", "XOM", "PFE", "KO", "PEP",
    "CSCO", "INTC", "VZ", "ADBE", "CMCSA", "NFLX", "T", "CRM", "ABT", "NKE",
    "MRK", "ORCL", "QCOM", "TXN", "COST", "AMD", "UNH", "CVX", "LLY", "WFC",
][:N_SMOKE_SYMBOLS]

# Log substrings that prove (or disprove) the invariants.
_FIT_LOG_NEEDLE = "per-fold-retrain: fit fresh"          # ≥1 fold trained a model
_NO_SAMPLES_NEEDLE = "no training samples"               # empty-matrix symptom
_SHALLOW_COVERAGE_NEEDLE = "Daily bars look shallow"     # PR #343 regression symptom


@dataclass
class CaptureHandler(logging.Handler):
    """Captures emitted log records so the smoke test can assert on the actual
    pipeline diagnostics (fit lines, 'no training samples', shallow-coverage warn)."""

    records: List[logging.LogRecord] = field(default_factory=list)

    def __post_init__(self):
        super().__init__(level=logging.INFO)

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        self.records.append(record)

    def messages(self) -> List[str]:
        out = []
        for r in self.records:
            try:
                out.append(r.getMessage())
            except Exception:
                out.append(str(r.msg))
        return out

    def fit_windows(self) -> List[str]:
        """Lines proving a fresh per-fold model was fit (includes trained_through=...)."""
        return [m for m in self.messages() if _FIT_LOG_NEEDLE in m]

    def shallow_coverage_fired(self) -> bool:
        return any(_SHALLOW_COVERAGE_NEEDLE in m for m in self.messages())


@dataclass
class SmokeResult:
    model: str
    passed: bool
    reasons: List[str] = field(default_factory=list)
    n_paths: int = 0
    n_skipped: int = 0
    mean_sharpe: float = 0.0
    is_true_wf: bool = False
    n_fit_windows: int = 0
    trained_through_match: bool = False
    daily_coverage_ok: Optional[bool] = None  # intraday only; None = n/a

    def fail(self, reason: str) -> None:
        self.passed = False
        self.reasons.append(reason)


def _build_args(model: str, as_of: str) -> argparse.Namespace:
    """Construct a fully-populated args Namespace matching what walkforward_tier3's
    CLI would produce for a `--per-fold-retrain --cpcv` run, with the MINIMAL smoke
    config. The CPCV helper functions read these attributes; we provide every one
    they access directly (plus the getattr-guarded extras) so the real production
    code path runs unchanged."""
    return argparse.Namespace(
        model=model,
        # CPCV shape
        cpcv=True,
        cpcv_k=CPCV_K,
        cpcv_paths=CPCV_PATHS,
        per_fold_retrain=True,
        # windows
        years=SWING_YEARS,
        days=INTRADAY_DAYS,
        intraday_top_n=INTRADAY_TOP_N,
        as_of=as_of,
        swing_train_years=None,
        # purge / embargo (production defaults)
        swing_purge_days=85,
        swing_embargo_days=None,
        intraday_purge_days=2,
        intraday_embargo_days=None,
        # costs
        swing_cost_bps=5.0,
        intraday_cost_bps=15.0,
        # ATR / targets
        stop_mult=0.5,
        target_mult=1.5,
        no_atr_stops=False,
        # PM abstention (off)
        pm_abstention_vix=0.0,
        pm_abstention_spy_ma_days=0,
        pm_abstention_spy_5d=False,
        # scoring / filters
        pm_opportunity_score=True,
        no_prefilters=True,
        dispersion_gate=True,
        intraday_multi_scan=False,
        # feature cache — thread executor + few workers keeps the small smoke run
        # light on this Windows box and avoids competing with a concurrent job.
        feature_cache_workers=2,
        feature_cache_executor="thread",
        feature_cache_disable=False,
        sim_scan_interval_days=1,
        # rebalance / shorts — all OFF (default long single-name swing path)
        rebalance_mode=False,
        # guards
        allow_sacred_holdout=False,
        allow_in_sample=False,
    )


def _run_one(model: str, as_of: str, dry_run: bool) -> SmokeResult:
    """Run one minimal real per-fold CPCV (swing or intraday) and evaluate the
    non-vacuous invariants. dry_run validates wiring without fetching data."""
    res = SmokeResult(model=model, passed=True)

    # Imports are LAZY (inside run, never at module import) so importing this file
    # — e.g. a future CI `--dry-run` or a `--help`/ast.parse check — triggers no
    # data fetch and no heavy strategy construction.
    from scripts import walkforward_tier3 as wf

    args = _build_args(model, as_of)
    swing_ver = None
    intraday_ver = None
    meta_model = None
    intraday_meta_model = None
    earnings_cal = None  # smoke run: skip the earnings-blackout pre-fetch (slow, optional)

    if dry_run:
        # Validate wiring only: confirm the helper + run_cpcv symbols import and the
        # args Namespace carries every attribute the helper reads directly. NO fetch.
        from scripts.walkforward.cpcv import run_cpcv  # noqa: F401
        helper = wf._run_cpcv_swing if model == "swing" else wf._run_cpcv_intraday
        assert callable(helper), f"{model} CPCV helper not callable"
        # Touch the attributes the helper reads so a missing one fails loudly here.
        _ = (args.cpcv_k, args.cpcv_paths, args.per_fold_retrain, args.as_of,
             args.swing_purge_days, args.intraday_purge_days, args.years, args.days,
             args.intraday_top_n, args.stop_mult, args.target_mult)
        res.reasons.append("dry-run: wiring OK (no data fetched)")
        return res

    # Attach a capture handler to the relevant pipeline loggers BEFORE the run so we
    # see the per-fold fit lines, any 'no training samples', and the shallow-coverage
    # warning. Root handler catches all; explicit names are belt-and-suspenders.
    cap = CaptureHandler()
    root = logging.getLogger()
    prev_level = root.level
    root.setLevel(logging.INFO)
    targets = [
        logging.getLogger(),
        logging.getLogger("scripts.walkforward.retrainers"),
        logging.getLogger("scripts.walkforward.cpcv"),
        logging.getLogger("scripts.walkforward.strategies.intraday"),
        logging.getLogger("app.ml.intraday_training"),
    ]
    for lg in targets:
        lg.addHandler(cap)

    try:
        if model == "swing":
            cpcv_result = wf._run_cpcv_swing(
                args, SMOKE_SYMBOLS, swing_ver, meta_model, earnings_cal, True
            )
        else:
            cpcv_result = wf._run_cpcv_intraday(
                args, SMOKE_SYMBOLS, intraday_ver, intraday_meta_model, earnings_cal, True
            )
    finally:
        for lg in targets:
            lg.removeHandler(cap)
        root.setLevel(prev_level)

    if cpcv_result is None:
        res.fail("run_cpcv returned None (no model loaded? _load_model failed) — "
                 "ensure a model artifact exists for this model_type")
        return res

    # ── Invariant 1: ≥1 surviving path (would have caught BOTH empty-matrix bugs) ──
    res.n_paths = int(cpcv_result.n_combinations)
    res.n_skipped = int(getattr(cpcv_result, "n_skipped", 0))
    res.mean_sharpe = float(cpcv_result.mean_sharpe)
    if res.n_paths <= 0:
        res.fail(f"n_combinations={res.n_paths} (expected > 0). ZERO surviving paths "
                 "is the exact signature of the per-fold empty-matrix bugs "
                 "(#339 swing regime-map, #342 intraday day-axis desync).")

    # ── Invariant 2: ≥1 fold actually fit a model (not ALL 'no training samples') ──
    fit_lines = cap.fit_windows()
    res.n_fit_windows = len(fit_lines)
    no_sample_lines = [m for m in cap.messages() if _NO_SAMPLES_NEEDLE in m]
    if res.n_fit_windows == 0:
        res.fail("no fold fit a model — 0 'per-fold-retrain: fit fresh ...' log lines. "
                 f"({len(no_sample_lines)} 'no training samples' messages seen) "
                 "Every fold produced an empty training matrix.")

    # ── Invariant 3: genuine per-fold OOS ──
    res.is_true_wf = bool(cpcv_result.is_true_walkforward)
    if not res.is_true_wf:
        res.fail("is_true_walkforward is False — this was a frozen-model run, not "
                 "the per-fold OOS path under test.")

    # ── Invariant 4: per-fold trained_through == fold tr_end for ≥1 fold ──
    # The fit log line embeds 'window [tr_start, tr_end]' and 'trained_through=tr_end';
    # the retrainer sets model.trained_through = tr_end, so a matching pair proves real
    # per-fold training. We confirm a fit line carries a trained_through= equal to the
    # window's upper bound.
    res.trained_through_match = _trained_through_matches_tr_end(fit_lines)
    if not res.trained_through_match:
        res.fail("no fold's trained_through matched its window tr_end — cannot prove "
                 "real per-fold training pinned the cutoff to the training-window end.")

    # ── Invariant 5 (intraday only): shallow daily-coverage warning did NOT fire ──
    if model == "intraday":
        fired = cap.shallow_coverage_fired()
        res.daily_coverage_ok = not fired
        if fired:
            res.fail("shallow daily-bar coverage WARNING fired — PR #343's yfinance "
                     "daily-feature fix appears regressed; 52w/vol features degraded.")

    return res


def _trained_through_matches_tr_end(fit_lines: List[str]) -> bool:
    """A fit log line reads:
        'per-fold-retrain: fit fresh <type> model for window [<tr_start>, <tr_end>]
         (N samples, M features, seed=S, trained_through=<tr_end>)'
    Return True if any line's trained_through value equals the window's upper bound."""
    import re
    pat = re.compile(
        r"window \[(?P<tr_start>[^,]+),\s*(?P<tr_end>[^\]]+)\].*trained_through=(?P<tt>\S+)"
    )
    for line in fit_lines:
        m = pat.search(line)
        if not m:
            continue
        tr_end = m.group("tr_end").strip()
        tt = m.group("tt").strip().rstrip(").,")
        # Compare on the date portion (handles 'YYYY-MM-DD' vs 'YYYY-MM-DD 00:00:00').
        if tr_end[:10] == tt[:10]:
            return True
    return False


def _print_summary(results: List[SmokeResult]) -> bool:
    all_pass = all(r.passed for r in results)
    print("\n" + "=" * 78)
    print("PER-FOLD REAL-DATA SMOKE TEST — SUMMARY")
    print("=" * 78)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        cov = ("n/a" if r.daily_coverage_ok is None
               else ("ok" if r.daily_coverage_ok else "SHALLOW!"))
        print(f"\n[{status}] {r.model}")
        print(f"    n_paths (surviving)      : {r.n_paths}")
        print(f"    n_skipped                : {r.n_skipped}")
        print(f"    mean Sharpe              : {r.mean_sharpe:.3f}")
        print(f"    is_true_walkforward      : {r.is_true_wf}")
        print(f"    folds that fit a model   : {r.n_fit_windows}")
        print(f"    trained_through==tr_end  : {r.trained_through_match}")
        print(f"    daily-coverage status    : {cov}")
        for reason in r.reasons:
            tag = "note" if r.passed else "FAIL"
            print(f"    - [{tag}] {reason}")
    print("\n" + "-" * 78)
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 78)
    return all_pass


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Real-data integration smoke test for the per-fold-retrain CPCV "
                    "pipeline (manual / nightly — NOT part of the pytest suite).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", choices=["swing", "intraday", "both"], default="both",
                        help="Which pipeline(s) to smoke-test (default: both).")
    parser.add_argument("--as-of", default=DEFAULT_AS_OF, metavar="YYYY-MM-DD",
                        help=f"Pin fold boundaries for reproducibility (default {DEFAULT_AS_OF}).")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Validate wiring (imports, helper, args) WITHOUT fetching "
                             "any data or training. Fast — safe for CI smoke wiring.")
    args = parser.parse_args(argv)

    models = ["swing", "intraday"] if args.model == "both" else [args.model]

    mode = "DRY-RUN (wiring only, no data)" if args.dry_run else "REAL DATA"
    print(f"per-fold smoke test — mode={mode}, models={models}, as_of={args.as_of}, "
          f"k={CPCV_K}, paths={CPCV_PATHS}, symbols={len(SMOKE_SYMBOLS)}")

    results: List[SmokeResult] = []
    for m in models:
        print(f"\n>>> running {m} ...")
        try:
            results.append(_run_one(m, args.as_of, args.dry_run))
        except Exception as exc:  # surface a crash as a FAIL, not a traceback-only exit
            r = SmokeResult(model=m, passed=False)
            r.fail(f"raised {type(exc).__name__}: {exc}")
            results.append(r)
            logging.getLogger(__name__).exception("smoke test for %s crashed", m)

    ok = _print_summary(results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
