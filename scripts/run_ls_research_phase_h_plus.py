"""
Phase H+ — Attribution and parameter-sensitivity walk-forward sweep.

Builds on Phase H scorers (A_QualityShort, B_MeanReversionShort, D_Combined).
Tests:
  1. Long-leg / short-leg attribution for A and B
  2. Parameter sensitivity sweeps (flags_required, max_shorts, quantiles)
  3. A+B union short leg (ABCombined)
  4. Analyst revision shorts (FMP analyst grades)
  5. PEAD with forced 5-day hold (via max_hold_bars_override)
  6. D_Combined parameter sweep

Results saved to docs/phase_h_plus_research_results.json.
Email update after each batch.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ml.retrain_config import MAX_THREADS as _max_threads  # noqa: E402
os.environ.setdefault("OMP_NUM_THREADS", str(_max_threads))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_max_threads))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [phaseHplus] %(message)s",
)
logger = logging.getLogger(__name__)

GATE = {"min_avg_sharpe": 0.80, "min_fold_sharpe": -0.30}
OUT_PATH = ROOT / "docs" / "phase_h_plus_research_results.json"


def _build_scorers():
    from app.ml.short_scorers import (
        QualityShortScorer,
        MeanReversionShortScorer,
        ABCombinedScorer,
        AnalystRevisionShortScorer,
        CombinedLSScorer,
    )
    from app.ml.pead_scorer import PEADScorer

    # Order: highest-priority research first. Each entry: (key, scorer, batch, extra_kwargs)
    return [
        # === BATCH 1: Attribution ===
        ("A1_QS_longs_only",
         QualityShortScorer(top_n=20, max_shorts=15, legs_mode="longs_only"),
         "attribution", {}),
        ("A2_QS_shorts_only",
         QualityShortScorer(top_n=20, max_shorts=15, legs_mode="shorts_only"),
         "attribution", {}),
        ("B1_MR_longs_only",
         MeanReversionShortScorer(top_n=20, max_shorts=15, legs_mode="longs_only"),
         "attribution", {}),
        ("B2_MR_shorts_only",
         MeanReversionShortScorer(top_n=20, max_shorts=15, legs_mode="shorts_only"),
         "attribution", {}),

        # === BATCH 2: QualityShort parameter sweep ===
        ("A3_QS_flags1_shorts10",
         QualityShortScorer(top_n=20, max_shorts=10, flags_required=1),
         "param_qs", {}),
        ("A4_QS_flags3_shorts20",
         QualityShortScorer(top_n=20, max_shorts=20, flags_required=3),
         "param_qs", {}),

        # === BATCH 3: MeanReversion parameter sweep ===
        ("B3_MR_aggressive",
         MeanReversionShortScorer(top_n=20, max_shorts=15,
                                  quantile_1m=0.75, quantile_20d=0.85),
         "param_mr", {}),
        ("B4_MR_selective",
         MeanReversionShortScorer(top_n=20, max_shorts=15,
                                  quantile_1m=0.85, quantile_20d=0.95),
         "param_mr", {}),

        # === BATCH 4: A+B union ===
        ("E_ABCombined",
         ABCombinedScorer(top_n=20, max_shorts=20),
         "ab_union", {}),

        # === BATCH 5: PEAD 5-day hold ===
        ("G_PEAD_hold5",
         PEADScorer(long_short=True),
         "pead_hold", {"max_hold_bars_override": 5}),

        # === BATCH 6: D_Combined parameter sweep ===
        ("D1_concentrated",
         CombinedLSScorer(top_n_long=10, max_shorts=8),
         "param_d", {}),
        ("D2_broad",
         CombinedLSScorer(top_n_long=20, max_shorts=15),
         "param_d", {}),

        # === BATCH 7: Analyst revision shorts ===
        ("F_AnalystRev",
         AnalystRevisionShortScorer(top_n=20, max_shorts=15, downgrade_threshold=-2.0),
         "analyst", {}),
    ]


def _run_one(name, scorer, extra_kwargs):
    from scripts.walkforward_tier3 import run_swing_walkforward

    logger.info("=" * 70)
    logger.info("Running scorer: %s  extras=%s", name, extra_kwargs)
    logger.info("=" * 70)
    t0 = time.time()
    wf = run_swing_walkforward(
        n_folds=5,
        total_years=6,
        use_opportunity_score=False,
        no_prefilters=True,
        use_factor_portfolio=False,
        scorer_instance=scorer,
        feature_cache_disable=True,
        **extra_kwargs,
    )
    elapsed = time.time() - t0
    folds = [
        {
            "fold": f.fold,
            "test_start": str(f.test_start),
            "test_end": str(f.test_end),
            "sharpe": float(f.sharpe),
            "trades": int(f.trades),
            "win_rate": float(f.win_rate),
            "max_dd": float(f.max_drawdown),
            "total_return": float(f.total_return),
        }
        for f in wf.folds
    ]
    avg_sh = float(wf.avg_sharpe)
    min_sh = float(wf.min_sharpe)
    fold2_sh = folds[1]["sharpe"] if len(folds) >= 2 else float("nan")
    gate_ok = avg_sh >= GATE["min_avg_sharpe"] and min_sh >= GATE["min_fold_sharpe"]
    verdict = "PASS" if gate_ok else "FAIL"
    logger.info(
        "[%s] %s — avg=%.3f min=%.3f fold2=%.3f elapsed=%.1fmin",
        name, verdict, avg_sh, min_sh, fold2_sh, elapsed / 60.0,
    )
    return {
        "name": name,
        "verdict": verdict,
        "avg_sharpe": avg_sh,
        "min_sharpe": min_sh,
        "fold2_sharpe": fold2_sh,
        "elapsed_min": elapsed / 60.0,
        "folds": folds,
    }


def _save(summary):
    OUT_PATH.parent.mkdir(exist_ok=True)
    with OUT_PATH.open("w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote %s (%d entries)", OUT_PATH, len(summary))


def _email_batch(subject_prefix, results, batch_label):
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from app.notifications.notifier import _smtp_send
    except Exception as exc:
        logger.warning("Email setup failed: %s", exc)
        return

    rows = []
    for r in results:
        fold_tbl = "".join(
            f"<tr><td>{f['fold']}</td><td>{f['test_start']}&rarr;{f['test_end']}</td>"
            f"<td style='text-align:right'>{f['sharpe']:.3f}</td>"
            f"<td style='text-align:right'>{f['trades']}</td>"
            f"<td style='text-align:right'>{f['win_rate']:.1%}</td>"
            f"<td style='text-align:right'>{f['total_return']:.1f}%</td></tr>"
            for f in r.get("folds", [])
        )
        color = "#0a0" if r.get("verdict") == "PASS" else ("#a00" if r.get("verdict") == "FAIL" else "#888")
        rows.append(f"""
<h3 style="color:{color}">{r['name']} — {r.get('verdict','?')}</h3>
<p>avg Sharpe = <b>{r.get('avg_sharpe', float('nan')):.3f}</b>
 | min fold = <b>{r.get('min_sharpe', float('nan')):.3f}</b>
 | Fold 2 = <b>{r.get('fold2_sharpe', float('nan')):.3f}</b>
 | elapsed = {r.get('elapsed_min', 0):.1f}min</p>
<table border="1" cellpadding="3" cellspacing="0" style="border-collapse:collapse">
<tr><th>Fold</th><th>Window</th><th>Sharpe</th><th>Trades</th><th>WinRate</th><th>Ret%</th></tr>
{fold_tbl}
</table>
""")
    body = f"<h2>Phase H+ — {batch_label} batch complete</h2>" + "\n".join(rows)
    _smtp_send(subject=f"{subject_prefix}: {batch_label}", html_body=body)
    logger.info("Email sent for batch %s", batch_label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Comma list of scorer name prefixes to run")
    parser.add_argument("--batches", type=str, default=None,
                        help="Comma list of batch labels to run "
                             "(attribution,param_qs,param_mr,ab_union,pead_hold,param_d,analyst)")
    args = parser.parse_args()

    all_scorers = _build_scorers()
    if args.only:
        keys = [k.strip() for k in args.only.split(",")]
        all_scorers = [t for t in all_scorers if any(t[0].startswith(k) for k in keys)]
    if args.batches:
        bs = {b.strip() for b in args.batches.split(",")}
        all_scorers = [t for t in all_scorers if t[2] in bs]
    logger.info("Running %d scorers: %s", len(all_scorers), [t[0] for t in all_scorers])

    # Load any prior partial results so we can resume / accumulate
    summary = []
    if OUT_PATH.exists():
        try:
            summary = json.load(OUT_PATH.open())
            logger.info("Loaded %d prior results from %s", len(summary), OUT_PATH)
        except Exception:
            summary = []
    done_names = {r["name"] for r in summary if r.get("verdict") in ("PASS", "FAIL")}

    current_batch = None
    batch_results = []
    for name, scorer, batch, extras in all_scorers:
        if name in done_names:
            logger.info("Skipping %s (already in results)", name)
            continue
        if current_batch is None:
            current_batch = batch
        if batch != current_batch:
            # Flush completed batch
            _email_batch("MrTrader Phase H+", batch_results, current_batch)
            batch_results = []
            current_batch = batch
        try:
            r = _run_one(name, scorer, extras)
        except Exception as exc:
            logger.exception("Scorer %s crashed: %s", name, exc)
            r = {"name": name, "verdict": "ERROR", "error": str(exc),
                 "avg_sharpe": float("nan"), "min_sharpe": float("nan"),
                 "fold2_sharpe": float("nan"), "elapsed_min": 0.0, "folds": []}
        summary.append(r)
        batch_results.append(r)
        _save(summary)

    if batch_results:
        _email_batch("MrTrader Phase H+", batch_results, current_batch or "final")

    logger.info("=" * 70)
    logger.info("ALL PHASE H+ RESULTS")
    logger.info("=" * 70)
    for r in summary:
        logger.info(
            "%-26s %-6s avg=%6.3f  min=%6.3f  fold2=%6.3f",
            r["name"], r.get("verdict", "?"),
            r.get("avg_sharpe", float("nan")),
            r.get("min_sharpe", float("nan")),
            r.get("fold2_sharpe", float("nan")),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
