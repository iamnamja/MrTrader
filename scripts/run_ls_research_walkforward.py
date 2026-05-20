"""
Phase H — L/S Short Selection Research walk-forward.

Runs each candidate scorer through 5-fold / 6-yr walk-forward on Russell 1000,
prints fold-by-fold results, and writes a JSON summary to docs/.

Usage:
    python scripts/run_ls_research_walkforward.py [--only A,B,C,D]
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
    format="%(asctime)s %(levelname)s [ls_research] %(message)s",
)
logger = logging.getLogger(__name__)

GATE = {"min_avg_sharpe": 0.80, "min_fold_sharpe": -0.30}


def _build_scorers():
    from app.ml.short_scorers import (
        QualityShortScorer,
        MeanReversionShortScorer,
        SectorRelativeScorer,
        CombinedLSScorer,
    )
    return {
        "A_QualityShort": QualityShortScorer(top_n=20, max_shorts=15),
        "B_MeanReversionShort": MeanReversionShortScorer(top_n=20, max_shorts=15),
        "C_SectorRelative": SectorRelativeScorer(longs_per_sector=3, shorts_per_sector=3),
        "D_Combined": CombinedLSScorer(top_n_long=15, max_shorts=12),
    }


def _run_one(name, scorer):
    from scripts.walkforward_tier3 import run_swing_walkforward

    logger.info("=" * 70)
    logger.info("Running scorer: %s", name)
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


def _email_summary(summary: list):
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from app.notifications.notifier import _smtp_send
    except Exception as exc:
        logger.warning("Email setup failed: %s", exc)
        return

    rows = []
    for r in summary:
        fold_tbl = "".join(
            f"<tr><td>{f['fold']}</td><td>{f['test_start']}→{f['test_end']}</td>"
            f"<td style='text-align:right'>{f['sharpe']:.3f}</td>"
            f"<td style='text-align:right'>{f['trades']}</td>"
            f"<td style='text-align:right'>{f['win_rate']:.1%}</td>"
            f"<td style='text-align:right'>{f['total_return']:.1f}%</td></tr>"
            for f in r["folds"]
        )
        color = "#0a0" if r["verdict"] == "PASS" else "#a00"
        rows.append(f"""
<h3 style="color:{color}">{r['name']} — {r['verdict']}</h3>
<p>avg Sharpe = <b>{r['avg_sharpe']:.3f}</b> | min fold = <b>{r['min_sharpe']:.3f}</b>
   | Fold 2 (2022 bear) = <b>{r['fold2_sharpe']:.3f}</b>
   | elapsed = {r['elapsed_min']:.1f}min</p>
<table border="1" cellpadding="3" cellspacing="0" style="border-collapse:collapse">
<tr><th>Fold</th><th>Window</th><th>Sharpe</th><th>Trades</th><th>WinRate</th><th>Ret%</th></tr>
{fold_tbl}
</table>
""")

    passing = [r["name"] for r in summary if r["verdict"] == "PASS"]
    header = f"<h2>Phase H L/S Research — {len(passing)}/{len(summary)} passed</h2>"
    if passing:
        header += f"<p><b>Passed:</b> {', '.join(passing)}</p>"
    else:
        header += "<p><b>No scorer passed the gate.</b></p>"
    header += ("<p>Gate: avg Sharpe >= 0.80, min fold Sharpe >= -0.30. "
               "Critical test: Fold 2 (2022 bear market) must be >= -0.30.</p>")

    body = header + "\n".join(rows)
    _smtp_send(
        subject=f"MrTrader Phase H — L/S Research Results ({len(passing)}/{len(summary)} passed)",
        html_body=body,
    )
    logger.info("Email sent.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Comma list of scorer name prefixes to run (A,B,C,D)")
    args = parser.parse_args()

    scorers = _build_scorers()
    if args.only:
        keys = [k.strip().upper() for k in args.only.split(",")]
        scorers = {k: v for k, v in scorers.items() if any(k.startswith(p) for p in keys)}
        logger.info("Restricted to: %s", list(scorers.keys()))

    summary = []
    for name, scorer in scorers.items():
        try:
            summary.append(_run_one(name, scorer))
        except Exception as exc:
            logger.exception("Scorer %s crashed: %s", name, exc)
            summary.append({
                "name": name, "verdict": "ERROR", "error": str(exc),
                "avg_sharpe": float("nan"), "min_sharpe": float("nan"),
                "fold2_sharpe": float("nan"), "elapsed_min": 0.0, "folds": [],
            })

    # Persist JSON
    out_dir = ROOT / "docs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "phase_h_ls_research_results.json"
    with out_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote %s", out_path)

    # Console table
    logger.info("=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    for r in summary:
        logger.info(
            "%-22s %-6s avg=%6.3f  min=%6.3f  fold2=%6.3f",
            r["name"], r["verdict"], r["avg_sharpe"], r["min_sharpe"], r["fold2_sharpe"],
        )

    _email_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
