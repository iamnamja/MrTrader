"""
Phase 80 — Intraday Bar-Offset Sensitivity Sweep.

Tests walk-forward performance for FEATURE_BARS values 9-15 around the
current baseline of 12 bars (1h of 5-min bars before entry signal).

Usage:
    python scripts/bar_sensitivity.py [--folds N] [--days D] [--symbols SYM ...]

Output: prints a summary table and appends results to docs/ML_EXPERIMENT_LOG.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

SWEEP_OFFSETS = [9, 10, 11, 12, 13, 14, 15]   # 12 = current baseline
DEFAULT_OFFSETS = SWEEP_OFFSETS                 # alias used by tests


def _run_one(offset: int, folds: int, total_days: int, symbols: Optional[List[str]]) -> dict:
    """Run an intraday walk-forward with a single scan_offsets=[offset] and return metrics."""
    from scripts.walkforward_tier3 import run_intraday_walkforward  # type: ignore

    report = run_intraday_walkforward(
        symbols=symbols,
        n_folds=folds,
        total_days=total_days,
        scan_offsets=[offset],
    )
    return {
        "offset": offset,
        "avg_sharpe": report.avg_sharpe,
        "min_sharpe": report.min_sharpe,
        "avg_win_rate": report.avg_win_rate,
        "total_trades": report.total_trades,
        "gate": report.gate_passed,
    }


def _print_table(rows: List[dict]) -> None:
    header = f"{'Offset':>8} {'Sharpe':>8} {'MinSharp':>9} {'WinRate':>8} {'Trades':>7} {'Gate':>5}"
    print("\n" + "=" * 55)
    print("Phase 80 — Bar Offset Sensitivity Sweep")
    print("=" * 55)
    print(header)
    print("-" * 55)
    for r in rows:
        gate_str = "PASS" if r["gate"] else "FAIL"
        marker = " <-- baseline" if r["offset"] == 12 else ""
        print(
            f"{r['offset']:>8}  {r['avg_sharpe']:>7.3f}  {r['min_sharpe']:>8.3f}  "
            f"{r['avg_win_rate']:>7.1%}  {r['total_trades']:>6}{marker:15}  {gate_str}"
        )
    print("=" * 55)


def _append_log(rows: List[dict], folds: int, total_days: int) -> None:
    log_path = REPO_ROOT / "docs" / "ML_EXPERIMENT_LOG.md"
    if not log_path.exists():
        return
    ts = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"\n### Phase 80 — Bar Sensitivity Sweep ({ts})\n",
        f"Folds: {folds}, Days: {total_days}\n\n",
        "| Offset | Avg Sharpe | Min Sharpe | Win Rate | Trades | Gate |\n",
        "|--------|-----------|-----------|----------|--------|------|\n",
    ]
    for r in rows:
        gate_str = "PASS" if r["gate"] else "FAIL"
        baseline = " (baseline)" if r["offset"] == 12 else ""
        lines.append(
            f"| {r['offset']}{baseline} | {r['avg_sharpe']:.3f} | {r['min_sharpe']:.3f} "
            f"| {r['avg_win_rate']:.1%} | {r['total_trades']} | {gate_str} |\n"
        )
    best = max(rows, key=lambda r: r["avg_sharpe"])
    lines.append(f"\nBest offset: **{best['offset']}** (Sharpe {best['avg_sharpe']:.3f})\n")
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.writelines(lines)
    print(f"Appended results to {log_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 80 bar-offset sensitivity sweep")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--offsets", type=int, nargs="+", default=SWEEP_OFFSETS)
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--no-log", action="store_true", help="Skip appending to ML_EXPERIMENT_LOG")
    args = ap.parse_args()

    results = []
    for offset in sorted(args.offsets):
        print(f"\nRunning offset={offset} ...")
        t0 = time.time()
        row = _run_one(offset, args.folds, args.days, args.symbols)
        row["elapsed"] = time.time() - t0
        results.append(row)
        print(f"  Done in {row['elapsed']:.1f}s — Sharpe {row['avg_sharpe']:.3f}")

    _print_table(results)
    if not args.no_log:
        _append_log(results, args.folds, args.days)


if __name__ == "__main__":
    main()
