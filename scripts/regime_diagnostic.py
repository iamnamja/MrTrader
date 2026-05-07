"""
Phase 5a lite — Regime Diagnostic: segment walk-forward trades by VIX/SPY regime bucket.

Runs swing v142 walk-forward with all Phase 1+2 corrections and segments every
simulated trade by VIX level, SPY trend, and 5d momentum. Reports per-bucket
win rate, avg R, and Sharpe so we know exactly which conditions the model works in.

Usage:
    python scripts/regime_diagnostic.py [--folds 3] [--output regime_diagnostic.csv]
"""
import argparse
import sys
from datetime import date
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.walkforward_tier3 import run_swing_walkforward, FoldResult


def _vix_bucket(vix: Optional[float]) -> str:
    if vix is None:
        return "unknown"
    if vix < 15:
        return "low (<15)"
    if vix < 20:
        return "normal (15-20)"
    if vix < 25:
        return "elevated (20-25)"
    if vix < 30:
        return "high (25-30)"
    return "extreme (>30)"


def _spy_trend_bucket(spy_above_ma20: Optional[bool]) -> str:
    if spy_above_ma20 is None:
        return "unknown"
    return "above_ma20" if spy_above_ma20 else "below_ma20"


def _spy_mom_bucket(spy_5d_ret: Optional[float]) -> str:
    if spy_5d_ret is None:
        return "unknown"
    if spy_5d_ret > 0.02:
        return "strong_up (>+2%)"
    if spy_5d_ret > 0.0:
        return "weak_up (0-2%)"
    if spy_5d_ret > -0.02:
        return "weak_down (-2-0%)"
    return "strong_down (<-2%)"


def run_diagnostic(n_folds: int = 3, output: Optional[str] = None):
    print("=" * 70)
    print("Phase 5a Lite — Regime Diagnostic (swing v142)")
    print("Walk-forward: 5yr, 3 folds, 5bps costs, 10d purge")
    print("=" * 70)

    # Run walk-forward and collect trade-level detail
    report = run_swing_walkforward(
        n_folds=n_folds,
        transaction_cost_pct=0.0005,  # 5bps
        purge_days=10,
        use_opportunity_score=True,
    )

    # Aggregate fold results
    all_folds: List[FoldResult] = report.folds
    print(f"\nFolds completed: {len(all_folds)}")
    for f in all_folds:
        print(f"  Fold {f.fold}: {f.test_start} → {f.test_end}  "
              f"trades={f.trades}  Sharpe={f.sharpe:+.2f}")

    avg_sharpe = np.mean([f.sharpe for f in all_folds])
    print(f"\nAvg Sharpe: {avg_sharpe:+.3f}")

    # If trade-level detail is not available, report fold-level only
    trade_records = getattr(report, "trade_detail", None)
    if not trade_records:
        print("\n[Note] Trade-level detail not available in this walk-forward run.")
        print("Fold-level results above are the diagnostic output.")
        print("\nFold 3 (most recent, 2025 tariff regime) Sharpe:", all_folds[-1].sharpe)
        print("Interpretation: fold 3 performance tells us how v142 fares in the")
        print("current tariff/vol regime. See ML_EXPERIMENT_LOG.md for full context.")
        return report

    # Segment trades by regime bucket
    rows = []
    for t in trade_records:
        rows.append({
            "date": t.get("date"),
            "symbol": t.get("symbol"),
            "pnl_r": t.get("pnl_r", 0.0),
            "win": int(t.get("pnl_r", 0.0) > 0),
            "vix": t.get("vix"),
            "spy_above_ma20": t.get("spy_above_ma20"),
            "spy_5d_ret": t.get("spy_5d_ret"),
            "vix_bucket": _vix_bucket(t.get("vix")),
            "spy_trend": _spy_trend_bucket(t.get("spy_above_ma20")),
            "spy_mom": _spy_mom_bucket(t.get("spy_5d_ret")),
        })
    df = pd.DataFrame(rows)

    def _report_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
        grp = df.groupby(col).agg(
            trades=("win", "count"),
            win_rate=("win", "mean"),
            avg_r=("pnl_r", "mean"),
            std_r=("pnl_r", "std"),
        ).reset_index()
        grp["sharpe"] = grp["avg_r"] / (grp["std_r"] + 1e-8) * np.sqrt(252)
        return grp.sort_values("sharpe", ascending=False)

    print("\n=== By VIX Bucket ===")
    print(_report_group(df, "vix_bucket").to_string(index=False))

    print("\n=== By SPY Trend (above/below MA20) ===")
    print(_report_group(df, "spy_trend").to_string(index=False))

    print("\n=== By SPY 5d Momentum ===")
    print(_report_group(df, "spy_mom").to_string(index=False))

    if output:
        df.to_csv(output, index=False)
        print(f"\nTrade-level detail saved to {output}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5a lite regime diagnostic")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--output", type=str, default=None,
                        help="CSV output path for trade-level detail")
    args = parser.parse_args()
    run_diagnostic(n_folds=args.folds, output=args.output)
