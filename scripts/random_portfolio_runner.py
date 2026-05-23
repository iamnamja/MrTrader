"""
Phase 1 — Null Benchmark: Random Portfolio.

Runs N random portfolios (same constraints as WF) to establish a baseline.
The WF result must beat this baseline by >= 2 sigma to count as real alpha.

This answers: "is our Sharpe better than random chance with the same execution?"

Usage:
    python scripts/random_portfolio_runner.py --n-seeds 100 --start 2021-01-01 --end 2025-12-31
    python scripts/random_portfolio_runner.py --compare-sharpe -0.91  # compare vs v216 WF result
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRADING_DAYS_PER_YEAR = 252
N_POSITIONS = 40         # match WF portfolio size
HOLD_DAYS = 20           # match label horizon


def _load_price_panel(start: date, end: date) -> pd.DataFrame:
    """Load close prices for universe into a (date, symbol) DataFrame."""
    cache_dir = ROOT / "data/cache/daily"
    if not cache_dir.exists():
        cache_dir = ROOT / "data/cache/5min"

    closes: dict = {}
    for pq in cache_dir.glob("*.parquet"):
        sym = pq.stem
        try:
            df = pd.read_parquet(pq)
            df.index = pd.to_datetime(df.index)
            col = "close" if "close" in df.columns else "Close"
            if col not in df.columns:
                continue
            s = df[col].loc[str(start):str(end)]
            if len(s) > 60:
                closes[sym] = s
        except Exception:
            continue

    panel = pd.DataFrame(closes).sort_index()
    panel = panel.loc[str(start):str(end)]
    return panel


def _simulate_random_portfolio(
    close_panel: pd.DataFrame,
    n_positions: int,
    hold_days: int,
    rng: np.random.Generator,
) -> float:
    """
    Simulate one random portfolio: every hold_days, randomly pick n_positions
    equal-weight. No stops, no costs (matching L2 isolation assumptions).
    Returns annualized Sharpe.
    """
    syms = list(close_panel.columns)
    if len(syms) < n_positions:
        return 0.0

    fwd_returns = close_panel.pct_change(hold_days).shift(-hold_days)
    trading_dates = close_panel.index.tolist()
    rebalance_dates = trading_dates[::hold_days]

    period_returns = []
    for dt in rebalance_dates:
        if dt not in fwd_returns.index:
            continue
        row = fwd_returns.loc[dt].dropna()
        eligible = [s for s in row.index if np.isfinite(row[s])]
        if len(eligible) < n_positions:
            continue
        picks = rng.choice(eligible, size=n_positions, replace=False)
        ret = float(row[picks].mean())
        period_returns.append(ret)

    if len(period_returns) < 5:
        return 0.0
    arr = np.array(period_returns)
    std = arr.std()
    if std < 1e-10:
        return 0.0
    ann_factor = np.sqrt(TRADING_DAYS_PER_YEAR / hold_days)
    return float(arr.mean() / std * ann_factor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--start", type=str, default="2021-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--n-positions", type=int, default=N_POSITIONS)
    parser.add_argument("--hold-days", type=int, default=HOLD_DAYS)
    parser.add_argument("--compare-sharpe", type=float, default=None,
                        help="Compare this Sharpe value against null distribution")
    parser.add_argument("--out-dir", type=str, default="data/diagnostics/null_benchmark")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    run_dir = out_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    start_dt = date.fromisoformat(args.start)
    end_dt = date.fromisoformat(args.end)

    print(f"Loading price panel {args.start} to {args.end}...")
    panel = _load_price_panel(start_dt, end_dt)
    print(f"  {panel.shape[0]} dates x {panel.shape[1]} symbols")

    print(f"Running {args.n_seeds} random portfolios (n={args.n_positions}, hold={args.hold_days}d)...")
    sharpes = []
    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        s = _simulate_random_portfolio(panel, args.n_positions, args.hold_days, rng)
        sharpes.append(s)
        if seed % 10 == 9:
            print(f"  Seed {seed+1}/{args.n_seeds}: running mean={np.mean(sharpes):.3f}")

    sharpes = np.array(sharpes)
    null_mean = float(sharpes.mean())
    null_std = float(sharpes.std())
    null_p5 = float(np.percentile(sharpes, 5))
    null_p95 = float(np.percentile(sharpes, 95))
    null_max = float(sharpes.max())

    print(f"\n=== Null Benchmark ({args.n_seeds} seeds) ===")
    print(f"  Mean Sharpe:  {null_mean:.3f}")
    print(f"  Std:          {null_std:.3f}")
    print(f"  P5-P95:       [{null_p5:.3f}, {null_p95:.3f}]")
    print(f"  Max:          {null_max:.3f}")

    # Compare vs WF result
    if args.compare_sharpe is not None:
        z = (args.compare_sharpe - null_mean) / (null_std + 1e-10)
        pct_better = float((sharpes < args.compare_sharpe).mean())
        sig_2sigma = z > 2.0
        print(f"\n  Comparing WF Sharpe={args.compare_sharpe:.3f} vs null:")
        print(f"    z-score: {z:.2f} ({'PASS' if sig_2sigma else 'FAIL'} >= 2.0)")
        print(f"    Beats {pct_better:.0%} of random portfolios")
        print(f"    2-sigma gate: {'PASS' if sig_2sigma else 'FAIL'}")

    results = {
        "timestamp": ts,
        "n_seeds": args.n_seeds,
        "start": args.start,
        "end": args.end,
        "n_positions": args.n_positions,
        "hold_days": args.hold_days,
        "null_mean_sharpe": round(null_mean, 4),
        "null_std_sharpe": round(null_std, 4),
        "null_p5": round(null_p5, 4),
        "null_p95": round(null_p95, 4),
        "null_max": round(null_max, 4),
    }
    if args.compare_sharpe is not None:
        z = (args.compare_sharpe - null_mean) / (null_std + 1e-10)
        results["compare_sharpe"] = args.compare_sharpe
        results["z_score"] = round(float(z), 3)
        results["beats_null_2sigma"] = bool(z > 2.0)

    (run_dir / "manifest.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame({"sharpe": sharpes}).to_csv(run_dir / "null_distribution.csv", index=False)
    print(f"\nResults written to: {run_dir}")


if __name__ == "__main__":
    main()
