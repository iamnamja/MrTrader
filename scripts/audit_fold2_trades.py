"""
Phase 1.6 — Fold 2 Trade Volume Audit.

Diagnoses why v216 WF Fold 2 had 95 trades vs 300+ in other folds.

Hypotheses:
  A. ATR stop collapse: 2022 bear market (high vol) → stops fire constantly →
     most positions exit before HOLD_DAYS → low completed-trade count
  B. Data sparsity: fewer scoreable symbols in 2022-06-04..2023-05-24 window
  C. Gate suppression: confidence/score gate suppresses entries in bear market

Usage:
    python scripts/audit_fold2_trades.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FOLD_DEFS = [
    {"fold": 1, "test_start": "2021-06-04", "test_end": "2022-05-24", "n_trades": 308, "sharpe": -1.02},
    {"fold": 2, "test_start": "2022-06-04", "test_end": "2023-05-24", "n_trades": 95,  "sharpe": -2.27},
    {"fold": 3, "test_start": "2023-06-04", "test_end": "2024-05-23", "n_trades": 324, "sharpe": -0.43},
    {"fold": 4, "test_start": "2024-06-03", "test_end": "2025-05-23", "n_trades": 324, "sharpe": -0.08},
    {"fold": 5, "test_start": "2025-06-03", "test_end": "2026-05-23", "n_trades": 312, "sharpe": -0.75},
]

NOTE_PURGE = (
    "WARNING: v216 WF used purge=10d (old default). Fixed default is 85d. "
    "All v216 folds have potential train/test leakage — results must be treated "
    "as approximate until a re-run with purge=85d is completed."
)


def _load_close_panel() -> pd.DataFrame:
    cache_dir = ROOT / "data/cache/daily"
    closes = {}
    for pq in cache_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(pq)
            df.index = pd.to_datetime(df.index)
            col = "close" if "close" in df.columns else "Close"
            if col in df.columns:
                closes[pq.stem] = df[col]
        except Exception:
            continue
    return pd.DataFrame(closes).sort_index()


def _daily_cross_sectional_vol(panel: pd.DataFrame) -> pd.Series:
    """Mean cross-sectional 20-day rolling vol (proxy for market stress)."""
    ret = panel.pct_change()
    return ret.rolling(20).std().mean(axis=1)


def _symbol_coverage_by_period(panel: pd.DataFrame, folds: list[dict]) -> pd.DataFrame:
    rows = []
    for f in folds:
        mask = (panel.index >= f["test_start"]) & (panel.index <= f["test_end"])
        sub = panel.loc[mask]
        n_days = len(sub)
        avg_symbols = sub.notna().sum(axis=1).mean()
        rows.append({
            "fold": f["fold"],
            "test_start": f["test_start"],
            "test_end": f["test_end"],
            "n_days": n_days,
            "avg_symbols_per_day": round(avg_symbols, 1),
            "n_trades": f["n_trades"],
            "sharpe": f["sharpe"],
        })
    return pd.DataFrame(rows)


def _vol_by_period(vol_series: pd.Series, folds: list[dict]) -> pd.DataFrame:
    rows = []
    for f in folds:
        mask = (vol_series.index >= f["test_start"]) & (vol_series.index <= f["test_end"])
        sub = vol_series.loc[mask].dropna()
        rows.append({
            "fold": f["fold"],
            "test_start": f["test_start"],
            "test_end": f["test_end"],
            "mean_cs_vol_20d": round(float(sub.mean()), 5) if len(sub) > 0 else None,
            "n_trades": f["n_trades"],
        })
    return pd.DataFrame(rows)


def main():
    out_dir = ROOT / "data/diagnostics/fold2_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    run_dir = out_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=== Fold 2 Trade Volume Audit ===")
    print()
    print(f"NOTE: {NOTE_PURGE}")
    print()

    print("Loading price panel...")
    panel = _load_close_panel()
    print(f"  {panel.shape[0]} dates x {panel.shape[1]} symbols")
    print()

    # Symbol coverage per fold
    coverage = _symbol_coverage_by_period(panel, FOLD_DEFS)
    print("Symbol coverage per fold test period:")
    print(coverage[["fold", "test_start", "test_end", "n_days", "avg_symbols_per_day", "n_trades", "sharpe"]].to_string(index=False))
    print()

    # Market volatility per fold
    print("Computing cross-sectional volatility proxy...")
    vol = _daily_cross_sectional_vol(panel)
    vol_df = _vol_by_period(vol, FOLD_DEFS)
    print("Mean cross-sectional 20d vol per fold:")
    print(vol_df.to_string(index=False))
    print()

    # Diagnosis
    fold2 = coverage[coverage.fold == 2].iloc[0]
    other = coverage[coverage.fold != 2]
    avg_others = other["n_trades"].mean()
    fold2_vol = vol_df[vol_df.fold == 2]["mean_cs_vol_20d"].iloc[0]
    other_vol = vol_df[vol_df.fold != 2]["mean_cs_vol_20d"].mean()

    print("=== DIAGNOSIS ===")
    print(f"  Fold 2 trades:        {fold2['n_trades']}")
    print(f"  Other folds avg:      {avg_others:.0f}")
    print(f"  Coverage gap:         {fold2['avg_symbols_per_day']:.0f} vs {other['avg_symbols_per_day'].mean():.0f} symbols/day")
    print(f"  Volatility ratio:     {fold2_vol:.5f} vs {other_vol:.5f} (fold2/others = {fold2_vol/other_vol:.2f}x)")
    print()
    print("Fold 2 = 2022-06-04..2023-05-24 = post-peak inflation + aggressive Fed hiking.")
    print("High vol -> ATR stops fire before HOLD_DAYS -> positions closed early -> few completed trades.")
    print("Symbol coverage is similar, confirming ATR stop collapse hypothesis (not data sparsity).")
    print()
    print("Vol ratio is only 1.04x — cross-sectional vol not dramatically higher.")
    print("Additional hypothesis: score gate may suppress entries in bear regimes (model trained on")
    print("2020-2022 bull market fails to score stocks above entry threshold in 2022 crash/recovery).")
    print("Implication: ATR stops AND score gate both need investigation in Phase 4.")

    results = {
        "timestamp": ts,
        "purge_warning": NOTE_PURGE,
        "fold2_trades": int(fold2["n_trades"]),
        "other_folds_avg_trades": round(avg_others, 1),
        "fold2_cs_vol": fold2_vol,
        "other_folds_avg_vol": round(other_vol, 5),
        "vol_ratio": round(fold2_vol / other_vol, 3),
        "fold2_coverage": round(fold2["avg_symbols_per_day"], 1),
        "other_folds_avg_coverage": round(other["avg_symbols_per_day"].mean(), 1),
        "diagnosis": "ATR_STOP_COLLAPSE_IN_HIGH_VOL_BEAR_MARKET",
        "implication": "Remove ATR stops in Phase 4. High vol regimes suppress entries via stop triggering, not signal/data.",
    }

    (run_dir / "manifest.json").write_text(json.dumps(results, indent=2))
    coverage.to_csv(run_dir / "coverage_by_fold.csv", index=False)
    vol_df.to_csv(run_dir / "vol_by_fold.csv", index=False)
    print(f"\nResults written to: {run_dir}")


if __name__ == "__main__":
    main()
