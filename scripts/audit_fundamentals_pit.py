"""
Phase 1 — PIT Audit of FMP Fundamentals.

Verifies that the fmp_fundamentals_history.parquet uses true filing dates
(FMP filingDate) as `as_of_date`, not fiscal period-end dates (which would
be look-ahead).

Checks:
  1. Lag distribution: as_of_date - period_end. Median should be 30-60 days.
  2. Negative lag: any row where as_of_date < period_end is look-ahead contamination.
  3. Zero-lag rows: as_of_date == period_end is suspicious (filing on last day of period).
  4. Coverage: fraction of symbols in training universe covered by FMP vs EDGAR.
  5. EDGAR path check: whether EDGAR fundamentals_history uses period_end (unsafe).

Usage:
    python scripts/audit_fundamentals_pit.py [--min-lag 45] [--out-dir data/diagnostics]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

FMP_PATH = REPO_ROOT / "data/fundamentals/fmp_fundamentals_history.parquet"
EDGAR_PATH = REPO_ROOT / "data/fundamentals/fundamentals_history.parquet"


def _audit_fmp(df: pd.DataFrame, min_lag: int) -> dict:
    df = df.copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["lag_days"] = (df["as_of_date"] - df["period_end"]).dt.days

    total = len(df)
    neg_lag = (df["lag_days"] < 0).sum()
    zero_lag = (df["lag_days"] == 0).sum()
    short_lag = (df["lag_days"] < min_lag).sum()
    median_lag = float(df["lag_days"].median())
    mean_lag = float(df["lag_days"].mean())
    p25 = float(df["lag_days"].quantile(0.25))
    p75 = float(df["lag_days"].quantile(0.75))

    by_year = (
        df.assign(year=df["as_of_date"].dt.year)
        .groupby("year")["lag_days"]
        .agg(["median", "mean", "min", "max", "count"])
        .rename(columns={"median": "median_lag", "mean": "mean_lag",
                         "min": "min_lag", "max": "max_lag"})
        .reset_index()
    )

    n_symbols = df["symbol"].nunique()

    verdict = "PASS"
    issues = []
    if neg_lag > 0:
        issues.append(f"{neg_lag} rows with negative lag (look-ahead!)")
        verdict = "FAIL"
    if median_lag < min_lag:
        issues.append(
            f"Median lag {median_lag:.0f}d < minimum {min_lag}d. "
            f"{short_lag}/{total} rows ({100*short_lag/total:.1f}%) below threshold."
        )
        # This is a WARNING, not necessarily FAIL if as_of_date is truly the filing date.
        # Real 10-Q filings can be available in 30-40 days for large filers.
        if verdict == "PASS":
            verdict = "WARN"

    return {
        "verdict": verdict,
        "issues": issues,
        "total_rows": total,
        "n_symbols": n_symbols,
        "neg_lag_rows": int(neg_lag),
        "zero_lag_rows": int(zero_lag),
        "rows_below_min_lag": int(short_lag),
        "pct_below_min_lag": round(100 * short_lag / total, 1),
        "median_lag_days": round(median_lag, 1),
        "mean_lag_days": round(mean_lag, 1),
        "p25_lag_days": round(p25, 1),
        "p75_lag_days": round(p75, 1),
        "by_year": by_year,
    }


def _audit_edgar(df: pd.DataFrame, min_lag: int) -> dict:
    """EDGAR path uses fiscal year end as as_of_date — no filing lag baked in."""
    df = df.copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    n_symbols = df["symbol"].nunique()
    # EDGAR doesn't store period_end separately; check month distribution
    month_counts = df["as_of_date"].dt.month.value_counts().sort_index().to_dict()
    dec_pct = 100 * df[df["as_of_date"].dt.month == 12].shape[0] / len(df)
    return {
        "verdict": "WARN",
        "issues": [
            f"EDGAR as_of_date is fiscal year end (December-heavy: {dec_pct:.0f}% of rows). "
            "No filing-date lag baked in. EDGAR path is NOT PIT-safe on its own."
        ],
        "total_rows": len(df),
        "n_symbols": n_symbols,
        "december_pct": round(dec_pct, 1),
        "month_distribution": month_counts,
    }


def _check_fmp_coverage(fmp_df: pd.DataFrame, edgar_df: pd.DataFrame) -> dict:
    """How much of the training universe has FMP coverage vs EDGAR-only?"""
    try:
        from app.utils.constants import RUSSELL_1000_TICKERS
        universe = set(RUSSELL_1000_TICKERS)
    except Exception:
        universe = set()

    fmp_syms = set(fmp_df["symbol"].unique())
    edgar_syms = set(edgar_df["symbol"].unique()) if edgar_df is not None else set()

    fmp_only = fmp_syms - edgar_syms
    both = fmp_syms & edgar_syms
    edgar_only = edgar_syms - fmp_syms

    if universe:
        in_universe_fmp = len(fmp_syms & universe)
        in_universe_edgar = len(edgar_syms & universe) if edgar_syms else 0
        not_covered = len(universe - (fmp_syms | edgar_syms))
    else:
        in_universe_fmp = in_universe_edgar = not_covered = 0

    return {
        "fmp_symbols": len(fmp_syms),
        "edgar_symbols": len(edgar_syms),
        "fmp_only": len(fmp_only),
        "both": len(both),
        "edgar_only": len(edgar_only),
        "universe_size": len(universe),
        "universe_covered_by_fmp": in_universe_fmp,
        "universe_covered_by_edgar_only": in_universe_edgar,
        "universe_not_covered": not_covered,
        "note": "FMP overrides EDGAR where both present (see features.py:527-549). "
                "EDGAR is used as fallback for symbols not in FMP. "
                "EDGAR path is NOT PIT-safe; only FMP path has correct filing-date lag.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-lag", type=int, default=30,
                        help="Minimum acceptable lag (days). Default=30 (SEC 10-Q deadline for large filers).")
    parser.add_argument("--out-dir", type=str, default="data/diagnostics/pit_audit")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    run_dir = out_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    results = {"timestamp": ts, "min_lag_threshold": args.min_lag}

    # ── FMP audit ─────────────────────────────────────────────────────────────
    if FMP_PATH.exists():
        fmp_df = pd.read_parquet(FMP_PATH)
        fmp_result = _audit_fmp(fmp_df, args.min_lag)
        fmp_by_year = fmp_result.pop("by_year")
        fmp_by_year.to_csv(run_dir / "fmp_lag_by_year.csv", index=False)
        results["fmp"] = fmp_result
        print("=== FMP Fundamentals Audit ===")
        print(f"  Verdict: {fmp_result['verdict']}")
        print(f"  Total rows: {fmp_result['total_rows']:,} | Symbols: {fmp_result['n_symbols']}")
        print(f"  Lag: median={fmp_result['median_lag_days']}d  p25={fmp_result['p25_lag_days']}d  p75={fmp_result['p75_lag_days']}d")
        print(f"  Negative lag (look-ahead): {fmp_result['neg_lag_rows']} rows")
        print(f"  Zero-lag rows: {fmp_result['zero_lag_rows']} rows")
        print(f"  Rows < {args.min_lag}d lag: {fmp_result['rows_below_min_lag']:,} ({fmp_result['pct_below_min_lag']}%)")
        for issue in fmp_result["issues"]:
            print(f"  WARN:  {issue}")
    else:
        print("FMP parquet not found — skipping FMP audit")
        fmp_df = pd.DataFrame()
        results["fmp"] = {"verdict": "SKIP", "issues": ["FMP parquet not found"]}

    # ── EDGAR audit ───────────────────────────────────────────────────────────
    if EDGAR_PATH.exists():
        edgar_df = pd.read_parquet(EDGAR_PATH)
        edgar_result = _audit_edgar(edgar_df, args.min_lag)
        results["edgar"] = edgar_result
        print("\n=== EDGAR Fundamentals Audit ===")
        print(f"  Verdict: {edgar_result['verdict']}")
        print(f"  Total rows: {edgar_result['total_rows']:,} | Symbols: {edgar_result['n_symbols']}")
        for issue in edgar_result["issues"]:
            print(f"  WARN:  {issue}")
    else:
        print("EDGAR parquet not found — skipping EDGAR audit")
        edgar_df = None
        results["edgar"] = {"verdict": "SKIP", "issues": ["EDGAR parquet not found"]}

    # ── Coverage audit ────────────────────────────────────────────────────────
    cov = _check_fmp_coverage(fmp_df, edgar_df)
    results["coverage"] = cov
    print("\n=== Coverage ===")
    print(f"  FMP symbols: {cov['fmp_symbols']} | EDGAR symbols: {cov['edgar_symbols']}")
    if cov["universe_size"]:
        print(f"  Training universe ({cov['universe_size']} tickers):")
        print(f"    FMP covered: {cov['universe_covered_by_fmp']}")
        print(f"    Not covered: {cov['universe_not_covered']}")

    # ── Verdict ────────────────────────────────────────────────────────────────
    fmp_v = results["fmp"]["verdict"]
    edgar_v = results["edgar"]["verdict"]

    if fmp_v == "FAIL":
        overall = "FAIL — FMP has look-ahead. All training results are contaminated."
    elif fmp_v == "WARN" and cov["universe_not_covered"] > 50:
        overall = "WARN — FMP lag is shorter than min_lag threshold for most rows. " \
                  "EDGAR fallback is unsafe. Significant universe not covered by FMP."
    elif fmp_v == "PASS":
        overall = "PASS — FMP is PIT-safe. EDGAR fallback is unsafe but overridden by FMP for covered symbols."
    else:
        overall = f"WARN — FMP verdict={fmp_v}. EDGAR is not PIT-safe."

    results["overall_verdict"] = overall
    print(f"\n=== OVERALL VERDICT ===\n  {overall}")

    manifest = {
        "timestamp": ts,
        "min_lag_threshold": args.min_lag,
        "fmp_verdict": fmp_v,
        "edgar_verdict": edgar_v,
        "overall_verdict": overall,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "full_report.json").write_text(
        json.dumps({k: v for k, v in results.items() if k != "coverage" or True}, indent=2, default=str)
    )

    print(f"\nResults written to: {run_dir}")
    return 0 if fmp_v != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
