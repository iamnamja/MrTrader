"""
Phase 2 — L2 Decile Spread Backtest.

Isolation test: long top decile, short bottom decile, equal-weight,
daily rebalance, NO stops, NO costs, NO slippage.

This separates signal quality from execution quality. A high Sharpe here
with low WF Sharpe = execution/sizing problem. Low Sharpe here = no alpha.

Gate thresholds (Opus 4.7 plan):
  Sharpe >= 0.60 -> signal exists, problem is execution. Phase 4.
  Sharpe 0.20-0.60 -> marginal, proceed cautiously to Phase 3.
  Sharpe < 0.20 -> STOP. Features don't have alpha. Pivot to feature rebuild.

Usage:
    python scripts/diag_decile_spread.py --model-version 216 \\
        --start 2021-01-01 --end 2025-12-31 --horizon 20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Gate thresholds
SHARPE_STRONG = 0.60
SHARPE_MARGINAL = 0.20
MIN_SYMS_PER_DAY = 30
TRADING_DAYS_PER_YEAR = 252


def _load_model(version: int):
    import pickle
    model_dir = ROOT / "app/ml/models"
    path = model_dir / f"swing_v{version}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if hasattr(obj, "is_trained"):
        return obj
    from app.ml.model import PortfolioSelectorModel
    m = PortfolioSelectorModel(model_type="xgboost")
    m.load(str(model_dir), version, model_name="swing")
    return m


def _load_bars(start: date, end: date, workers: int,
               max_symbols: Optional[int]) -> Dict[str, pd.DataFrame]:
    from app.data.universe_history import pit_union
    from app.data import get_provider
    symbols = pit_union("russell1000", start=start, end=end)
    if max_symbols:
        symbols = symbols[:max_symbols]
    logger.info("Fetching bars: %d symbols %s -> %s", len(symbols), start, end)
    provider = get_provider("polygon")
    bars_map = provider.get_daily_bars_bulk(symbols, start=start, end=end)
    logger.info("Got bars for %d symbols", len(bars_map))
    return bars_map


def _build_score_panel(
    bars_map: Dict[str, pd.DataFrame],
    model,
    start: date,
    end: date,
    workers: int,
    vix_history: Optional[pd.Series],
    macro_history: Optional[pd.DataFrame],
    horizon: int,
) -> pd.DataFrame:
    """Build (date, symbol, score, fwd_return) panel."""
    from app.backtesting.feature_cache import build_feature_cache

    feature_names = list(model.feature_names) if model.feature_names else []
    if not feature_names:
        raise ValueError("Model has no feature_names")

    trading_days: List[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            trading_days.append(d)
        d += timedelta(days=1)

    logger.info("Building feature cache: %d symbols x %d days", len(bars_map), len(trading_days))
    t0 = time.time()
    cache = build_feature_cache(
        symbols_data=bars_map,
        trading_days=trading_days,
        feature_names=feature_names,
        vix_history=vix_history,
        macro_history=macro_history,
        workers=workers,
    )
    logger.info("Feature cache built in %.1fs", time.time() - t0)

    # Build close price panel for forward returns
    close_by_sym: Dict[str, pd.Series] = {}
    for sym, df in bars_map.items():
        col = "close" if "close" in df.columns else "Close"
        if col in df.columns:
            close_by_sym[sym] = df[col].sort_index()

    # Collect (date, symbol) feature vectors
    by_day: Dict[date, list] = {}
    for sym, date_index in cache.date_index.items():
        for day, row_idx in date_index.items():
            feat_vec = cache.matrix[sym][row_idx]
            by_day.setdefault(day, []).append((sym, feat_vec))

    records = []
    for day in sorted(by_day.keys()):
        pairs = by_day[day]
        if len(pairs) < MIN_SYMS_PER_DAY:
            continue

        syms_day = [p[0] for p in pairs]
        X = np.stack([p[1] for p in pairs]).astype(np.float32)
        X = np.nan_to_num(X)
        scores = model.predict(X)

        for sym, score in zip(syms_day, scores):
            if sym not in close_by_sym:
                continue
            prices = close_by_sym[sym]
            # Find bar at `day`
            day_ts = pd.Timestamp(day)
            if day_ts not in prices.index:
                continue
            loc = prices.index.get_loc(day_ts)
            if loc + horizon >= len(prices):
                continue
            fwd = float(prices.iloc[loc + horizon] / prices.iloc[loc] - 1.0)
            if not np.isfinite(fwd):
                continue
            records.append({"date": day, "symbol": sym, "score": float(score), "fwd_return": fwd})

    return pd.DataFrame(records)


def _decile_spread(df: pd.DataFrame, n_deciles: int = 10, horizon: int = 20) -> dict:
    """Compute L/S decile spread statistics."""
    daily_ls = []
    daily_long = []
    daily_short = []
    decile_perf: Dict[int, list] = {i: [] for i in range(1, n_deciles + 1)}

    for dt, grp in df.groupby("date"):
        if len(grp) < n_deciles * 2:
            continue
        grp = grp.copy()
        try:
            grp["decile"] = pd.qcut(grp["score"], n_deciles, labels=range(1, n_deciles + 1),
                                    duplicates="drop")
        except ValueError:
            continue
        grp["decile"] = grp["decile"].astype(int)

        top = grp[grp["decile"] == n_deciles]
        bottom = grp[grp["decile"] == 1]
        if len(top) == 0 or len(bottom) == 0:
            continue

        long_ret = float(top["fwd_return"].mean())
        short_ret = float(bottom["fwd_return"].mean())
        ls_ret = long_ret - short_ret

        daily_long.append(long_ret)
        daily_short.append(-short_ret)
        daily_ls.append(ls_ret)

        for d in range(1, n_deciles + 1):
            sub = grp[grp["decile"] == d]
            if len(sub) > 0:
                decile_perf[d].append(float(sub["fwd_return"].mean()))

    def _sharpe(returns: list) -> float:
        if len(returns) < 5:
            return 0.0
        arr = np.array(returns)
        std = arr.std()
        if std < 1e-10:
            return 0.0
        # Returns are horizon-day, scale annualization accordingly
        ann_factor = np.sqrt(TRADING_DAYS_PER_YEAR / horizon)
        return float(arr.mean() / std * ann_factor)

    ls_sharpe = _sharpe(daily_ls)
    decile_means = {d: float(np.mean(v)) if v else 0.0 for d, v in decile_perf.items()}
    means_list = [decile_means[i] for i in range(1, n_deciles + 1)]
    n_pairs = n_deciles - 1
    monotone = sum(1 for i in range(n_pairs) if means_list[i + 1] > means_list[i])

    return {
        "ls_sharpe": round(ls_sharpe, 3),
        "long_only_sharpe": round(_sharpe(daily_long), 3),
        "short_only_sharpe": round(_sharpe(daily_short), 3),
        "n_dates": len(daily_ls),
        "avg_ls_return_per_period": round(float(np.mean(daily_ls)) if daily_ls else 0.0, 6),
        "decile_means": {str(k): round(v, 6) for k, v in decile_means.items()},
        "monotonicity_fraction": round(monotone / n_pairs if n_pairs > 0 else 0.0, 3),
    }


def _verdict(ls_sharpe: float) -> str:
    if ls_sharpe >= SHARPE_STRONG:
        return (f"SIGNAL EXISTS — Sharpe={ls_sharpe:.2f} >= {SHARPE_STRONG}. "
                "Problem is execution/portfolio construction. Proceed to Phase 4.")
    elif ls_sharpe >= SHARPE_MARGINAL:
        return (f"MARGINAL SIGNAL — Sharpe={ls_sharpe:.2f} in [{SHARPE_MARGINAL}, {SHARPE_STRONG}). "
                "Proceed cautiously to Phase 3. Run factor attribution.")
    else:
        return (f"NO SIGNAL — Sharpe={ls_sharpe:.2f} < {SHARPE_MARGINAL}. "
                "STOP. Features don't have alpha. Pivot to feature rebuild.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", type=int, default=216)
    parser.add_argument("--start", type=str, default="2021-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--n-deciles", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="data/diagnostics/decile_spread")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    run_dir = out_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"L2 Decile Spread Backtest — v{args.model_version}")
    print(f"  Period: {args.start} to {args.end} | Horizon: {args.horizon}d | Deciles: {args.n_deciles}")

    start_dt = date.fromisoformat(args.start)
    end_dt = date.fromisoformat(args.end)

    model = _load_model(args.model_version)
    bars_map = _load_bars(start_dt, end_dt, args.workers, args.max_symbols)

    from app.data.macro_history import load_macro_history
    macro = load_macro_history()
    vix_history = macro.get("vix") if hasattr(macro, "get") else (
        macro["vix"] if "vix" in macro.columns else None
    )

    df = _build_score_panel(
        bars_map=bars_map,
        model=model,
        start=start_dt,
        end=end_dt,
        workers=args.workers,
        vix_history=vix_history,
        macro_history=macro,
        horizon=args.horizon,
    )

    if df.empty:
        print("ERROR: No scores built.")
        return 1

    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(run_dir / "scores_panel.parquet", index=False)
    print(f"Scores built: {len(df):,} (date, symbol) pairs")

    print("\nComputing decile spread...")
    perf = _decile_spread(df, n_deciles=args.n_deciles, horizon=args.horizon)
    verdict = _verdict(perf["ls_sharpe"])

    print(f"\n=== L2 Decile Spread Results ===")
    print(f"  L/S Sharpe:         {perf['ls_sharpe']:.3f}")
    print(f"  Long-only Sharpe:   {perf['long_only_sharpe']:.3f}")
    print(f"  Short-only Sharpe:  {perf['short_only_sharpe']:.3f}")
    print(f"  N periods scored:   {perf['n_dates']}")
    print(f"  Monotonicity:       {perf['monotonicity_fraction']:.0%}")
    print(f"\n  Decile mean returns ({args.horizon}d horizon):")
    for k in range(1, args.n_deciles + 1):
        v = perf["decile_means"].get(str(k), 0.0)
        bar = "#" * min(40, int(abs(v) * 1000))
        sign = "+" if v >= 0 else "-"
        print(f"    D{k:2d}: {sign}{abs(v):.4f}  {bar}")

    # By year
    print(f"\n=== By Year ===")
    by_year_records = []
    for yr in sorted(df["date"].dt.year.unique()):
        yr_df = df[df["date"].dt.year == yr]
        yr_perf = _decile_spread(yr_df, n_deciles=args.n_deciles, horizon=args.horizon)
        print(f"  {yr}: L/S Sharpe={yr_perf['ls_sharpe']:.3f}  n={yr_perf['n_dates']}")
        by_year_records.append({"year": yr, **yr_perf})
    pd.DataFrame(by_year_records).to_csv(run_dir / "by_year.csv", index=False)

    print(f"\n=== VERDICT ===\n  {verdict}")

    manifest = {
        "model_version": args.model_version,
        "start": args.start,
        "end": args.end,
        "horizon": args.horizon,
        "n_deciles": args.n_deciles,
        "timestamp": ts,
        "verdict": verdict,
        **perf,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.DataFrame([perf]).to_csv(run_dir / "summary.csv", index=False)
    print(f"\nResults written to: {run_dir}")

    return 0 if perf["ls_sharpe"] >= SHARPE_MARGINAL else 1


if __name__ == "__main__":
    sys.exit(main())
