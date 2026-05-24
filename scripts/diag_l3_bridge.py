"""
Phase 2.3 — L3 Bridge Test.

Bridges L2 (pure signal, no costs) and the full WF (signal + execution + costs).

Setup:
  - Long-only top-N (default 40) by model score
  - Equal-weight, hold HORIZON days, no stops
  - Realistic transaction costs (round-trip bps)
  - Rebalance every HORIZON days (no daily churn)

Pass criterion: L3 Sharpe >= 0.5 * L2 Sharpe
  (execution should preserve at least half the raw signal alpha)

If L3 fails: model has no alpha at realistic costs. Rebuild features.
If L3 passes: model has alpha. Proceed to v217 retrain with Phase 3 labels.

Usage:
    python scripts/diag_l3_bridge.py --model-version 216 \\
        --start 2021-01-01 --end 2025-12-31 --horizon 20 --top-n 40
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
SHARPE_L2 = 0.397   # from Phase 2.1 diag_decile_spread run
L3_PASS_RATIO = 0.5  # L3 must achieve >= 50% of L2 Sharpe


def _load_model(version: int):
    import pickle
    model_dir = ROOT / "app" / "ml" / "models"
    candidates = sorted(model_dir.glob(f"swing_v{version}*.pkl"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No swing model v{version} found in {model_dir}")
    path = candidates[-1]
    logger.info("Loading model: %s", path)
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
    vix_history,
    macro_history,
    horizon: int,
) -> pd.DataFrame:
    """Score every symbol on every rebalance date using the model."""
    from datetime import timedelta
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

    close_by_sym: Dict[str, pd.Series] = {}
    for sym, df in bars_map.items():
        col = "close" if "close" in df.columns else "Close"
        if col in df.columns:
            close_by_sym[sym] = df[col].sort_index()

    # collect all scored (day, sym, feature_vec) triples
    by_day: Dict[date, list] = {}
    for sym, date_index in cache.date_index.items():
        for day, row_idx in date_index.items():
            feat_vec = cache.matrix[sym][row_idx]
            by_day.setdefault(day, []).append((sym, feat_vec))

    all_dates = sorted(by_day.keys())
    rebalance_dates = all_dates[::horizon]
    logger.info("Scoring %d rebalance dates", len(rebalance_dates))

    records = []
    for day in rebalance_dates:
        pairs = by_day.get(day, [])
        if len(pairs) < 5:
            continue
        syms_day = [p[0] for p in pairs]
        X = np.stack([p[1] for p in pairs]).astype(np.float32)
        X = np.nan_to_num(X)
        _, scores = model.predict(X)

        day_ts = pd.Timestamp(day)
        for sym, score in zip(syms_day, scores):
            if sym not in close_by_sym:
                continue
            prices = close_by_sym[sym]
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


def _l3_simulate(df: pd.DataFrame, top_n: int, horizon: int, cost_bps: float) -> dict:
    """
    Simulate long-only top-N portfolio. Equal-weight, hold HORIZON days,
    round-trip cost applied at entry+exit.
    """
    round_trip_cost = cost_bps * 2 / 10_000  # entry + exit

    period_returns = []
    by_year: Dict[int, list] = {}

    for dt, grp in df.groupby("date"):
        if len(grp) < top_n:
            continue
        grp = grp.sort_values("score", ascending=False)
        top = grp.head(top_n)
        # Equal-weight gross return
        gross_ret = float(top["fwd_return"].mean())
        # Subtract round-trip costs (all positions turned over each period)
        net_ret = gross_ret - round_trip_cost
        period_returns.append(net_ret)
        yr = dt.year if hasattr(dt, "year") else pd.Timestamp(dt).year
        by_year.setdefault(yr, []).append(net_ret)

    if len(period_returns) < 5:
        return {"sharpe": 0.0, "n_periods": len(period_returns), "error": "too few periods"}

    arr = np.array(period_returns)
    ann_factor = np.sqrt(TRADING_DAYS_PER_YEAR / horizon)
    sharpe = float(arr.mean() / (arr.std() + 1e-10) * ann_factor)
    annual_ret = float(arr.mean() * (TRADING_DAYS_PER_YEAR / horizon))

    yr_sharpes = {}
    for yr, rets in sorted(by_year.items()):
        r = np.array(rets)
        yr_sharpes[str(yr)] = round(float(r.mean() / (r.std() + 1e-10) * ann_factor), 3)

    return {
        "sharpe": round(sharpe, 3),
        "annual_return_pct": round(annual_ret * 100, 2),
        "n_periods": len(period_returns),
        "mean_period_return": round(float(arr.mean()), 5),
        "std_period_return": round(float(arr.std()), 5),
        "by_year": yr_sharpes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", type=int, default=216)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--cost-bps", type=float, default=5.0,
                        help="One-way cost in bps (default 5bps = 0.05%); round-trip is 2x")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--out-dir", default="data/diagnostics/l3_bridge")
    args = parser.parse_args()

    from datetime import datetime as _dt

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.now().strftime("%Y%m%dT%H%M%S")
    run_dir = out_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== L3 Bridge Test - v{args.model_version} ===")
    print(f"  Long top-{args.top_n}, hold {args.horizon}d, cost {args.cost_bps}bps one-way")
    print(f"  Pass criterion: L3 Sharpe >= {L3_PASS_RATIO} * L2 Sharpe ({L3_PASS_RATIO * SHARPE_L2:.3f})")
    print()

    start_dt = date.fromisoformat(args.start)
    end_dt = date.fromisoformat(args.end)

    model = _load_model(args.model_version)

    max_sym = args.max_symbols if args.max_symbols > 0 else None
    bars = _load_bars(start_dt, end_dt, args.workers, max_symbols=max_sym)

    from app.data.macro_history import load_macro_history
    macro = load_macro_history()
    vix_history = macro.get("vix") if hasattr(macro, "get") else (
        macro["vix"] if "vix" in macro.columns else None
    )

    df = _build_score_panel(
        bars_map=bars,
        model=model,
        start=start_dt,
        end=end_dt,
        workers=args.workers,
        vix_history=vix_history,
        macro_history=macro,
        horizon=args.horizon,
    )
    logger.info("Score panel: %d rows, %d dates, %d symbols",
                len(df), df["date"].nunique(), df["symbol"].nunique())

    stats = _l3_simulate(df, args.top_n, args.horizon, args.cost_bps)

    l3_sharpe = stats["sharpe"]
    pass_threshold = L3_PASS_RATIO * SHARPE_L2
    passed = l3_sharpe >= pass_threshold

    print(f"\n=== L3 Results ===")
    print(f"  L3 Sharpe:       {l3_sharpe:.3f}")
    print(f"  L2 Sharpe:       {SHARPE_L2:.3f}")
    print(f"  Pass threshold:  {pass_threshold:.3f} (50% of L2)")
    print(f"  Result:          {'PASS' if passed else 'FAIL'}")
    print(f"  Annual return:   {stats.get('annual_return_pct', 0):.1f}%")
    print(f"  N periods:       {stats['n_periods']}")
    print()
    print("By year:")
    for yr, s in stats.get("by_year", {}).items():
        print(f"  {yr}: {s:+.3f}")

    if passed:
        print("\nL3 PASS: Model has alpha that survives realistic costs.")
        print("Proceed to v217 retrain with Phase 3 labels (10d horizon, rolling 3yr, long-only).")
        verdict = "PASS — proceed to Phase 3 retrain"
    else:
        print("\nL3 FAIL: Alpha insufficient at realistic costs.")
        print("Must revisit features/labels before retraining. Check by-year breakdown for regime-dependency.")
        verdict = "FAIL — revisit features/labels before retraining"

    results = {
        "timestamp": ts,
        "model_version": args.model_version,
        "start": args.start,
        "end": args.end,
        "horizon": args.horizon,
        "top_n": args.top_n,
        "cost_bps_one_way": args.cost_bps,
        "l2_sharpe": SHARPE_L2,
        "l3_sharpe": l3_sharpe,
        "pass_threshold": pass_threshold,
        "passed": passed,
        "verdict": verdict,
        **{k: v for k, v in stats.items() if k != "by_year"},
        "by_year": stats.get("by_year", {}),
    }
    (run_dir / "manifest.json").write_text(json.dumps(results, indent=2))
    df.to_parquet(run_dir / "scores_panel.parquet", index=False)
    print(f"\nResults written to: {run_dir}")


if __name__ == "__main__":
    main()
