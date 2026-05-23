"""
scripts/diag_model_rank_ic.py — Phase 1 Signal Diagnostics: Model Rank-IC.

Measures whether the composite model score (XGBRanker prediction) actually
correlates with subsequent forward returns. This is the primary "does the model
have alpha?" test — distinct from feature IC (which tests individual features).

Decision thresholds (Opus synthesis):
    rank-IC@20d >= 0.025, t-stat > 2.5  → signal exists, proceed to label engineering
    rank-IC@20d in [0.015, 0.025)       → weak signal, try binary labels + wider purge
    rank-IC@20d < 0.015                 → consider pivoting signal class entirely

Usage:
    python scripts/diag_model_rank_ic.py
    python scripts/diag_model_rank_ic.py --model-version 216
    python scripts/diag_model_rank_ic.py --start 2021-01-01 --end 2025-12-31
    python scripts/diag_model_rank_ic.py --horizons 5 10 20 --by-year
    python scripts/diag_model_rank_ic.py --max-symbols 200  # quick sanity check

Output:
    data/diagnostics/model_rank_ic/<timestamp>/rank_ic_summary.csv
    data/diagnostics/model_rank_ic/<timestamp>/rank_ic_daily.parquet
    data/diagnostics/model_rank_ic/<timestamp>/rank_ic_by_year.csv   (if --by-year)
    data/diagnostics/model_rank_ic/<timestamp>/report.md
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
for _noisy in ("botocore", "boto3", "urllib3", "s3transfer"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ── Thresholds (Opus synthesis recommendation) ────────────────────────────────
IC_STRONG   = 0.025   # rank-IC@20d: proceed to label engineering
IC_WEAK     = 0.015   # rank-IC@20d: weak, try label change
IC_DEAD     = 0.015   # rank-IC@20d: below this → consider pivot
TSTAT_MIN   = 2.5     # t-stat threshold for statistical significance
MIN_SYMS    = 30      # minimum symbols per day for a valid IC observation


def _load_model(version: int):
    """Load swing model by version number (same fallback path as walkforward_tier3)."""
    import pickle
    model_dir = ROOT / "app/ml/models"
    path = model_dir / f"swing_v{version}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if hasattr(obj, "is_trained"):
        logger.info("Loaded swing_v%d (pickle object)", version)
        return obj
    from app.ml.model import PortfolioSelectorModel
    m = PortfolioSelectorModel(model_type="xgboost")
    m.load(str(model_dir), version, model_name="swing")
    logger.info("Loaded swing_v%d via PortfolioSelectorModel", version)
    return m


def _load_bars(start: date, end: date, workers: int,
               max_symbols: Optional[int]) -> Dict[str, pd.DataFrame]:
    from app.data.universe_history import pit_union
    from app.data import get_provider

    symbols = pit_union("russell1000", start=start, end=end)
    if max_symbols:
        symbols = symbols[:max_symbols]
    logger.info("Fetching bars: %d symbols %s → %s", len(symbols), start, end)
    provider = get_provider("polygon")
    bars_map = provider.get_daily_bars_bulk(symbols, start=start, end=end)
    logger.info("Got bars for %d symbols", len(bars_map))
    return bars_map


def _build_forward_returns(bars_map: Dict[str, pd.DataFrame],
                            horizons: List[int]) -> pd.DataFrame:
    """MultiIndex (date, symbol) → fwd_Nd columns (simple returns)."""
    records = []
    max_h = max(horizons)
    for sym, df in bars_map.items():
        if df.empty:
            continue
        col = "close" if "close" in df.columns else "Close"
        if col not in df.columns:
            continue
        prices = df[col].sort_index()
        n = len(prices)
        for i, ts in enumerate(prices.index):
            day = ts.date() if hasattr(ts, "date") else ts
            row: dict = {"date": day, "symbol": sym}
            for h in horizons:
                if i + h < n:
                    fwd = float(prices.iloc[i + h] / prices.iloc[i] - 1.0)
                    row[f"fwd_{h}d"] = fwd
            records.append(row)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).set_index(["date", "symbol"])



def _build_score_panel_simple(
    bars_map: Dict[str, pd.DataFrame],
    model,
    start: date,
    end: date,
    workers: int,
    vix_history: Optional[pd.Series],
    macro_history: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Score panel — computes model scores for each (date, symbol)."""
    from app.backtesting.feature_cache import build_feature_cache

    # Use the model's own feature list for exact alignment
    if model.feature_names:
        feature_names = list(model.feature_names)
    else:
        from app.ml.retrain_config import PHASE_C_V2_FEATURE_KEEP_LIST
        feature_names = list(PHASE_C_V2_FEATURE_KEEP_LIST)

    trading_days: List[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            trading_days.append(d)
        d += timedelta(days=1)

    logger.info("Building feature cache: %d symbols × %d days, %d workers",
                len(bars_map), len(trading_days), workers)
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

    # Collect (day, symbol, feat_vec) triples
    by_day: Dict[date, list] = {}
    for sym, date_index in cache.date_index.items():
        for day, row_idx in date_index.items():
            feat_vec = cache.matrix[sym][row_idx]
            by_day.setdefault(day, []).append((sym, feat_vec))

    score_records = []
    days_done = 0
    for day in sorted(by_day.keys()):
        pairs = by_day[day]
        if len(pairs) < MIN_SYMS:
            continue
        syms_day = [p[0] for p in pairs]
        X = np.stack([p[1] for p in pairs]).astype(np.float32)
        np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        try:
            _, scores = model.predict(X)
        except Exception as e:
            logger.warning("Score failed on %s: %s", day, e)
            continue
        for sym, sc in zip(syms_day, scores):
            score_records.append({"date": day, "symbol": sym, "score": float(sc)})
        days_done += 1
        if days_done % 100 == 0:
            logger.info("Scored %d days (%s)", days_done, day)

    if not score_records:
        return pd.DataFrame()
    return pd.DataFrame(score_records).set_index(["date", "symbol"])


def _compute_rank_ic(score_panel: pd.DataFrame, fwd_returns: pd.DataFrame,
                     horizons: List[int]) -> pd.DataFrame:
    """Compute daily cross-sectional Spearman IC of model score vs forward return."""
    records = []
    dates = score_panel.index.get_level_values("date").unique()
    for day in dates:
        try:
            scores = score_panel.loc[day, "score"]
        except KeyError:
            continue
        if isinstance(scores, float):
            continue
        for h in horizons:
            col = f"fwd_{h}d"
            if col not in fwd_returns.columns:
                continue
            try:
                fwd = fwd_returns.loc[day, col]
            except KeyError:
                continue
            if isinstance(fwd, float):
                continue
            common = scores.index.intersection(fwd.index)
            if len(common) < MIN_SYMS:
                continue
            s_vals = scores.loc[common].values.astype(float)
            r_vals = fwd.loc[common].values.astype(float)
            mask = np.isfinite(s_vals) & np.isfinite(r_vals)
            if mask.sum() < MIN_SYMS:
                continue
            try:
                corr, _ = spearmanr(s_vals[mask], r_vals[mask])
                if np.isfinite(corr):
                    records.append({
                        "date": day,
                        "horizon": h,
                        "ic": float(corr),
                        "n_symbols": int(mask.sum()),
                    })
            except Exception:
                continue
    return pd.DataFrame(records)


def _summarize_rank_ic(daily_ic: pd.DataFrame,
                       horizons: List[int]) -> pd.DataFrame:
    """Aggregate daily IC to summary stats per horizon."""
    rows = []
    for h in horizons:
        sub = daily_ic.loc[daily_ic["horizon"] == h, "ic"]
        n = len(sub)
        if n == 0:
            continue
        mu = float(sub.mean())
        sigma = float(sub.std()) if n > 1 else 1.0
        t_stat = mu / (sigma / np.sqrt(n)) if sigma > 0 else 0.0
        ir = mu / sigma * np.sqrt(252) if sigma > 0 else 0.0
        hit = float((sub > 0).mean())
        rows.append({
            "horizon": h,
            "ic_mean": round(mu, 6),
            "ic_std": round(sigma, 6),
            "ic_ir": round(ir, 4),
            "t_stat": round(t_stat, 4),
            "hit_rate": round(hit, 4),
            "n_days": n,
        })
    return pd.DataFrame(rows).set_index("horizon")


def _summarize_by_year(daily_ic: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """IC breakdown by calendar year to detect temporal drift."""
    daily_ic = daily_ic.copy()
    daily_ic["year"] = pd.to_datetime(daily_ic["date"]).dt.year
    rows = []
    for (year, h), grp in daily_ic.groupby(["year", "horizon"]):
        sub = grp["ic"]
        n = len(sub)
        if n == 0:
            continue
        mu = float(sub.mean())
        sigma = float(sub.std()) if n > 1 else 1.0
        t_stat = mu / (sigma / np.sqrt(n)) if sigma > 0 else 0.0
        rows.append({"year": int(year), "horizon": int(h),
                     "ic_mean": round(mu, 6), "t_stat": round(t_stat, 4),
                     "n_days": n})
    return pd.DataFrame(rows)


def _verdict(summary: pd.DataFrame) -> str:
    """Apply decision tree from Opus synthesis."""
    if summary.empty:
        return "UNKNOWN — no IC data"
    h20 = summary.loc[20] if 20 in summary.index else None
    if h20 is None:
        h20 = summary.iloc[-1]
    ic = h20["ic_mean"]
    t = h20["t_stat"]
    if ic >= IC_STRONG and t >= TSTAT_MIN:
        return f"STRONG SIGNAL — rank-IC@20d={ic:.4f}, t={t:.2f}. Proceed to label engineering (Phase 2)."
    elif ic >= IC_WEAK:
        return f"WEAK SIGNAL — rank-IC@20d={ic:.4f}, t={t:.2f}. Try policy-realized binary labels + 60d purge."
    else:
        return f"NO SIGNAL — rank-IC@20d={ic:.4f}, t={t:.2f}. Consider pivoting feature set or signal class."


def _build_report(summary: pd.DataFrame, by_year: Optional[pd.DataFrame],
                  model_version: int, start: date, end: date) -> str:
    lines = [
        f"# Model Rank-IC Report — swing_v{model_version}",
        f"",
        f"**Period:** {start} → {end}",
        f"**Model:** swing_v{model_version} (XGBRanker, LambdaRank, 20d horizon, 19 features)",
        f"",
        f"## Summary by Horizon",
        f"",
        f"| Horizon | IC Mean | IC Std | IC IR | t-stat | Hit Rate | N Days |",
        f"|---------|---------|--------|-------|--------|----------|--------|",
    ]
    for h, row in summary.iterrows():
        lines.append(
            f"| {h}d | {row['ic_mean']:.4f} | {row['ic_std']:.4f} | "
            f"{row['ic_ir']:.3f} | {row['t_stat']:.2f} | "
            f"{row['hit_rate']:.3f} | {int(row['n_days'])} |"
        )

    lines += ["", f"## Verdict", "", _verdict(summary)]

    lines += [
        "",
        "## Decision Tree (Opus synthesis thresholds)",
        "",
        f"- rank-IC@20d ≥ {IC_STRONG}, t > {TSTAT_MIN}  → STRONG: proceed to label engineering",
        f"- rank-IC@20d ∈ [{IC_WEAK}, {IC_STRONG})      → WEAK: try binary labels + 60d purge",
        f"- rank-IC@20d < {IC_DEAD}                     → DEAD: pivot signal class",
    ]

    if by_year is not None and not by_year.empty:
        lines += ["", "## IC by Year (h=20d)", "",
                  "| Year | IC Mean | t-stat | N Days |",
                  "|------|---------|--------|--------|"]
        for _, row in by_year[by_year["horizon"] == 20].iterrows():
            lines.append(
                f"| {int(row['year'])} | {row['ic_mean']:.4f} | "
                f"{row['t_stat']:.2f} | {int(row['n_days'])} |"
            )

    return "\n".join(lines)


def main() -> None:
    from app.ml.retrain_config import _parse_sacred_holdout_start

    holdout = _parse_sacred_holdout_start()
    default_end = holdout - timedelta(days=1)
    default_start = date(2021, 1, 1)  # 4-year window by default

    ap = argparse.ArgumentParser(description="Model Rank-IC Diagnostic")
    ap.add_argument("--model-version", type=int, default=216)
    ap.add_argument("--start", default=str(default_start))
    ap.add_argument("--end", default=str(default_end))
    ap.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 20])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-symbols", type=int, default=None,
                    help="Limit symbols for quick sanity runs")
    ap.add_argument("--by-year", action="store_true")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    horizons = sorted(args.horizons)

    # Output directory
    ts = time.strftime("%Y%m%dT%H%M%S")
    out_dir = ROOT / f"data/diagnostics/model_rank_ic/{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Model Rank-IC Diagnostic ===")
    logger.info("Model: swing_v%d | Window: %s → %s | Horizons: %s",
                args.model_version, start, end, horizons)

    # Load model
    model = _load_model(args.model_version)

    # Load bars
    bars_map = _load_bars(start, end - timedelta(days=max(horizons)),
                          args.workers, args.max_symbols)
    if not bars_map:
        logger.error("No bar data loaded — exiting")
        return

    # Load macro history (includes VIX, HYG, IEF, RSP, SPY columns)
    vix_history: Optional[pd.Series] = None
    macro_history: Optional[pd.DataFrame] = None
    try:
        from app.data.macro_history import load_macro_history
        macro_history = load_macro_history()
        logger.info("Macro history loaded: %d days", len(macro_history))
        if "vix" in macro_history.columns:
            vix_history = macro_history["vix"].dropna()
            logger.info("VIX series extracted: %d days", len(vix_history))
    except Exception as e:
        logger.warning("Macro/VIX history unavailable: %s", e)

    # Build score panel
    score_panel = _build_score_panel_simple(
        bars_map, model, start, end, args.workers, vix_history, macro_history
    )
    if score_panel.empty:
        logger.error("Score panel empty — cannot compute IC")
        return
    logger.info("Score panel: %d (date, symbol) pairs", len(score_panel))

    # Build forward returns (need extra buffer days for longest horizon)
    logger.info("Building forward returns...")
    fwd_returns = _build_forward_returns(bars_map, horizons)
    logger.info("Forward returns: %d rows", len(fwd_returns))

    # Compute rank-IC
    logger.info("Computing rank-IC...")
    daily_ic = _compute_rank_ic(score_panel, fwd_returns, horizons)
    logger.info("Daily IC: %d rows", len(daily_ic))

    if daily_ic.empty:
        logger.error("No IC computed — check data alignment")
        return

    # Summarize
    summary = _summarize_rank_ic(daily_ic, horizons)
    by_year = _summarize_by_year(daily_ic, horizons) if args.by_year else None

    # Print summary
    print("\n" + "=" * 60)
    print(f"MODEL RANK-IC: swing_v{args.model_version}")
    print("=" * 60)
    print(summary.to_string())
    print()
    print("VERDICT:", _verdict(summary))
    print("=" * 60 + "\n")

    # Save outputs
    daily_ic.to_parquet(out_dir / "rank_ic_daily.parquet")
    summary.to_csv(out_dir / "rank_ic_summary.csv")
    if by_year is not None:
        by_year.to_csv(out_dir / "rank_ic_by_year.csv")

    report_md = _build_report(summary, by_year, args.model_version, start, end)
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    manifest = {
        "model_version": args.model_version,
        "start": str(start),
        "end": str(end),
        "horizons": horizons,
        "n_symbols": len(bars_map),
        "n_days_scored": int(daily_ic["date"].nunique()) if not daily_ic.empty else 0,
        "verdict": _verdict(summary),
        "timestamp": ts,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    logger.info("Outputs saved to %s", out_dir)
    logger.info("Report:\n%s", report_md)


if __name__ == "__main__":
    main()
