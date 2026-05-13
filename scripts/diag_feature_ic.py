"""
scripts/diag_feature_ic.py — Phase A1: Feature Information Coefficient diagnostic.

Computes cross-sectional Spearman IC for each feature vs. forward returns over
5/10/20-day horizons across a configurable historical window. Produces CSV and
markdown artifacts under data/diagnostics/feature_ic/<timestamp>/.

This is the primary "does signal exist?" diagnostic before any retraining.

Kill criterion: if max |IC_mean| < 0.015 across all features over 5 years,
the feature set has no exploitable signal at this cost level → Phase C.

Usage:
    python scripts/diag_feature_ic.py
    python scripts/diag_feature_ic.py --start 2020-01-01 --end 2026-05-09
    python scripts/diag_feature_ic.py --horizons 5 10 20 --workers 8
    python scripts/diag_feature_ic.py --regime-breakout   # per-regime IC table
    python scripts/diag_feature_ic.py --by-year           # yearly IC table

Output:
    data/diagnostics/feature_ic/<timestamp>/daily_ic.parquet
    data/diagnostics/feature_ic/<timestamp>/ic_summary.csv
    data/diagnostics/feature_ic/<timestamp>/ic_by_regime.csv  (if --regime-breakout)
    data/diagnostics/feature_ic/<timestamp>/ic_by_year.csv    (if --by-year)
    data/diagnostics/feature_ic/<timestamp>/top_features.md
    data/diagnostics/feature_ic/<timestamp>/manifest.json
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

from app.ml.retrain_config import (
    MAX_WORKERS,
    SACRED_HOLDOUT_START,
    _parse_sacred_holdout_start,
)

os.environ.setdefault("OMP_NUM_THREADS", str(MAX_WORKERS))

import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Safe default window: just before sacred holdout ───────────────────────────
_HOLDOUT = _parse_sacred_holdout_start()
_DEFAULT_END = _HOLDOUT - timedelta(days=1)
_DEFAULT_START = date(2019, 1, 1)


def _load_symbols_and_bars(
    start: date,
    end: date,
    workers: int,
    max_symbols: Optional[int] = None,
) -> Dict[str, "pd.DataFrame"]:
    """Load daily OHLCV bars for the Russell 1000 universe via Polygon S3.

    Returns {symbol: DataFrame} with DatetimeIndex.
    """
    from app.data.universe_history import get_russell1000_symbols
    from app.data.polygon_s3 import fetch_bulk_daily_bars

    symbols = get_russell1000_symbols()
    if max_symbols:
        symbols = symbols[:max_symbols]
    logger.info("Loading bars for %d symbols %s -> %s", len(symbols), start, end)
    bars_map = fetch_bulk_daily_bars(symbols, start_date=start, end_date=end, workers=workers)
    logger.info("Loaded bars for %d symbols", len(bars_map))
    return bars_map


def _build_feature_panel(
    bars_map: Dict[str, "pd.DataFrame"],
    start: date,
    end: date,
    feature_names: List[str],
    workers: int,
    vix_history: Optional["pd.Series"] = None,
    macro_history: Optional["pd.DataFrame"] = None,
) -> "pd.DataFrame":
    """Build MultiIndex (date, symbol) feature panel using FeatureEngineer.

    Reuses the same worker-based FeatureCache infrastructure as the WF pipeline
    to guarantee identical feature values (point-in-time correct).
    """
    from app.backtesting.feature_cache import build_feature_cache
    from app.ml.features import FeatureEngineer

    # Build list of trading days in [start, end]
    trading_days: List[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri (approximate; excludes holidays)
            trading_days.append(d)
        d += timedelta(days=1)

    logger.info("Building feature cache: %d trading days, %d workers", len(trading_days), workers)
    cache = build_feature_cache(
        symbols_data=bars_map,
        trading_days=trading_days,
        feature_names=feature_names,
        vix_history=vix_history,
        macro_history=macro_history,
        workers=workers,
    )

    # Unpack cache into MultiIndex DataFrame
    records = []
    for sym in cache.matrix:
        for day, row_idx in cache.date_index[sym].items():
            row = cache.matrix[sym][row_idx].tolist()
            records.append({"date": day, "symbol": sym, **dict(zip(feature_names, row))})

    if not records:
        logger.warning("Feature panel is empty — no features computed")
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index(["date", "symbol"])
    logger.info("Feature panel: %d rows, %d features", len(df), len(feature_names))
    return df


def _build_forward_returns(
    bars_map: Dict[str, "pd.DataFrame"],
    horizons: List[int],
    max_horizon: int,
) -> "pd.DataFrame":
    """Compute forward simple returns for each (symbol, date, horizon).

    Returns MultiIndex (date, symbol) DataFrame with columns fwd_Nd.
    """
    records = []
    for sym, df in bars_map.items():
        if df.empty:
            continue
        close_col = "close" if "close" in df.columns else "Close"
        if close_col not in df.columns:
            continue
        prices = df[close_col].sort_index()
        idx = prices.index
        for i, ts in enumerate(idx):
            day = ts.date() if hasattr(ts, "date") else ts
            row: Dict = {"date": day, "symbol": sym}
            for h in horizons:
                j = i + h
                if j < len(prices):
                    ret = float(prices.iloc[j] / prices.iloc[i] - 1.0)
                    row[f"fwd_{h}d"] = ret
            if len(row) > 2:  # at least one horizon computed
                records.append(row)

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).set_index(["date", "symbol"])
    logger.info("Forward returns: %d rows for %d horizons", len(df), len(horizons))
    return df


def _load_regime_labels(start: date, end: date) -> Dict[date, str]:
    """Load daily regime labels from RegimeSnapshot table (best-effort)."""
    try:
        from app.database.session import get_session, init_db
        from app.database.models import RegimeSnapshot
        init_db()
        with get_session() as session:
            snaps = (
                session.query(RegimeSnapshot)
                .filter(
                    RegimeSnapshot.snapshot_date >= start,
                    RegimeSnapshot.snapshot_date <= end,
                )
                .all()
            )
        labels: Dict[date, str] = {}
        for s in snaps:
            score = getattr(s, "composite_score", None) or 0.5
            if score >= 0.6:
                labels[s.snapshot_date] = "RISK_ON"
            elif score <= 0.3:
                labels[s.snapshot_date] = "RISK_OFF"
            else:
                labels[s.snapshot_date] = "NEUTRAL"
        logger.info("Loaded %d regime labels", len(labels))
        return labels
    except Exception as exc:
        logger.warning("Could not load regime labels: %s", exc)
        return {}


def _write_manifest(out_dir: Path, args: argparse.Namespace, runtime_s: float, feature_names: List[str]) -> None:
    import subprocess
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_sha = "unknown"
    manifest = {
        "script": "diag_feature_ic.py",
        "git_sha": git_sha,
        "start": str(args.start),
        "end": str(args.end),
        "horizons": list(args.horizons),
        "workers": args.workers,
        "min_symbols_per_day": args.min_symbols,
        "max_symbols": args.max_symbols,
        "sacred_holdout_start": SACRED_HOLDOUT_START,
        "n_features": len(feature_names),
        "feature_names_hash": hash(tuple(feature_names)),
        "runtime_seconds": round(runtime_s, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _write_report(out_dir: Path, agg: "pd.DataFrame", horizon: int, n_features_pass: int) -> None:
    from app.ml.diagnostics.ic import IC_IR_MIN, IC_MEAN_MIN, HIT_RATE_MIN, format_ic_markdown

    lines = [
        "# Phase A1 — Feature IC Diagnostic Report",
        "",
        f"**Generated:** {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Kill Criteria",
        "",
        f"- max |IC_mean| >= {IC_MEAN_MIN} (h={horizon}d): **{'PASS' if not agg.empty and agg.get(f'ic_mean_h{horizon}', pd.Series()).abs().max() >= IC_MEAN_MIN else 'FAIL'}**",
        f"- Features passing all thresholds: **{n_features_pass}** "
        f"(need >= 3 to confirm edge exists)",
        "",
        "## Interpretation",
        "",
        "- **|IC_mean| >= 0.02**: minimum signal floor for a 5bps-cost strategy",
        "- **|IC_IR| >= 0.5**: annualised risk-adjusted IC persistence",
        "- **hit_rate >= 0.53**: IC is consistently in the right direction",
        "",
        "If < 3 features pass all thresholds -> feature set has insufficient signal.",
        "Go to Phase C (re-architect: label, model, or strategy change).",
        "",
    ]
    if not agg.empty:
        lines.append(format_ic_markdown(agg, top_n=20, horizon=horizon))
    (out_dir / "top_features.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase A1: Feature IC diagnostic — measures cross-sectional signal in the feature set"
    )
    parser.add_argument("--start", type=date.fromisoformat, default=_DEFAULT_START)
    parser.add_argument("--end", type=date.fromisoformat, default=_DEFAULT_END)
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--min-symbols", type=int, default=30,
                        help="Min symbols per day to compute IC (default 30)")
    parser.add_argument("--max-symbols", type=int, default=None,
                        help="Cap universe size (for fast dev runs)")
    parser.add_argument("--regime-breakout", action="store_true",
                        help="Produce per-regime IC breakdown")
    parser.add_argument("--by-year", action="store_true",
                        help="Produce per-year IC breakdown")
    parser.add_argument("--out-dir", type=Path, default=Path("data/diagnostics/feature_ic"))
    args = parser.parse_args()

    # Sacred holdout guard
    from app.ml.retrain_config import assert_no_sacred_holdout
    assert_no_sacred_holdout(args.end, context="diag_feature_ic")

    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    logger.info("Phase A1 IC diagnostic: %s -> %s | horizons=%s", args.start, args.end, args.horizons)

    # ── Load active model's feature names ──
    try:
        from app.ml.model import load_active_model
        model = load_active_model(model_type="swing")
        feature_names = model.feature_names
        logger.info("Loaded active swing model feature names: %d features", len(feature_names))
    except Exception as exc:
        logger.warning("Could not load active model: %s — using fallback feature set", exc)
        # Fallback to the well-known v195 feature names (69 features post-R3 prune)
        from app.ml.training import _BASE_PRUNED  # noqa: SLF001
        from app.ml.features import FeatureEngineer
        fe = FeatureEngineer()
        # Get the full feature list (we'll need a sample bar to call engineer_features)
        feature_names = getattr(fe, "FEATURE_NAMES", None)
        if feature_names is None:
            logger.error("Cannot determine feature names — aborting")
            return 1

    # ── Load bars ──
    bars_map = _load_symbols_and_bars(args.start, args.end, args.workers, args.max_symbols)
    if not bars_map:
        logger.error("No bar data loaded — aborting")
        return 1

    # ── Load VIX + macro history for feature computation ──
    vix_history = None
    macro_history = None
    try:
        from app.data.macro_history import load_macro_history
        macro_history = load_macro_history()
        if "vix" in macro_history.columns:
            vix_history = macro_history.set_index("date")["vix"] if "date" in macro_history.columns else None
        logger.info("Loaded macro history: %d rows", len(macro_history))
    except Exception as exc:
        logger.warning("Could not load macro history: %s", exc)

    # ── Build feature panel ──
    feature_panel = _build_feature_panel(
        bars_map, args.start, args.end, feature_names, args.workers,
        vix_history=vix_history, macro_history=macro_history,
    )
    if feature_panel.empty:
        logger.error("Feature panel is empty — aborting")
        return 1

    # ── Build forward returns ──
    max_horizon = max(args.horizons)
    forward_returns = _build_forward_returns(bars_map, args.horizons, max_horizon)
    if forward_returns.empty:
        logger.error("Forward returns are empty — aborting")
        return 1

    # ── Compute IC ──
    from app.ml.diagnostics.ic import (
        aggregate_ic,
        compute_daily_ic,
        passes_ic_threshold,
        summarize_by_regime,
        summarize_by_year,
    )

    logger.info("Computing daily cross-sectional IC...")
    daily_ic = compute_daily_ic(
        feature_panel, forward_returns,
        horizons=args.horizons,
        min_symbols_per_day=args.min_symbols,
    )
    logger.info("Daily IC computed: %d rows", len(daily_ic))

    # ── Save raw data ──
    daily_ic.to_parquet(out_dir / "daily_ic.parquet")

    # ── Aggregate ──
    agg = aggregate_ic(daily_ic)
    if not agg.empty:
        agg.to_csv(out_dir / "ic_summary.csv")

    # ── Optional regime breakout ──
    if args.regime_breakout:
        regime_labels = _load_regime_labels(args.start, args.end)
        by_regime = summarize_by_regime(daily_ic, regime_labels)
        if not by_regime.empty:
            by_regime.to_csv(out_dir / "ic_by_regime.csv")
            logger.info("Regime IC table: %d rows", len(by_regime))

    # ── Optional year breakout ──
    if args.by_year:
        by_year = summarize_by_year(daily_ic)
        if not by_year.empty:
            by_year.to_csv(out_dir / "ic_by_year.csv")
            logger.info("Yearly IC table: %d rows", len(by_year))

    # ── Pass/fail assessment ──
    passing, failing = passes_ic_threshold(agg, horizon=args.horizons[0]) if not agg.empty else ([], [])
    logger.info("IC assessment (h=%d): %d features PASS, %d FAIL",
                args.horizons[0], len(passing), len(failing))
    if passing:
        logger.info("Passing features: %s", passing[:10])

    h_first = args.horizons[0]
    if not agg.empty and f"ic_mean_h{h_first}" in agg.columns:
        max_ic = agg[f"ic_mean_h{h_first}"].abs().max()
        logger.info("Max |IC_mean| (h=%dd): %.4f (kill criterion: < 0.015)", h_first, max_ic)
        if max_ic < 0.015:
            logger.warning(
                "KILL CRITERION HIT: max |IC_mean| = %.4f < 0.015. "
                "Feature set has insufficient signal at 5bps cost level. "
                "Consider Phase C (label/model/strategy pivot).",
                max_ic,
            )
        elif len(passing) >= 3:
            logger.info(
                "Signal confirmed: %d features pass IC thresholds. "
                "Proceed with Phase B (label/architecture changes).",
                len(passing),
            )

    # ── Write report and manifest ──
    _write_report(out_dir, agg, h_first, len(passing))
    runtime_s = time.time() - t_start
    _write_manifest(out_dir, args, runtime_s, feature_names)

    logger.info("Artifacts written to: %s", out_dir)
    logger.info("Done in %.1fs", runtime_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
