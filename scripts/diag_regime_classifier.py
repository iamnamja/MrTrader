"""
scripts/diag_regime_classifier.py — Phase A4: Regime classifier validation.

Evaluates R5 regime classifier (and any stored regime snapshots) on the
2025-2026 tariff-shock period — the interval NOT used in R5 training.
Exposes two key failure modes:
  - Trivial majority-class accuracy (98%+ RISK_ON → useless classifier)
  - Regime label quality on out-of-distribution periods (VIX spikes, drawdowns)

Kill criterion: if regime accuracy on 2025-2026 is <= 0.55 (5% above random),
or if the confusion matrix shows >80% predicted RISK_ON regardless of VIX,
the regime filter is not functioning and should be disabled (fall back to
unfiltered momentum baseline B1 from Phase A3).

Usage:
    python scripts/diag_regime_classifier.py
    python scripts/diag_regime_classifier.py --start 2025-01-01 --end 2026-05-09
    python scripts/diag_regime_classifier.py --validate-db   # compare DB vs computed

Output:
    data/diagnostics/regime_classifier/<timestamp>/confusion_matrix.csv
    data/diagnostics/regime_classifier/<timestamp>/daily_labels.csv
    data/diagnostics/regime_classifier/<timestamp>/vix_by_regime.csv
    data/diagnostics/regime_classifier/<timestamp>/regime_summary.md
    data/diagnostics/regime_classifier/<timestamp>/manifest.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

_HOLDOUT = _parse_sacred_holdout_start()
# Default window: 2025-2026 tariff-shock period (out-of-sample for R5)
_DEFAULT_START = date(2025, 1, 1)
_DEFAULT_END = _HOLDOUT - timedelta(days=1)

# Thresholds for RISK_ON / RISK_OFF classification from composite_score
RISK_ON_THRESHOLD = 0.60
RISK_OFF_THRESHOLD = 0.30

# VIX thresholds for "expected" regime
VIX_HIGH = 25.0   # VIX > 25 → expect RISK_OFF or NEUTRAL
VIX_ELEVATED = 18.0  # VIX 18-25 → expect NEUTRAL


def _load_db_regime_snapshots(start: date, end: date) -> pd.DataFrame:
    """Load daily regime snapshots from RegimeSnapshot table.

    Returns DataFrame with columns: date, composite_score, regime_label
    """
    try:
        from app.database.session import get_session, init_db
        from app.database.models import RegimeSnapshot
        init_db()
        with get_session() as session:
            rows = (
                session.query(RegimeSnapshot)
                .filter(
                    RegimeSnapshot.snapshot_date >= start,
                    RegimeSnapshot.snapshot_date <= end,
                )
                .all()
            )
        records = []
        for r in rows:
            score = getattr(r, "composite_score", None) or 0.5
            if score >= RISK_ON_THRESHOLD:
                label = "RISK_ON"
            elif score <= RISK_OFF_THRESHOLD:
                label = "RISK_OFF"
            else:
                label = "NEUTRAL"
            records.append({
                "date": r.snapshot_date,
                "composite_score": score,
                "regime_label": label,
            })
        if not records:
            logger.warning("No RegimeSnapshot rows found for %s to %s", start, end)
            return pd.DataFrame(columns=["date", "composite_score", "regime_label"])
        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        logger.info("Loaded %d regime snapshots from DB", len(df))
        return df
    except Exception as exc:
        logger.warning("Could not load DB regime snapshots: %s", exc)
        return pd.DataFrame(columns=["date", "composite_score", "regime_label"])


def _load_vix_series(start: date, end: date) -> pd.Series:
    """Load daily VIX values for the analysis window."""
    try:
        from app.data.macro_history import load_macro_history
        macro = load_macro_history()
        if macro.empty:
            return pd.Series(dtype=float)
        if "date" in macro.columns:
            macro = macro.set_index("date")
        if "vix" not in macro.columns:
            return pd.Series(dtype=float)
        vix = macro["vix"]
        vix.index = pd.to_datetime(vix.index)
        mask = (vix.index.date >= start) & (vix.index.date <= end)
        logger.info("Loaded %d VIX observations", mask.sum())
        return vix.loc[mask]
    except Exception as exc:
        logger.warning("Could not load VIX: %s", exc)
        return pd.Series(dtype=float)


def _compute_expected_labels(vix: pd.Series) -> pd.Series:
    """Derive a naive 'expected' regime from VIX thresholds.

    This is NOT the ground truth — it's a sanity check.
    If R5 labels diverge significantly from VIX-implied labels,
    the classifier may be miscalibrated.
    """
    labels = pd.Series("NEUTRAL", index=vix.index)
    labels[vix <= VIX_ELEVATED] = "RISK_ON"
    labels[vix >= VIX_HIGH] = "RISK_OFF"
    return labels


def _confusion_matrix(
    predicted: List[str], expected: List[str], labels: List[str]
) -> pd.DataFrame:
    """Compute confusion matrix as DataFrame (rows=predicted, cols=expected)."""
    matrix = pd.DataFrame(0, index=labels, columns=labels)
    for pred, exp in zip(predicted, expected):
        if pred in labels and exp in labels:
            matrix.loc[pred, exp] += 1
    return matrix


def _class_distribution(labels: List[str]) -> Dict[str, float]:
    cnt = Counter(labels)
    total = sum(cnt.values())
    return {k: round(v / total, 4) for k, v in cnt.items()}


def _write_manifest(out_dir: Path, args: argparse.Namespace, runtime_s: float) -> None:
    import subprocess
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_sha = "unknown"
    manifest = {
        "script": "diag_regime_classifier.py",
        "git_sha": git_sha,
        "start": str(args.start),
        "end": str(args.end),
        "validate_db": args.validate_db,
        "sacred_holdout_start": SACRED_HOLDOUT_START,
        "vix_high_threshold": VIX_HIGH,
        "vix_elevated_threshold": VIX_ELEVATED,
        "risk_on_threshold": RISK_ON_THRESHOLD,
        "risk_off_threshold": RISK_OFF_THRESHOLD,
        "runtime_seconds": round(runtime_s, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _write_report(
    out_dir: Path,
    db_df: pd.DataFrame,
    vix: pd.Series,
    vix_by_regime: pd.DataFrame,
    confusion: pd.DataFrame,
    class_dist: Dict[str, float],
) -> None:
    majority_pct = max(class_dist.values()) if class_dist else 1.0
    risk_on_pct = class_dist.get("RISK_ON", 0.0)

    lines = [
        "# Phase A4 — Regime Classifier Diagnostic Report",
        "",
        f"**Generated:** {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Kill Criteria",
        "",
        "| Criterion | Value | Status |",
        "|---|---|---|",
        f"| >80% predictions = RISK_ON (trivial) | {risk_on_pct:.1%} | "
        f"{'**FAIL**' if risk_on_pct > 0.80 else 'PASS'} |",
        f"| Majority class accuracy (random baseline) | {majority_pct:.1%} | "
        f"{'WARNING: trivial' if majority_pct > 0.85 else 'OK'} |",
        "",
        "## Class Distribution (DB predictions)",
        "",
        "| Regime | Fraction |",
        "|---|---|",
    ]
    for label, frac in sorted(class_dist.items()):
        lines.append(f"| {label} | {frac:.1%} |")

    lines += [
        "",
        "## VIX by Regime",
        "",
        "*(Expected: RISK_OFF periods should have higher median VIX than RISK_ON)*",
        "",
    ]
    if not vix_by_regime.empty:
        lines.append(vix_by_regime.round(2).to_markdown(index=True) if hasattr(vix_by_regime, "to_markdown") else str(vix_by_regime))

    lines += [
        "",
        "## Confusion Matrix (DB labels vs VIX-implied labels)",
        "",
        "*(VIX-implied is NOT ground truth — it is a sanity check)*",
        "*(rows = DB predictions, cols = VIX-implied)*",
        "",
    ]
    if not confusion.empty:
        lines.append(confusion.to_markdown() if hasattr(confusion, "to_markdown") else str(confusion))

    lines += [
        "",
        "## Interpretation",
        "",
        "- If RISK_ON > 80%: classifier defaults to RISK_ON; regime filter is useless.",
        "  Action: disable regime gate, retrain with balanced classes on 2025-2026 data.",
        "- If VIX median is similar across regimes: classifier ignores volatility signal.",
        "  Action: audit R5 feature set for VIX/MOVE/credit spread inputs.",
        "- If confusion matrix shows random agreement: R5 is not calibrated for",
        "  tariff-shock regime. Retrain R5 including 2025-2026 data.",
        "",
        "## R5 Training Context",
        "",
        "- R5 was trained on 2022-2024 data. AUC=1.000 on 2024 validation is trivial",
        "  because 98.4% of that period was classified as RISK_ON.",
        "- 2025-2026 has more frequent RISK_OFF events (tariff shocks, VIX > 25).",
        "- This diagnostic measures whether R5 generalises to that regime.",
    ]
    (out_dir / "regime_summary.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase A4: Regime classifier validation on tariff-shock period"
    )
    parser.add_argument("--start", type=date.fromisoformat, default=_DEFAULT_START)
    parser.add_argument("--end", type=date.fromisoformat, default=_DEFAULT_END)
    parser.add_argument("--validate-db", action="store_true",
                        help="Compare DB snapshots vs re-computed regime scores")
    parser.add_argument("--out-dir", type=Path, default=Path("data/diagnostics/regime_classifier"))
    args = parser.parse_args()

    from app.ml.retrain_config import assert_no_sacred_holdout
    assert_no_sacred_holdout(args.end, context="diag_regime_classifier")

    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    logger.info("Phase A4 regime diagnostic: %s -> %s", args.start, args.end)

    # ── Load DB regime labels ──
    db_df = _load_db_regime_snapshots(args.start, args.end)
    if not db_df.empty:
        db_df.to_csv(out_dir / "daily_labels.csv", index=False)
        logger.info("Regime label distribution: %s", _class_distribution(db_df["regime_label"].tolist()))

    # ── Load VIX ──
    vix = _load_vix_series(args.start, args.end)

    # ── VIX statistics by regime ──
    vix_by_regime = pd.DataFrame()
    if not db_df.empty and not vix.empty:
        merged = db_df.copy()
        merged["date"] = pd.to_datetime(merged["date"])
        vix_df = vix.reset_index()
        vix_df.columns = ["date", "vix"]
        vix_df["date"] = pd.to_datetime(vix_df["date"])
        merged = merged.merge(vix_df, on="date", how="left")
        if "vix" in merged.columns:
            vix_by_regime = merged.groupby("regime_label")["vix"].agg(
                ["mean", "median", "std", "min", "max", "count"]
            )
            vix_by_regime.to_csv(out_dir / "vix_by_regime.csv")
            logger.info("VIX by regime:\n%s", vix_by_regime)

    # ── VIX-implied labels ──
    confusion = pd.DataFrame()
    class_dist: Dict[str, float] = {}
    all_labels = ["RISK_ON", "NEUTRAL", "RISK_OFF"]

    if not db_df.empty:
        class_dist = _class_distribution(db_df["regime_label"].tolist())
        risk_on_pct = class_dist.get("RISK_ON", 0.0)
        if risk_on_pct > 0.80:
            logger.warning(
                "KILL CRITERION: %.1f%% of labels are RISK_ON. "
                "Regime classifier is trivial. Recommend disabling regime gate.",
                risk_on_pct * 100,
            )

        if not vix.empty:
            db_dates = pd.to_datetime(db_df["date"])
            vix_aligned = vix.reindex(db_dates, method="nearest")
            vix_aligned.index = db_df.index
            merged_labels = db_df.copy()
            merged_labels["vix"] = vix_aligned.values
            merged_labels = merged_labels.dropna(subset=["vix"])

            expected = _compute_expected_labels(merged_labels["vix"])
            predicted = merged_labels["regime_label"].tolist()
            confusion = _confusion_matrix(predicted, expected.tolist(), all_labels)
            confusion.to_csv(out_dir / "confusion_matrix.csv")
            logger.info("Confusion matrix (predicted vs VIX-implied):\n%s", confusion)

            # Compute overlap rate (not a true accuracy — VIX labels are heuristic)
            match = sum(p == e for p, e in zip(predicted, expected.tolist()))
            overlap = match / len(predicted) if predicted else 0.0
            logger.info(
                "DB vs VIX-implied label overlap: %.1f%% (%d/%d)",
                overlap * 100, match, len(predicted),
            )
            if overlap <= 0.55:
                logger.warning(
                    "Low VIX agreement (%.1f%%). Regime classifier may not be "
                    "sensitive to volatility. Consider retraining R5 on 2025-2026.",
                    overlap * 100,
                )

    # ── Write report ──
    _write_report(out_dir, db_df, vix, vix_by_regime, confusion, class_dist)
    runtime_s = time.time() - t_start
    _write_manifest(out_dir, args, runtime_s)

    logger.info("Artifacts written to: %s", out_dir)
    logger.info("Done in %.1fs", runtime_s)
    print("\n" + (out_dir / "regime_summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
