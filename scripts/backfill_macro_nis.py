"""
Phase 90 — Macro NIS Backfill

Derives per-day macro NIS context from existing per-symbol NewsSignalCache rows
and writes results to MacroSignalCache.  No LLM calls needed — this is a pure
aggregation of data we already have.

Macro features computed per trading day:
  macro_avg_direction     Mean direction_score across all symbols with NIS data
  macro_pct_bearish       Fraction of symbols with direction_score < -0.3
  macro_pct_bullish       Fraction of symbols with direction_score > +0.3
  macro_avg_materiality   Mean materiality_score (how market-moving is today's news)
  macro_pct_high_risk     Fraction of symbols with downside_risk_score > 0.7

These are stored in macro_signal_cache.events_payload as a JSON dict keyed
"macro_nis_features" so _get_macro_nis_features_pit() can retrieve them.

Usage:
  python scripts/backfill_macro_nis.py [--days 365] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("macro_nis_backfill")


def _trading_days(start: date, end: date) -> list[date]:
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            days.append(d)
        d += timedelta(days=1)
    return days


def _compute_macro_features(db, date_str: str) -> dict | None:
    """Aggregate per-symbol NIS rows for date_str into macro features."""
    from app.database.models import NewsSignalCache
    from sqlalchemy import func

    rows = (
        db.query(NewsSignalCache)
        .filter(NewsSignalCache.as_of_date == date_str)
        .all()
    )
    if not rows:
        return None

    directions = [r.direction_score for r in rows if r.direction_score is not None]
    materialities = [r.materiality_score for r in rows if r.materiality_score is not None]
    downside_risks = [r.downside_risk_score for r in rows if r.downside_risk_score is not None]

    if not directions:
        return None

    n = len(directions)
    avg_dir = sum(directions) / n
    pct_bearish = sum(1 for d in directions if d < -0.3) / n
    pct_bullish = sum(1 for d in directions if d > 0.3) / n
    avg_mat = sum(materialities) / len(materialities) if materialities else 0.0
    pct_high_risk = sum(1 for r in downside_risks if r > 0.7) / len(downside_risks) if downside_risks else 0.0

    return {
        "macro_avg_direction": round(avg_dir, 4),
        "macro_pct_bearish": round(pct_bearish, 4),
        "macro_pct_bullish": round(pct_bullish, 4),
        "macro_avg_materiality": round(avg_mat, 4),
        "macro_pct_high_risk": round(pct_high_risk, 4),
        "n_symbols": n,
    }


def _derive_risk_level(feats: dict) -> str:
    if feats["macro_pct_high_risk"] > 0.15 or feats["macro_avg_direction"] < -0.4:
        return "HIGH"
    if feats["macro_pct_bearish"] > 0.4 or feats["macro_avg_direction"] < -0.2:
        return "MEDIUM"
    return "LOW"


def _derive_direction(feats: dict) -> str:
    d = feats["macro_avg_direction"]
    if d > 0.15:
        return "BULLISH"
    if d < -0.15:
        return "BEARISH"
    return "NEUTRAL"


def _derive_sizing_factor(feats: dict) -> float:
    risk = _derive_risk_level(feats)
    if risk == "HIGH":
        return 0.5
    if risk == "MEDIUM":
        return 0.75
    return 1.0


def run(days: int = 365, dry_run: bool = False) -> None:
    from app.database.session import get_session
    from app.database.models import MacroSignalCache

    db = get_session()
    try:
        end = date.today()
        start = end - timedelta(days=days)
        trading_days = _trading_days(start, end)

        existing = {
            r.date for r in db.query(MacroSignalCache.date).all()
        }
        logger.info(
            "Backfill range: %s → %s (%d trading days, %d already exist)",
            start, end, len(trading_days), len(existing),
        )

        written = skipped = no_data = 0
        for d in trading_days:
            date_str = d.strftime("%Y-%m-%d")

            if date_str in existing:
                skipped += 1
                continue

            feats = _compute_macro_features(db, date_str)
            if feats is None:
                no_data += 1
                continue

            risk = _derive_risk_level(feats)
            direction = _derive_direction(feats)
            sizing = _derive_sizing_factor(feats)

            if dry_run:
                logger.info(
                    "[DRY-RUN] %s: risk=%s dir=%s sizing=%.2f n_symbols=%d avg_dir=%.3f",
                    date_str, risk, direction, sizing, feats["n_symbols"], feats["macro_avg_direction"],
                )
                written += 1
                continue

            row = MacroSignalCache(
                date=date_str,
                prompt_version="backfill_v1",
                risk_level=risk,
                direction=direction,
                sizing_factor=sizing,
                block_new_entries=(risk == "HIGH"),
                rationale=f"Derived from {feats['n_symbols']} symbol NIS scores",
                events_payload={"macro_nis_features": feats},
            )
            db.add(row)
            written += 1

            if written % 50 == 0:
                db.commit()
                logger.info("Progress: %d written, %d skipped, %d no-data", written, skipped, no_data)

        if not dry_run:
            db.commit()

        logger.info(
            "Done: %d written, %d skipped (already existed), %d dates had no NIS data",
            written, skipped, no_data,
        )
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill MacroSignalCache from NewsSignalCache aggregates")
    parser.add_argument("--days", type=int, default=365, help="Calendar days to backfill")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be written without writing")
    args = parser.parse_args()
    run(days=args.days, dry_run=args.dry_run)
