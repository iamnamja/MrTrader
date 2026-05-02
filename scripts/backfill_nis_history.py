"""
Phase 64 — NIS History Backfill

Backfills NewsSignalCache with point-in-time news scores for past trading days
using Finnhub's historical company-news endpoint.

IMPORTANT — No-Lookahead Guarantee:
  For each training day D, only articles published BEFORE 10:30 AM ET on day D
  are fetched and scored. This matches the bar-12 entry time used in production.
  The `as_of_date` column stores D so walk-forward can filter correctly:
    WHERE as_of_date <= training_date

Usage:
  python scripts/backfill_nis_history.py [--days 252] [--workers 4] [--dry-run]

Cost estimate:
  ~500 symbols × 252 days × 3 articles/day avg × $0.00025/Haiku call ≈ $95
  Runtime: 8–12 hours (Finnhub free tier: 60 req/min)
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ET = timezone(timedelta(hours=-4))  # EDT; adjust to -5 in winter if needed


# ── Trading day helpers ───────────────────────────────────────────────────────

def _is_trading_day(d: date) -> bool:
    """Exclude weekends. Does not exclude holidays (acceptable for backfill)."""
    return d.weekday() < 5


def _past_trading_days(n: int) -> list[date]:
    """Return the last n trading days in reverse chronological order (most recent first)."""
    days = []
    d = date.today() - timedelta(days=1)
    while len(days) < n:
        if _is_trading_day(d):
            days.append(d)
        d -= timedelta(days=1)
    return days


# ── DB helpers ────────────────────────────────────────────────────────────────

def _existing_symbols_for_date(db, as_of_date: date) -> set:
    """Return set of symbols already backfilled for this date."""
    from app.database.models import NewsSignalCache
    rows = db.query(NewsSignalCache.symbol).filter(
        NewsSignalCache.as_of_date == as_of_date
    ).all()
    return {r[0] for r in rows}


def _write_row(db, symbol: str, as_of_date: date, result: dict, articles: list) -> bool:
    """Write a NewsSignalCache row. Returns True if written, False if already exists."""
    from app.database.models import NewsSignalCache

    # Cache key includes the date so same-symbol different-days are separate rows
    cache_key = hashlib.sha256(
        f"stock_v1|{symbol}|{as_of_date.isoformat()}|{result.get('rationale', '')}".encode()
    ).hexdigest()[:64]

    existing = db.query(NewsSignalCache).filter_by(cache_key=cache_key).first()
    if existing:
        return False

    # Set evaluated_at to 10:30 AM ET on as_of_date (bar-12 entry time)
    entry_time = datetime(
        as_of_date.year, as_of_date.month, as_of_date.day,
        10, 30, 0, tzinfo=ET
    ).astimezone(timezone.utc).replace(tzinfo=None)

    db.add(NewsSignalCache(
        symbol=symbol,
        cache_key=cache_key,
        prompt_version="stock_v1",
        direction_score=result["direction_score"],
        materiality_score=result["materiality_score"],
        downside_risk_score=result["downside_risk_score"],
        upside_catalyst_score=result["upside_catalyst_score"],
        confidence=result["confidence"],
        already_priced_in_score=result["already_priced_in_score"],
        action_policy=result.get("action_policy", "ignore"),
        sizing_multiplier=result["sizing_multiplier"],
        rationale=result.get("rationale", ""),
        top_headlines=[a["headline"] for a in articles[:5]],
        evaluated_at=entry_time,
        as_of_date=as_of_date,
    ))
    db.commit()
    return True


# ── Per-symbol worker ─────────────────────────────────────────────────────────

def _score_symbol(symbol: str, as_of_date: date, dry_run: bool) -> dict:
    """
    Fetch and score news for symbol on as_of_date.
    Returns dict with keys: symbol, as_of_date, status, articles_found, written.
    """
    from app.news.sources.finnhub_source import fetch_company_news
    from app.news.llm_scorer import stock_score

    # Fetch articles published before 10:30 AM ET on as_of_date only (no lookahead)
    to_dt = datetime(as_of_date.year, as_of_date.month, as_of_date.day, 10, 30, tzinfo=ET)
    from_dt = to_dt - timedelta(hours=24)  # 24-hour lookback window before bar-12

    try:
        articles = fetch_company_news(symbol, lookback_hours=24)
        # Filter to only articles before 10:30 AM ET on as_of_date
        cutoff_ts = to_dt.timestamp()
        articles = [a for a in articles if a.get("datetime", 0) <= cutoff_ts]
    except Exception as exc:
        return {"symbol": symbol, "as_of_date": as_of_date, "status": f"fetch_error: {exc}",
                "articles_found": 0, "written": False}

    if dry_run:
        return {"symbol": symbol, "as_of_date": as_of_date, "status": "dry_run",
                "articles_found": len(articles), "written": False}

    try:
        result = stock_score(
            symbol=symbol,
            articles=articles,
            sector="Unknown",
            macro_context_summary="Historical backfill — no macro context",
            lookback_hours=24,
        )
        if result is None:
            return {"symbol": symbol, "as_of_date": as_of_date, "status": "no_score",
                    "articles_found": len(articles), "written": False}
    except Exception as exc:
        return {"symbol": symbol, "as_of_date": as_of_date, "status": f"score_error: {exc}",
                "articles_found": len(articles), "written": False}

    return {"symbol": symbol, "as_of_date": as_of_date, "status": "ok",
            "articles_found": len(articles), "result": result, "articles": articles,
            "written": False}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backfill NIS history for Phase 64")
    parser.add_argument("--days", type=int, default=252, help="Trading days to backfill (default 252 = ~1 year)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel Finnhub fetch workers (default 4; free tier: keep ≤4)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch articles but skip LLM scoring and DB writes")
    parser.add_argument("--symbols", nargs="+", help="Override symbol list (default: SP500)")
    args = parser.parse_args()

    # Load universe
    if args.symbols:
        universe = args.symbols
    else:
        from app.utils.constants import SP_500_TICKERS
        universe = list(SP_500_TICKERS)

    trading_days = _past_trading_days(args.days)
    logger.info(
        "Backfill plan: %d symbols × %d trading days | workers=%d | dry_run=%s",
        len(universe), len(trading_days), args.workers, args.dry_run,
    )
    estimated_cost = len(universe) * len(trading_days) * 3 * 0.00025
    logger.info("Estimated LLM cost: $%.2f (assuming ~3 articles/symbol/day avg)", estimated_cost)

    from app.database.session import get_session

    total_written = 0
    total_skipped = 0
    total_errors = 0
    total_cost = 0.0

    for day_idx, as_of_date in enumerate(trading_days):
        db = get_session()
        try:
            already_done = _existing_symbols_for_date(db, as_of_date)
        finally:
            db.close()

        symbols_todo = [s for s in universe if s not in already_done]
        skipped_today = len(universe) - len(symbols_todo)

        if not symbols_todo:
            logger.info("[%d/%d] %s — all %d symbols already backfilled, skipping",
                        day_idx + 1, len(trading_days), as_of_date, len(universe))
            continue

        logger.info("[%d/%d] %s — scoring %d symbols (%d already done)",
                    day_idx + 1, len(trading_days), as_of_date,
                    len(symbols_todo), skipped_today)

        day_written = 0
        day_errors = 0

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_score_symbol, sym, as_of_date, args.dry_run): sym
                for sym in symbols_todo
            }
            for future in as_completed(futures, timeout=300):
                out = future.result()
                sym = out["symbol"]

                if out["status"] not in ("ok", "dry_run"):
                    day_errors += 1
                    if "error" in out["status"]:
                        logger.debug("  %s: %s", sym, out["status"])
                    continue

                if args.dry_run or "result" not in out:
                    continue

                db = get_session()
                try:
                    written = _write_row(db, sym, as_of_date, out["result"], out.get("articles", []))
                    if written:
                        day_written += 1
                        # Rough cost tracking: ~200 input + 100 output tokens per call
                        total_cost += (200 * 0.80 + 100 * 4.00) / 1_000_000
                finally:
                    db.close()

            # Finnhub rate limit: 60 req/min free tier. With 4 workers, pause between symbol batches.
            time.sleep(1.0)

        total_written += day_written
        total_skipped += skipped_today
        total_errors += day_errors

        logger.info(
            "  Day done: written=%d skipped=%d errors=%d | running_total written=%d cost=$%.2f",
            day_written, skipped_today, day_errors, total_written, total_cost,
        )

    logger.info(
        "=== Backfill complete: %d written, %d skipped (already existed), %d errors | total cost $%.2f ===",
        total_written, total_skipped, total_errors, total_cost,
    )


if __name__ == "__main__":
    main()
