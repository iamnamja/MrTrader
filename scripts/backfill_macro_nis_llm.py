"""
Phase 90b — Macro NIS LLM Backfill

Fetches historical market-wide news from Polygon (SPY, QQQ, TLT, VIX proxy)
for each past trading day, scores it with the LLM macro classifier, and writes
results to MacroSignalCache.

Safe to run during market hours — only reads Polygon historical news and writes
to MacroSignalCache rows for past dates. The live trading system only reads
MacroSignalCache for today's date (written at 9 AM premarket); historical rows
are never touched by the live path.

Usage:
  python scripts/backfill_macro_nis_llm.py [--days 365] [--workers 4] [--dry-run]

Cost estimate:
  ~252 days × 1 LLM call/day × $0.00025/Haiku call ≈ $0.06 (essentially free)
  Runtime: ~30-60 min depending on Polygon rate limits and LLM latency
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("macro_llm_backfill")

# Market-wide tickers — news about these proxies macro sentiment
MACRO_TICKERS = ["SPY", "QQQ", "TLT", "IWM"]
MAX_HEADLINES_PER_DAY = 15  # enough context without bloating the prompt
POLYGON_PAUSE = 0.25  # seconds between Polygon requests (well within free tier)


def _trading_days(start: date, end: date) -> list[date]:
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _fetch_day_headlines(polygon_client, day: date) -> list[str]:
    """Fetch macro-relevant news headlines from Polygon for a given date."""
    from app.config import settings
    import requests

    date_str = day.strftime("%Y-%m-%d")
    next_date = (day + timedelta(days=1)).strftime("%Y-%m-%d")
    headlines = []

    for ticker in MACRO_TICKERS:
        try:
            r = requests.get(
                "https://api.polygon.io/v2/reference/news",
                params={
                    "apiKey": settings.polygon_api_key,
                    "ticker": ticker,
                    "published_utc.gte": f"{date_str}T00:00:00Z",
                    "published_utc.lte": f"{next_date}T00:00:00Z",
                    "limit": 5,
                    "sort": "published_utc",
                    "order": "desc",
                },
                timeout=15,
            )
            time.sleep(POLYGON_PAUSE)
            if r.status_code != 200:
                continue
            for item in r.json().get("results", []):
                title = item.get("title", "").strip()
                if title:
                    headlines.append(title)
        except Exception as exc:
            logger.debug("Polygon news fetch failed for %s on %s: %s", ticker, date_str, exc)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for h in headlines:
        if h not in seen:
            seen.add(h)
            unique.append(h)

    return unique[:MAX_HEADLINES_PER_DAY]


def _score_headlines(headlines: list[str], day: date) -> dict | None:
    """Run LLM macro classifier on a list of market news headlines."""
    if not headlines:
        return {
            "risk_level": "LOW",
            "direction": "NEUTRAL",
            "sizing_factor": 1.0,
            "block_new_entries": False,
            "consensus_summary": "No market news found for this date",
            "rationale": "No headlines available — neutral default",
        }

    # Build synthetic event list compatible with macro_classify format
    # Each headline becomes a "news event" with medium importance
    events = [
        {
            "event_type": f"market_news_{i+1}",
            "event_name": h,
            "event_time": datetime(day.year, day.month, day.day, 9, 30, tzinfo=timezone.utc),
            "importance": "medium",
            "estimate": None,
            "prior": None,
            "actual": None,
        }
        for i, h in enumerate(headlines)
    ]

    # Use a custom prompt variant that works with news headlines instead of
    # economic calendar events — same output schema as macro_classify
    try:
        from app.news.llm_scorer import _call_llm, _MACRO_SYSTEM, MACRO_PROMPT_VERSION, _cache_key, _cache_get

        headline_block = "\n".join(f"- {h}" for h in headlines)
        ckey = _cache_key(
            f"macro_news_backfill_{MACRO_PROMPT_VERSION}",
            str(day),
            *headlines,
        )
        cached = _cache_get(ckey, ttl=86400 * 30)
        if cached:
            return cached

        prompt = f"""{_MACRO_SYSTEM}

Today is {day.strftime('%Y-%m-%d')}. The following market news headlines were published today:

{headline_block}

Based on these headlines, classify the overall day-level macro risk for NEW equity swing entries (3-5 day holds).

Return JSON with exactly these fields:
{{
  "risk_level": "<LOW|MEDIUM|HIGH>",
  "direction": "<BULLISH|NEUTRAL|BEARISH>",
  "sizing_factor": <float 0.5-1.0, where 1.0=no change, 0.5=half size>,
  "block_new_entries": <true only when risk_level=HIGH AND outcome is genuinely uncertain>,
  "consensus_summary": "<one sentence: what the market mood was and why>",
  "rationale": "<one sentence: why this risk level was chosen>"
}}"""

        result = _call_llm(prompt, "macro_backfill")
        return result
    except Exception as exc:
        logger.debug("LLM macro classify failed for %s: %s", day, exc)
        return None


def _process_day(day: date, dry_run: bool) -> str:
    """Fetch headlines, score, write to DB. Returns status string."""
    from app.database.session import get_session
    from app.database.models import MacroSignalCache

    date_str = day.strftime("%Y-%m-%d")

    db = get_session()
    try:
        existing = db.query(MacroSignalCache).filter_by(date=date_str).first()
        # Only overwrite our placeholder backfill rows (prompt_version=backfill_v1)
        # Never overwrite real LLM rows
        if existing and existing.prompt_version != "backfill_v1":
            return f"{date_str}: skip (real LLM row exists)"

        headlines = _fetch_day_headlines(None, day)
        result = _score_headlines(headlines, day)

        if result is None:
            return f"{date_str}: LLM failed — skipped"

        # Also compute aggregated stock NIS features for events_payload
        macro_nis = {}
        try:
            from app.database.models import NewsSignalCache
            rows = db.query(NewsSignalCache).filter(
                NewsSignalCache.as_of_date == date_str
            ).all()
            if rows:
                dirs = [r.direction_score for r in rows if r.direction_score is not None]
                mats = [r.materiality_score for r in rows if r.materiality_score is not None]
                risks = [r.downside_risk_score for r in rows if r.downside_risk_score is not None]
                n = len(dirs)
                if n > 0:
                    macro_nis = {
                        "macro_avg_direction": round(sum(dirs) / n, 4),
                        "macro_pct_bearish": round(sum(1 for d in dirs if d < -0.3) / n, 4),
                        "macro_pct_bullish": round(sum(1 for d in dirs if d > 0.3) / n, 4),
                        "macro_avg_materiality": round(sum(mats) / len(mats), 4) if mats else 0.0,
                        "macro_pct_high_risk": round(sum(1 for r in risks if r > 0.7) / len(risks), 4) if risks else 0.0,
                        "n_symbols": n,
                    }
        except Exception:
            pass

        if dry_run:
            return (
                f"{date_str}: [DRY-RUN] risk={result['risk_level']} "
                f"dir={result['direction']} sizing={result['sizing_factor']} "
                f"headlines={len(headlines)}"
            )

        if existing:
            existing.prompt_version = "llm_news_v1"
            existing.risk_level = result["risk_level"]
            existing.direction = result["direction"]
            existing.sizing_factor = float(result["sizing_factor"])
            existing.block_new_entries = bool(result.get("block_new_entries", False))
            existing.rationale = result.get("rationale", "")
            existing.events_payload = {
                "headlines": headlines,
                "llm_result": result,
                "macro_nis_features": macro_nis,
            }
        else:
            row = MacroSignalCache(
                date=date_str,
                prompt_version="llm_news_v1",
                risk_level=result["risk_level"],
                direction=result["direction"],
                sizing_factor=float(result["sizing_factor"]),
                block_new_entries=bool(result.get("block_new_entries", False)),
                rationale=result.get("rationale", ""),
                events_payload={
                    "headlines": headlines,
                    "llm_result": result,
                    "macro_nis_features": macro_nis,
                },
            )
            db.add(row)

        db.commit()
        return (
            f"{date_str}: risk={result['risk_level']} dir={result['direction']} "
            f"sizing={result['sizing_factor']} headlines={len(headlines)}"
        )
    except Exception as exc:
        db.rollback()
        return f"{date_str}: ERROR — {exc}"
    finally:
        db.close()


def run(days: int = 365, workers: int = 4, dry_run: bool = False) -> None:
    end = date.today() - timedelta(days=1)  # yesterday — today handled by live system
    start = end - timedelta(days=days)
    trading_days = _trading_days(start, end)

    logger.info(
        "Macro LLM backfill: %s → %s (%d trading days, %d workers, dry_run=%s)",
        start, end, len(trading_days), workers, dry_run,
    )

    done = errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_day, d, dry_run): d for d in trading_days}
        for fut in as_completed(futures):
            msg = fut.result()
            if "ERROR" in msg:
                errors += 1
                logger.warning(msg)
            elif "skip" in msg:
                logger.debug(msg)
            else:
                done += 1
                logger.info(msg)

    logger.info("Done: %d processed, %d errors", done, errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill MacroSignalCache with LLM-scored market news")
    parser.add_argument("--days", type=int, default=365, help="Calendar days to backfill")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(days=args.days, workers=args.workers, dry_run=args.dry_run)
