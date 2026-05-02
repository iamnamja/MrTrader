"""
Phase 83 — Deadman watchdog script.

Run as a separate process (cron every 2 min, or `python scripts/watchdog.py`
in a loop) on any always-on machine.

If the PM heartbeat in DB is stale (> STALE_THRESHOLD_SECONDS) AND the
market is open, this script calls the kill-switch API to halt trading and
close all positions.

Usage:
    python scripts/watchdog.py               # runs one check then exits
    python scripts/watchdog.py --loop        # runs every CHECK_INTERVAL_SECONDS forever
    python scripts/watchdog.py --url http://localhost:8000  # custom base URL
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("watchdog")
logging.basicConfig(
    format="%(asctime)s %(levelname)s [watchdog] %(message)s",
    level=logging.INFO,
)

STALE_THRESHOLD_SECONDS = 300   # 5 min without heartbeat → trigger
CHECK_INTERVAL_SECONDS = 120    # 2 min between checks in --loop mode
MARKET_OPEN_HOUR = 9            # ET
MARKET_CLOSE_HOUR = 16          # ET
KILL_REASON = "deadman: PM heartbeat stale (watchdog.py)"


def _market_open_et() -> bool:
    """Return True if current ET time is within 09:30–16:00."""
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        if now_et.weekday() >= 5:
            return False
        t = now_et.hour * 60 + now_et.minute
        return (9 * 60 + 30) <= t < (16 * 60)
    except Exception:
        return False


def _get_last_heartbeat(base_url: str) -> datetime | None:
    """
    Fetch the last PM heartbeat timestamp from the API.
    Falls back to direct DB query if no API route exists yet.
    """
    try:
        import requests
        resp = requests.get(f"{base_url}/api/health/heartbeat", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            ts = data.get("last_beat")
            if ts:
                return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    except Exception:
        pass

    # Direct DB fallback (works on same host)
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from app.database.session import get_session
        from app.database.models import ProcessHeartbeat
        db = get_session()
        try:
            row = db.query(ProcessHeartbeat).filter_by(process_name="portfolio_manager").first()
            if row and row.last_beat:
                return row.last_beat.replace(tzinfo=timezone.utc)
        finally:
            db.close()
    except Exception as exc:
        logger.warning("Could not read heartbeat from DB: %s", exc)

    return None


def _activate_kill_switch(base_url: str) -> bool:
    """POST to kill switch API. Returns True on success."""
    try:
        import requests
        resp = requests.post(
            f"{base_url}/api/orchestrator/kill-switch/activate",
            json={"reason": KILL_REASON},
            timeout=10,
        )
        if resp.status_code == 200:
            logger.critical("Kill switch activated via API: %s", resp.json())
            return True
        logger.error("Kill switch API returned %d: %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        logger.error("Could not reach kill switch API: %s", exc)
    return False


def check_once(base_url: str) -> bool:
    """Run one watchdog check. Returns True if kill switch was triggered."""
    if not _market_open_et():
        logger.info("Market closed — skipping heartbeat check")
        return False

    last_beat = _get_last_heartbeat(base_url)
    if last_beat is None:
        logger.warning("No heartbeat record found — PM may not have started yet")
        return False

    age = (datetime.now(timezone.utc) - last_beat).total_seconds()
    logger.info("PM heartbeat age: %.0fs (threshold=%ds)", age, STALE_THRESHOLD_SECONDS)

    if age > STALE_THRESHOLD_SECONDS:
        logger.critical(
            "STALE HEARTBEAT: last beat %.0fs ago — activating kill switch", age
        )
        return _activate_kill_switch(base_url)

    logger.info("Heartbeat OK")
    return False


def main():
    parser = argparse.ArgumentParser(description="MrTrader deadman watchdog")
    parser.add_argument("--url", default="http://localhost:8000", help="MrTrader base URL")
    parser.add_argument("--loop", action="store_true", help="Run continuously every 2 min")
    args = parser.parse_args()

    if args.loop:
        logger.info("Watchdog running in loop mode (interval=%ds)", CHECK_INTERVAL_SECONDS)
        while True:
            try:
                check_once(args.url)
            except Exception as exc:
                logger.error("Watchdog check error: %s", exc)
            time.sleep(CHECK_INTERVAL_SECONDS)
    else:
        triggered = check_once(args.url)
        sys.exit(1 if triggered else 0)


if __name__ == "__main__":
    main()
