"""
MrTrader notification watcher — drains the SQLite notification queue and sends emails.

Launch (keep running in background):
    nohup python scripts/notify_watcher.py > logs/notify_watcher.log 2>&1 &

Windows PowerShell:
    Start-Process python -ArgumentList "scripts/notify_watcher.py" `
        -RedirectStandardOutput logs/notify_watcher.log `
        -RedirectStandardError  logs/notify_watcher.err `
        -WindowStyle Hidden

Smoke test (enqueue a test notification and verify it arrives):
    python scripts/notify_watcher.py --test
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load .env before importing notifier
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from app.notifications import notifier  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("notify_watcher")

POLL_INTERVAL = 5   # seconds between queue polls
_running = True


def _stop(*_) -> None:
    global _running
    _running = False
    log.info("shutdown requested")


def run_watcher() -> None:
    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)
    log.info("notify_watcher started (db=%s)", notifier.DB_PATH)

    while _running:
        try:
            rows = notifier.pending(limit=20)
            for row_id, event_type, payload_json, created_at, attempts in rows:
                if notifier.should_throttle(event_type):
                    # Drop throttled progress events cleanly — mark sent so queue stays tidy
                    log.info("throttled event_type=%s id=%d (dropped)", event_type, row_id)
                    notifier.mark_sent(row_id)
                    continue
                try:
                    payload = json.loads(payload_json)
                    subject, html = notifier.render(event_type, payload)
                    ok = notifier._smtp_send(subject, html)
                    if ok:
                        notifier.mark_sent(row_id)
                        log.info("sent id=%d type=%s subject=%r", row_id, event_type, subject[:60])
                    else:
                        notifier.mark_failed(row_id, "smtp returned False")
                        log.warning("send failed id=%d type=%s (attempts=%d)", row_id, event_type, attempts + 1)
                except Exception as exc:
                    notifier.mark_failed(row_id, str(exc))
                    log.exception("send error id=%d type=%s", row_id, event_type)
        except Exception:
            log.exception("watcher loop error (continuing)")

        time.sleep(POLL_INTERVAL)

    log.info("notify_watcher stopped")


def run_test() -> None:
    """Enqueue a smoke-test notification and flush it immediately."""
    log.info("smoke test: enqueuing diag_complete notification…")
    row_id = notifier.enqueue("diag_complete", {
        "script":    "smoke_test",
        "duration":  "0s",
        "outcome":   "notify_watcher is working correctly",
        "artifacts": ["(none)"],
    })
    if row_id is None:
        log.error("enqueue returned None — check notifier.py")
        sys.exit(1)
    log.info("enqueued row_id=%d, sending now…", row_id)
    rows = notifier.pending(limit=1)
    if not rows:
        log.error("pending() returned nothing — DB write may have failed")
        sys.exit(1)
    rid, event_type, payload_json, _, _ = rows[0]
    subject, html = notifier.render(event_type, json.loads(payload_json))
    ok = notifier._smtp_send(subject, html)
    if ok:
        notifier.mark_sent(rid)
        log.info("smoke test email sent to %s — check your inbox", notifier.RECIPIENT)
    else:
        log.error("smtp send failed — check NOTIFY_GMAIL_USER / NOTIFY_GMAIL_APP_PASSWORD in .env")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MrTrader notification watcher")
    parser.add_argument("--test", action="store_true", help="send a smoke-test email and exit")
    args = parser.parse_args()

    if args.test:
        run_test()
    else:
        run_watcher()
