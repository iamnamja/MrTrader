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
import atexit
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

# Singleton pidfile — lives next to the notifications DB in data/ (gitignored
# via data/.gitignore which ignores everything except whitelisted dirs).
PID_PATH = ROOT / "data" / "notify_watcher.pid"


def _pid_is_live(pid: int) -> bool:
    """Best-effort, stdlib-only liveness check for ``pid``.

    Fail-safe contract: if we *cannot determine* whether the pid is alive, we
    return ``False`` so the caller treats the pidfile as stale and reclaims it.
    This favors *running* the watcher (email draining must never be stuck at
    zero) over silently never starting because of an indeterminate stale lock.
    On Windows a recorded pid may belong to the venv launcher rather than the
    base interpreter; an inability to confirm it is alive therefore correctly
    biases toward reclaiming the lock.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        # Use tasklist to ask whether the PID exists at all. We cannot reliably
        # distinguish the python interpreter from the venv launcher with stdlib
        # alone, so "PID exists" is the strongest signal we use; if tasklist
        # errors for any reason we fail safe to False (reclaim).
        try:
            import subprocess
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                capture_output=True, text=True, timeout=10,
            )
            return f'"{pid}"' in out.stdout
        except Exception:
            return False
    # POSIX: signal 0 probes existence without sending a signal.
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but owned by another user — count as alive.
        return True
    except Exception:
        return False


def acquire_singleton(pid_path: Path = PID_PATH,
                      is_live=_pid_is_live) -> bool:
    """Acquire the singleton lock by writing our pid to ``pid_path``.

    Returns ``True`` if this process now owns the lock and should run the
    watcher loop; ``False`` if another live watcher already holds it (caller
    should exit quietly WITHOUT logging the "started" banner).

    Behavior:
      * No pidfile           → acquire (write our pid), return True.
      * Pidfile names a LIVE  → return False (duplicate; do not log "started").
      * Pidfile is STALE      → reclaim (overwrite with our pid), return True.
        (dead pid, unreadable/garbage contents, or our own pid)
    """
    pid_path = Path(pid_path)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    me = os.getpid()

    try:
        existing = pid_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        existing = ""
    except Exception:
        existing = ""

    if existing:
        try:
            other = int(existing)
        except ValueError:
            other = -1  # garbage contents → treat as stale
        if other != me and other > 0 and is_live(other):
            log.info("notify_watcher already running (pid=%d) — exiting", other)
            return False
        # else: stale (dead pid / garbage / our own pid) → fall through, reclaim

    pid_path.write_text(str(me), encoding="utf-8")
    return True


def release_singleton(pid_path: Path = PID_PATH) -> None:
    """Remove the pidfile on clean exit, but only if it still names us."""
    pid_path = Path(pid_path)
    try:
        if pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pid_path.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _stop(*_) -> None:
    global _running
    _running = False
    log.info("shutdown requested")


def run_watcher() -> None:
    # Singleton guard: if another live watcher already holds the lock, exit
    # quietly (no "started" banner, no second drainer). A stale lock is
    # reclaimed so draining is never stuck at zero watchers.
    if not acquire_singleton():
        return

    atexit.register(release_singleton)
    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)
    log.info("notify_watcher started (db=%s, pid=%d)", notifier.DB_PATH, os.getpid())

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

    release_singleton()
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
