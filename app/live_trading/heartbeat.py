"""
heartbeat.py — Alpha-v10 H5: a DURABLE liveness heartbeat for the external dead-man watchdog.

The brain writes a small timestamped file every minute (a scheduler job). An EXTERNAL process (the
dead-man watchdog, a separate Python process) reads it WITHOUT touching the app — if the file goes
stale, the brain has died/hung and the watchdog alerts (and optionally flattens). A file is the
simplest truly-out-of-band channel on one host; the cross-host upgrade (Postgres/Redis) is noted for
the always-on-host step (R1).

The write is ATOMIC (tmp + os.replace) so the watchdog can never read a half-written file.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Optional

log = logging.getLogger(__name__)

HEARTBEAT_PATH = os.path.join("data", "heartbeat.json")

# Off-box dead-man's-snitch: the one liveness failure the on-box watchdog CANNOT catch is total-machine
# death (power/OS) — it dies with the box. If this env var is set to an external check URL (e.g. a free
# healthchecks.io check), the heartbeat job pings it every minute; when the box dies the pings stop and
# the EXTERNAL service alerts you. No-op until the env var is set.
SNITCH_URL_ENV = "MRTRADER_SNITCH_URL"


def write_heartbeat(path: str = HEARTBEAT_PATH, *, now: Optional[float] = None) -> bool:
    """Atomically write {ts, iso, pid}. Returns True on success; never raises (a heartbeat-write
    failure must not break the scheduler tick — the watchdog will treat a stale file as down)."""
    ts = time.time() if now is None else now
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {"ts": ts, "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)),
                   "pid": os.getpid()}
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)            # atomic on POSIX + Windows (tmp & path share `data/` -> same volume)
        return True
    except Exception as e:  # noqa: BLE001 — never break the scheduler tick
        log.debug("heartbeat write failed: %s", e)
        return False


def read_heartbeat(path: str = HEARTBEAT_PATH) -> Optional[dict]:
    """The last heartbeat payload, or None if missing/corrupt (treated as 'down' by the watchdog)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict) or "ts" not in d:
            return None
        return d
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def heartbeat_age_seconds(path: str = HEARTBEAT_PATH, *, now: Optional[float] = None) -> Optional[float]:
    """Seconds since the last heartbeat, or None if there is no readable heartbeat (no file yet /
    corrupt). None means 'unknown' — the watchdog treats both None and over-threshold as a trigger."""
    d = read_heartbeat(path)
    if d is None:
        return None
    try:
        ts = float(d["ts"])
    except (KeyError, TypeError, ValueError):
        return None
    return (time.time() if now is None else now) - ts


def ping_snitch(url: Optional[str] = None, *, timeout: float = 5.0) -> bool:
    """Best-effort off-box liveness ping (H5 follow-up for total-machine-death detection).

    If ``url`` (or the ``MRTRADER_SNITCH_URL`` env var) is set, fire a GET each heartbeat so an
    external dead-man's-snitch (e.g. healthchecks.io) alerts you when the pings stop — i.e. when the
    whole box dies and the on-box watchdog can't. Fire-and-forget on a daemon thread so a slow/hung
    request never delays the beat loop; never raises. Returns True if a ping was dispatched, False if
    no URL is configured (no-op). NOTE: True means 'dispatched', not 'delivered' — delivery is the
    external service's concern.
    """
    target = (url if url is not None else os.environ.get(SNITCH_URL_ENV, "")).strip()
    if not target:
        return False

    def _do() -> None:
        try:
            urllib.request.urlopen(target, timeout=timeout).close()   # nosec B310 — operator-set URL
        except Exception as e:  # noqa: BLE001 — a snitch failure must never affect the beat loop
            log.debug("snitch ping failed: %s", e)

    threading.Thread(target=_do, daemon=True, name="snitch-ping").start()
    return True
