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
import time
from typing import Optional

log = logging.getLogger(__name__)

HEARTBEAT_PATH = os.path.join("data", "heartbeat.json")


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
