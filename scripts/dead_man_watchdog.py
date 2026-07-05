"""Alpha-v10 H5 — EXTERNAL dead-man watchdog (operator process).

Run as a SEPARATE process from the brain. It reads the brain's durable heartbeat file (written every
minute by the orchestrator) and, if the heartbeat goes stale (the brain died/hung) or is missing, it
ALERTS (email via the notifier). Auto-flatten is OPT-IN (--auto-flatten, default OFF) per the panel:
"alert loudly; auto-flatten only when you're very confident."

  Alert-only (recommended):  PYTHONPATH=. venv/Scripts/python scripts/dead_man_watchdog.py
  With auto-flatten:         ... --auto-flatten
  One-shot (cron/test):      ... --once

It alerts ONCE per stale episode (re-arms after the heartbeat recovers), so a long outage doesn't
spam. Keep it running on the same host as the brain (cross-host = move the heartbeat to Postgres/Redis
in R1). This process is intentionally tiny and has no trading authority unless --auto-flatten is set.
"""
from __future__ import annotations

import argparse
import time


def _check_once(max_stale: float, auto_flatten: bool, already_alerted: bool) -> bool:
    """One staleness check. Returns the new `already_alerted` state. Alerts on a fresh stale episode;
    re-arms once the heartbeat is healthy again."""
    from app.live_trading.heartbeat import heartbeat_age_seconds

    age = heartbeat_age_seconds()
    stale = (age is None) or (age > max_stale)

    if not stale:
        if already_alerted:
            print(f"[watchdog] heartbeat RECOVERED (age={age:.0f}s) — re-armed")
        return False                                   # healthy -> re-arm

    if already_alerted:
        return True                                    # still stale, already alerted -> stay quiet

    age_str = "unknown (no heartbeat file)" if age is None else f"{age:.0f}s"
    print(f"[watchdog] HEARTBEAT STALE: age={age_str} > threshold={max_stale:.0f}s")

    flatten_result = "n/a (auto-flatten disabled)"
    if auto_flatten:
        try:
            from app.live_trading.emergency_flatten import flatten_alpaca
            rep = flatten_alpaca(execute=True)
            flatten_result = f"ok={rep.get('ok')} errors={rep.get('errors')}"
            try:
                from app.live_trading.kill_switch import kill_switch
                kill_switch.activate(reason="dead-man watchdog: stale heartbeat")
            except Exception as e:  # noqa: BLE001
                flatten_result += f" | kill-switch failed: {e}"
        except Exception as e:  # noqa: BLE001
            flatten_result = f"flatten FAILED: {e}"
        print(f"[watchdog] AUTO-FLATTEN: {flatten_result}")

    try:
        from app.notifications import notifier
        notifier.enqueue("dead_man_alert", {
            "age_seconds": age, "threshold_seconds": max_stale,
            "auto_flatten": auto_flatten, "flatten_result": flatten_result,
        }, dedup_key="dead_man_alert")
        print("[watchdog] alert enqueued")
    except Exception as e:  # noqa: BLE001
        print(f"[watchdog] alert enqueue FAILED: {e}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="External dead-man watchdog for the brain heartbeat.")
    ap.add_argument("--max-stale-sec", type=float, default=600.0,
                    help="alert if the heartbeat is older than this (default 600s = 10 min)")
    ap.add_argument("--interval-sec", type=float, default=60.0,
                    help="check cadence (default 60s)")
    ap.add_argument("--auto-flatten", action="store_true",
                    help="on stale: flatten the broker + trip the kill-switch (default: alert-only)")
    ap.add_argument("--once", action="store_true",
                    help="run a single check and exit (tests/cron). NOTE: the one-alert-per-episode "
                         "guarantee holds in the persistent loop; under repeated cron --once a "
                         "sustained outage re-alerts each run once the prior email is sent.")
    ap.add_argument("--start-grace-sec", type=float, default=0.0,
                    help="on the persistent loop, wait this long before the FIRST check so a "
                         "co-launched server has time to boot and write a fresh heartbeat — avoids a "
                         "false stale-alert on startup (e.g. serve.ps1 auto-launch). Ignored for "
                         "--once. Default 0.")
    args = ap.parse_args()

    print(f"[watchdog] start: max_stale={args.max_stale_sec:.0f}s interval={args.interval_sec:.0f}s "
          f"auto_flatten={args.auto_flatten}")
    if args.once:
        stale = _check_once(args.max_stale_sec, args.auto_flatten, already_alerted=False)
        return 1 if stale else 0

    if args.start_grace_sec > 0:
        print(f"[watchdog] startup grace: waiting {args.start_grace_sec:.0f}s for a fresh heartbeat "
              f"before the first check")
        time.sleep(args.start_grace_sec)

    alerted = False
    while True:
        try:
            alerted = _check_once(args.max_stale_sec, args.auto_flatten, alerted)
        except Exception as e:  # noqa: BLE001 — the watchdog must never die on a transient error
            print(f"[watchdog] check error (continuing): {e}")
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    raise SystemExit(main())
