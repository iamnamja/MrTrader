"""verify_enforce_rebalance.py — post-rebalance verification of the FIRST enforce-mode trend
rebalance (Mon 2026-07-13, then every Monday).

Confirms three things after the ~09:45 ET Monday trend rebalance:
  1. the enforce config is actually in force (whole-book gate + reconciliation = enforce);
  2. the rebalance ran LIVE and CLEAN — no spurious enforce HOLD (whole-book gate / reconciliation
     / kill-switch) blocking a legitimate rebalance;
  3. the CH0b live-forward scorecard CAPTURED this rebalance's per-governor multipliers +
     ungoverned counterfactual — that capture happens ONCE, at the rebalance, and CANNOT be
     backfilled, so a miss is unrecoverable.

Read-only. Prints + emails a PASS / ATTENTION summary. Safe to run any day: off-Mondays / before
the rebalance it just reports "no live rebalance recorded today" without alarming.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone

EXPECT = {"pm.whole_book_gate_mode": "enforce", "pm.reconciliation_mode": "enforce",
          "pm.per_name_gate_mode": "shadow", "pm.trend_enabled": "true", "pm.trend_shadow": "false"}
HOLD_REASONS = {"whole_book_gate", "reconciliation", "per_name_gate", "kill_switch", "kill_switch_sm"}


def _json(s):
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}


def check() -> dict:
    today = date.today().isoformat()
    rep: dict = {"date": today, "attention": [], "config": {}, "scorecard": None, "decisions": {}}

    def flag(msg):
        rep["attention"].append(msg)

    # ── 1. enforce config actually in force ──
    try:
        from app.database.agent_config import get_agent_config
        from app.database.session import get_session
        with get_session() as db:
            for k, want in EXPECT.items():
                got = str(get_agent_config(db, k) or "").strip().lower()
                rep["config"][k] = got
                if got != want:
                    flag(f"config {k}={got!r} (expected {want!r})")
    except Exception as exc:  # noqa: BLE001
        flag(f"could not read config: {exc}")

    # ── 2. CH0b scorecard capture (the UN-BACKFILLABLE governor data) ──
    try:
        from app.live_trading import back_validation as bv
        rows = [r for r in bv.read_daily(since=today) if r.get("trade_date") == today]
        if not rows:
            rep["scorecard"] = {"present": False}
            flag("no scorecard row for today — the live rebalance either did not run yet "
                 "(off-Monday / pre-09:45), was HELD (see 'ENFORCE HOLD' below if so), or ran "
                 "FLAT-by-design (TSMOM all-cash → no intent to record, which is OK). Cross-check "
                 "the decisions + the enforce-HOLD flag to disambiguate")
        else:
            r = rows[0]
            intent, ungov = _json(r.get("intended_weights")), _json(r.get("ungoverned_weights"))
            rep["scorecard"] = {
                "present": True, "n_intended": len(intent), "n_ungoverned": len(ungov),
                "crash_mult": r.get("crash_mult"), "credit_mult": r.get("credit_mult"),
                "ladder_mult": r.get("ladder_mult"), "overlay_mult": r.get("overlay_mult"),
                "n_blocked": r.get("n_blocked"),
            }
            if not intent:
                flag("scorecard: intended_weights EMPTY — no LIVE intent recorded (held or shadow)")
            if not ungov:
                flag("scorecard: ungoverned_weights MISSING — CH0b counterfactual NOT captured "
                     "(cannot backfill)")
            if r.get("crash_mult") is None:
                flag("scorecard: crash_mult NULL — per-governor multipliers not captured")
    except Exception as exc:  # noqa: BLE001
        flag(f"could not read scorecard: {exc}")

    # ── 3. today's trend decisions — detect a spurious enforce HOLD ──
    try:
        from zoneinfo import ZoneInfo
        from app.database.models import DecisionAudit
        from app.database.session import get_session
        start_et = datetime.now(ZoneInfo("America/New_York")).replace(hour=0, minute=0, second=0,
                                                                      microsecond=0)
        with get_session() as db:
            rows = (db.query(DecisionAudit)
                    .filter(DecisionAudit.strategy == "trend")
                    .filter(DecisionAudit.decided_at >= start_et.astimezone(timezone.utc))
                    .all())
        blocks = [x for x in rows if x.final_decision == "block"]
        reasons = Counter(x.block_reason for x in blocks)
        rep["decisions"] = {"n_total": len(rows), "n_enter": sum(x.final_decision == "enter" for x in rows),
                            "n_block": len(blocks), "block_reasons": dict(reasons)}
        held = HOLD_REASONS & set(reasons)
        if held:
            flag(f"ENFORCE HOLD detected (block_reason {sorted(held)}) — VERIFY this is a REAL "
                 f"breach, not a spurious hold; one-line revert if spurious: set the mode back to "
                 f"'shadow' via set_agent_config")
    except Exception as exc:  # noqa: BLE001
        flag(f"could not read decisions: {exc}")

    rep["status"] = "ATTENTION" if rep["attention"] else "OK"
    return rep


def run_and_report() -> dict:
    """check() + email the summary (dedup'd per day). Never raises. Returns the report. This is
    the entry the orchestrator's Monday job calls; main() wraps it for CLI use."""
    rep = check()
    try:
        from app.notifications import notifier
        notifier.enqueue("enforce_rebalance_verification", rep,
                         dedup_key=f"enforce_verify_{rep['date']}")
    except Exception as exc:  # noqa: BLE001
        rep["attention"].append(f"email enqueue failed (non-fatal): {exc}")
    return rep


def main() -> int:
    rep = run_and_report()
    print(f"[enforce-rebalance verify {rep['date']}] STATUS: {rep['status']}")
    print(f"  config:    {rep['config']}")
    print(f"  scorecard: {rep['scorecard']}")
    print(f"  decisions: {rep['decisions']}")
    for a in rep["attention"]:
        print(f"    - ATTENTION: {a}")
    return 0 if rep["status"] == "OK" else 2


if __name__ == "__main__":
    sys.exit(main())
