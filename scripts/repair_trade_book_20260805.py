"""One-off trade-book repair (2026-08-05) — align DB intent to broker reality before an
unattended 2-week window.

WHY
---
The Alpaca DBC book drifted from the DB: the DB ACTIVE row still carried the ORIGINAL 2026-07-20
fill (257) and never applied the two later fills (+7 on 07-27, -23 on 08-03). Broker truth is
257 + 7 - 23 = 241. With `pm.reconciliation_mode=enforce` and `pm.kill_switch_sm_mode=enforce`,
the weekly cash-sleeve reconciliation FAIL_CLOSEDs on that break and auto-escalates the kill-switch
state machine to HALT_NEW_RISK — which de-escalates ONLY by human action. Left alone it would have
frozen new risk from the next Monday run until someone was back at the keyboard.

It also collapses two stale PENDING_FILL rows whose orders had long since filled. That matters for
more than tidiness: `reconcile()` treats a pending qty as a tolerance BAND [held, held+pend], so a
phantom pending WIDENS the band and can mask a real future break. SGOV sat at held=0 / pend=487,
i.e. band [0, 487] — a total liquidation of the cash sleeve would NOT have been flagged. Promoting
it to ACTIVE collapses the band to an exact point check.

WHAT (broker is the source of truth in every case)
--------------------------------------------------
  id=144 DBC   ACTIVE       qty 257 -> 241   (apply the missed +7 / -23 fills)
  id=143 DBC   PENDING_FILL -> CANCELLED     (duplicate of 144; its sell f64fa07e already filled)
  id=136 SGOV  PENDING_FILL -> ACTIVE        (buy filled 2026-06-22; broker holds 487)
  id=127 HPE   RECONCILE_GHOST_UNRESOLVED -> FORCE_CLOSED_NO_POSITION  (no broker position; 2mo old)

Run `--dry-run` first. Every prior value is written to a timestamped JSON backup before any commit.

    python scripts/repair_trade_book_20260805.py --dry-run
    python scripts/repair_trade_book_20260805.py --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# (trade_id, symbol, {field: new_value}, human reason)
REPAIRS = [
    (144, "DBC", {"quantity": 241.0},
     "apply missed fills: 257 (07-20) +7 (07-27) -23 (08-03) = 241 broker truth"),
    (143, "DBC", {"status": "CANCELLED",
                  "status_reason": "duplicate DBC row superseded by id=144; its order "
                                   "f64fa07e (SELL 23, filled 08-03) is reflected in id=144 qty"},
     "collapse phantom PENDING_FILL that widened the recon tolerance band"),
    (136, "SGOV", {"status": "ACTIVE"},
     "buy d01c0718 filled 2026-06-22; broker holds 487 — promote so recon does an exact check"),
    (127, "HPE", {"status": "FORCE_CLOSED_NO_POSITION",
                  "status_reason": "ghost since 2026-06-04; no broker position — resolved 2026-08-05"},
     "clear long-stale unresolved ghost"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    from app.database import SessionLocal
    from app.database.models import Trade
    import app.live_trading.reconciliation as R

    db = SessionLocal()
    backup: list[dict] = []

    print(f"{'id':>5}  {'sym':<5} {'field':<14} {'before':<22} -> after")
    print("-" * 78)
    for tid, symbol, changes, reason in REPAIRS:
        t = db.get(Trade, tid)
        if t is None:
            print(f"{tid:>5}  {symbol:<5} !! row missing — skipped")
            continue
        if t.symbol != symbol:
            print(f"{tid:>5}  {symbol:<5} !! symbol mismatch (row is {t.symbol}) — skipped")
            continue

        row_backup = {"id": tid, "symbol": symbol, "reason": reason, "before": {}}
        for field, new in changes.items():
            before = getattr(t, field, None)
            row_backup["before"][field] = before
            print(f"{tid:>5}  {symbol:<5} {field:<14} {str(before):<22} -> {new}")
            if args.apply:
                setattr(t, field, new)
        backup.append(row_backup)
        print(f"       -- {reason}")

    if args.apply:
        out = Path(__file__).parent.parent / "logs" / "trade_book_repair_20260805_backup.json"
        out.write_text(json.dumps(backup, indent=2, default=str))
        db.commit()
        print(f"\nCOMMITTED. Prior values backed up to {out}")
    else:
        db.rollback()
        print("\nDRY RUN — nothing written.")

    # Read-only verification: what will the reconciler see?
    print("\n" + "=" * 78)
    print("POST-STATE (read-only reconciliation preview)")
    db2 = SessionLocal()
    expected = R.db_expected_positions(db2, R.im.ALPACA)
    pending = R.db_pending_positions(db2, R.im.ALPACA)
    from app.integrations.alpaca import AlpacaClient
    raw = AlpacaClient().get_positions()          # broker truth, fetched here and passed in
    actual = R.alpaca_actual_positions(raw, R.im.ALPACA)
    print(f"  expected(ACTIVE) : { {k[1]: v for k, v in expected.items()} }")
    print(f"  pending          : { {k[1]: v for k, v in pending.items()} }")
    print(f"  broker actual    : { {p.instrument_id: p.quantity for p in actual} }")
    res = R.reconcile(expected, actual, pending_qty=pending)
    print(f"\n  STATUS: {res.status}")
    for b in res.position_breaks:
        print(f"    BREAK {b.instrument_id}: expected {b.expected_qty} actual {b.actual_qty}")
    for n in res.notes:
        print(f"    note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
