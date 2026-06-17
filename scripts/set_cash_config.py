"""
One-shot applier for the Alpha-v9 P1-1 live cash / T-bill sleeve config (DB values
override schema defaults; read live by the orchestrator — NO uvicorn restart needed).

The cash sleeve parks idle SETTLED cash (beyond pm.cash_buffer_pct of NAV) into a
T-bill ETF (SGOV/BIL) so the ~50% idle book earns the risk-free rate instead of zero,
and sells T-bills to refill the buffer when cash dips below it (risk-off). It runs
weekly on pm.cash_rebalance_weekday at 09:50 ET, AFTER the trend rebalance.

Run ONCE post-deploy:

    python -m scripts.set_cash_config                 # safe baseline (dormant + shadow)
    python -m scripts.set_cash_config --enable        # RUN the sleeve in SHADOW (no orders)
    python -m scripts.set_cash_config --enable --arm  # RUN it LIVE (sends real T-bill orders)
    python -m scripts.set_cash_config --show          # just print current values
    python -m scripts.set_cash_config --dry-run       # force a SHADOW rebalance now + print the
                                                      #   T-bill orders it WOULD place (sends none)

What it sets (idempotent):
  * pm.cash_enabled        = false    (master flag; --enable flips it true to run)
  * pm.cash_shadow         = true     (dry-run: logs would-be orders, sends nothing;
                                        --arm flips it false to send real orders)
  * pm.cash_buffer_pct     = 0.02     (settled-cash buffer kept out of T-bills, 2% of NAV)
  * pm.cash_universe       = SGOV,BIL (T-bill ETFs; first = primary buy target)
  * pm.cash_rebalance_weekday = 0     (Monday; runs after the trend rebalance)

Two independent switches (same shape as set_trend_config):
  --enable  -> pm.cash_enabled=true   (does the sleeve run at all?)
  --arm     -> pm.cash_shadow=false   (does it send orders, or just log them?)
Recommended sequence: --enable first (shadow), inspect a rebalance, THEN --enable --arm.
The cash sleeve is far lower-risk than trend (it buys cash-equivalent T-bills with idle
settled cash, excluded from the risk gross cap, fail-closed), so going straight to
--enable --arm after a --dry-run is reasonable.
"""
from __future__ import annotations

import argparse

from app.database.session import get_session
from app.database.agent_config import get_agent_config, set_agent_config

# Baseline = the safety defaults (dormant + shadow). --enable/--arm flip the two switches.
# buffer/universe/weekday are written explicitly so the live DB carries an auditable,
# self-documenting row rather than relying on the schema default.
BASE = {
    "pm.cash_enabled": "false",
    "pm.cash_shadow": "true",
    "pm.cash_buffer_pct": 0.02,
    "pm.cash_universe": "SGOV,BIL",
    "pm.cash_rebalance_weekday": 0,
}

WATCH = list(BASE.keys())


def _show(db) -> None:
    print("Current cash-sleeve config:")
    for k in WATCH:
        print(f"  {k:28s} = {get_agent_config(db, k)}")


def _dry_run(db) -> None:
    """Force a SHADOW rebalance against the live (paper) account and print what it would do.
    Sends NO orders regardless of the live flags: we force enabled=True for the run but pin
    shadow=True for the duration, then restore both flags exactly."""
    print("Cash sleeve DRY-RUN (forced shadow — no orders will be sent):\n")
    prev_enabled = get_agent_config(db, "pm.cash_enabled")
    prev_shadow = get_agent_config(db, "pm.cash_shadow")
    try:
        set_agent_config(db, "pm.cash_enabled", "true")
        set_agent_config(db, "pm.cash_shadow", "true")
        from app.live_trading import cash_sleeve
        summary = cash_sleeve.run_cash_rebalance(db, force=True)
    finally:
        # Restore the operator's intended flags no matter what the run did.
        set_agent_config(db, "pm.cash_enabled", prev_enabled)
        set_agent_config(db, "pm.cash_shadow", prev_shadow)

    print(f"  status      = {summary.get('status')}")
    print(f"  mode        = {summary.get('mode')}")
    print(f"  action      = {summary.get('action')}")
    print(f"  nav         = {summary.get('nav')}")
    print(f"  cash_on_hand= {summary.get('cash_on_hand')}")
    print(f"  buffer      = {summary.get('buffer')}")
    print(f"  deployable  = {summary.get('deployable')}")
    approved = summary.get("approved") or []
    if approved:
        print(f"  would place {len(approved)} order(s):")
        for it in approved:
            print(f"    - {it.get('side'):4s} {it.get('symbol'):6s} x{it.get('qty')}")
    else:
        print("  would place 0 orders (within buffer band or nothing deployable)")
    print("\n  (flags restored — no live behavior changed by this dry-run)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply live cash/T-bill-sleeve config")
    ap.add_argument("--enable", action="store_true",
                    help="set pm.cash_enabled=true (run the sleeve)")
    ap.add_argument("--arm", action="store_true",
                    help="set pm.cash_shadow=false (send real orders)")
    ap.add_argument("--show", action="store_true", help="print current values and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="force a SHADOW rebalance now and print what it would do (no orders)")
    args = ap.parse_args()

    db = get_session()
    try:
        if args.show:
            _show(db)
            return
        if args.dry_run:
            _dry_run(db)
            return

        values = dict(BASE)
        if args.enable:
            values["pm.cash_enabled"] = "true"
        if args.arm:
            values["pm.cash_shadow"] = "false"

        for k, v in values.items():
            set_agent_config(db, k, v)
            print(f"set {k} = {v}")

        print("\nAfter:")
        _show(db)

        enabled = str(values["pm.cash_enabled"]).lower() == "true"
        shadow = str(values["pm.cash_shadow"]).lower() == "true"
        if not enabled:
            print("\nCash sleeve is DORMANT (won't run). Re-run with --enable to run it in "
                  "shadow; add --arm to also send real orders.")
        elif shadow:
            print("\nCash sleeve is ENABLED + SHADOW: on the next cash rebalance weekday it "
                  "will compute & LOG the T-bill orders it would place (decision_audit "
                  "strategy='cash', block_reason='shadow') but send NOTHING. Inspect, then "
                  "re-run with --enable --arm to go live.")
        else:
            print("\n[LIVE] Cash sleeve is ENABLED + LIVE: on the next cash rebalance weekday "
                  "it WILL place real T-bill orders (paper account), parking idle settled "
                  "cash beyond the buffer into SGOV/BIL.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
