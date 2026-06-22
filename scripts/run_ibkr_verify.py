"""Alpha-v10 P2.2 — live IBKR read-only smoke + verify-on-connect.

Connects (read-only) to the running IB Gateway/TWS paper session, prints the account + positions as
canonical objects, and VERIFIES the futures contract master (multiplier/exchange/currency) against
the live reqContractDetails. CRITICAL mismatches (multiplier / unresolvable) must be fixed before any
futures order is ever sized. Read-only: no orders are or can be placed.

Run: PYTHONPATH=. venv/Scripts/python scripts/run_ibkr_verify.py [--host H --port P --client-id N]
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7497)   # TWS paper
    ap.add_argument("--client-id", type=int, default=2)  # distinct from the app's default 1
    args = ap.parse_args()

    from app.live_trading.ibkr_adapter import IBKRReadOnlyAdapter

    ad = IBKRReadOnlyAdapter(host=args.host, port=args.port, client_id=args.client_id)
    h = ad.connect()
    print(f"[connect] connected={h.connected} detail={h.detail}")
    if not h.connected:
        print("NOT CONNECTED - is TWS/Gateway running in paper with API enabled?")
        return 2
    try:
        acct = ad.get_account()
        print(f"[account] venue={acct.venue} NAV={acct.nav:,.2f} cash={acct.cash:,.2f} "
              f"buying_power={acct.buying_power:,.2f} maint_margin={acct.maintenance_margin}")
        positions = ad.get_positions()
        print(f"[positions] {len(positions)} held")
        for p in positions:
            print(f"   {p.broker_symbol:6s} iid={p.instrument_id:8s} qty={p.quantity:g} "
                  f"px={p.price:g} mult={p.multiplier:g} notional={p.notional:,.0f} mapped={p.mapped}")

        print("[verify-on-connect] checking the futures contract master vs live reqContractDetails ...")
        mm = ad.verify_contracts()
        crit = [m for m in mm if m.critical]
        warn = [m for m in mm if not m.critical]
        if not mm:
            print("   ALL VERIFIED - every futures multiplier/exchange/currency matches the live API.")
        else:
            for m in crit:
                print(f"   CRITICAL {m.instrument_id}: {m.field} expected={m.expected} actual={m.actual}")
            for m in warn:
                print(f"   warn     {m.instrument_id}: {m.field} expected={m.expected} actual={m.actual}")
        print(f"[verify-on-connect] critical={len(crit)} warn={len(warn)}")
        return 1 if crit else 0
    finally:
        ad.disconnect()
        print("[disconnect] done")


if __name__ == "__main__":
    raise SystemExit(main())
