"""Alpha-v10 H4 — OUT-OF-BAND emergency flatten (operator tool).

Cancels ALL open orders + liquidates ALL positions on Alpaca, talking ONLY to the broker (no DB /
Redis / orchestrator). The "close everything now" button you can run from a terminal (or phone via
SSH) even if the brain is wedged. DRY-RUN by default — you must pass --execute to actually liquidate.

  DRY-RUN (safe, default):  PYTHONPATH=. venv/Scripts/python scripts/emergency_flatten.py
  EXECUTE  (liquidates!):   PYTHONPATH=. venv/Scripts/python scripts/emergency_flatten.py --execute
  EXECUTE + halt the brain: ... --execute --kill

Test it WEEKLY on the paper account (panel guidance): a dry-run is always safe; an --execute on paper
liquidates the paper book (the weekly rebalance re-establishes it). The IBKR flatten is R1 (TWS
Read-Only API blocks order placement until trading is enabled).
"""
from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser(description="Out-of-band broker-only emergency flatten (Alpaca).")
    ap.add_argument("--execute", action="store_true",
                    help="ACTUALLY cancel orders + liquidate positions (default: dry-run report only)")
    ap.add_argument("--kill", action="store_true",
                    help="also activate the kill-switch (halt the brain) — only with --execute")
    args = ap.parse_args()

    from app.live_trading.emergency_flatten import flatten_alpaca

    print(f"=== EMERGENCY FLATTEN ({'EXECUTE' if args.execute else 'DRY-RUN'}) ===")
    report = flatten_alpaca(execute=args.execute)
    print(json.dumps(report, indent=2, default=str))

    if args.execute and args.kill:
        try:
            from app.live_trading.kill_switch import kill_switch
            kill_switch.activate(reason="emergency_flatten --kill")
            print("[kill] kill-switch ACTIVATED — the brain will halt new trading.")
        except Exception as e:  # noqa: BLE001
            print(f"[kill] kill-switch activation FAILED (broker flatten still done): {e}")

    if report.get("errors"):
        print(f"\nCompleted with {len(report['errors'])} error(s) — review above.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
