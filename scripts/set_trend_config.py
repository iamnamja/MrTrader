"""
One-shot applier for the Alpha-v4 live trend-sleeve config (DB values override
schema defaults; read live by the agents — NO uvicorn restart needed).

Run ONCE post-deploy:

    python -m scripts.set_trend_config             # safe baseline (dormant + shadow)
    python -m scripts.set_trend_config --enable     # RUN the sleeve in SHADOW (no orders)
    python -m scripts.set_trend_config --enable --arm   # RUN it LIVE (sends real orders)
    python -m scripts.set_trend_config --show       # just print current values

What it sets (idempotent):
  * pm.trend_enabled        = false   (master flag; --enable flips it true to run)
  * pm.trend_shadow         = true    (dry-run: logs would-be orders, sends nothing;
                                        --arm flips it false to send real orders)
  * pm.trend_allocation_pct = 0.40    (equal-capital 50/50 with PEAD under 80% gross)
  * pm.pead_size_mult       = 1.0     (PEAD telemetry dial — was B4 ramp 3.0)
  * pm.pead_max_position_pct= 0.05    (PEAD telemetry dial — was B4 ramp 0.10)

Two independent switches:
  --enable  -> pm.trend_enabled=true  (does the sleeve run at all?)
  --arm     -> pm.trend_shadow=false  (does it send orders, or just log them?)
Recommended sequence: run with --enable first (shadow), inspect a rebalance, THEN
re-run with --enable --arm to go live.
"""
from __future__ import annotations

import argparse

from app.database.session import get_session
from app.database.agent_config import get_agent_config, set_agent_config

BASE = {
    "pm.trend_enabled": "false",
    "pm.trend_shadow": "true",
    "pm.trend_allocation_pct": 0.40,
    "pm.pead_size_mult": 1.0,
    "pm.pead_max_position_pct": 0.05,
}

WATCH = list(BASE.keys()) + [
    "pm.trend_max_position_pct", "pm.trend_universe", "pm.trend_rebalance_weekday",
]


def _show(db) -> None:
    print("Current trend / PEAD-dial config:")
    for k in WATCH:
        print(f"  {k:28s} = {get_agent_config(db, k)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply live trend-sleeve config")
    ap.add_argument("--enable", action="store_true",
                    help="set pm.trend_enabled=true (run the sleeve)")
    ap.add_argument("--arm", action="store_true",
                    help="set pm.trend_shadow=false (send real orders)")
    ap.add_argument("--show", action="store_true", help="print current values and exit")
    args = ap.parse_args()

    db = get_session()
    try:
        if args.show:
            _show(db)
            return

        values = dict(BASE)
        if args.enable:
            values["pm.trend_enabled"] = "true"
        if args.arm:
            values["pm.trend_shadow"] = "false"

        for k, v in values.items():
            set_agent_config(db, k, v)
            print(f"set {k} = {v}")

        print("\nAfter:")
        _show(db)

        enabled = str(values["pm.trend_enabled"]).lower() == "true"
        shadow = str(values["pm.trend_shadow"]).lower() == "true"
        if not enabled:
            print("\nTrend sleeve is DORMANT (won't run). Re-run with --enable to run it "
                  "in shadow; add --arm to also send real orders.")
        elif shadow:
            print("\nTrend sleeve is ENABLED + SHADOW: it will compute & LOG would-be "
                  "orders on the next rebalance weekday but send NOTHING. Inspect "
                  "decision_audit (strategy='trend'), then re-run with --enable --arm "
                  "to go live.")
        else:
            print("\n⚠️  Trend sleeve is ENABLED + LIVE: it WILL place real ETF orders "
                  "on the next rebalance weekday (paper account).")
    finally:
        db.close()


if __name__ == "__main__":
    main()
