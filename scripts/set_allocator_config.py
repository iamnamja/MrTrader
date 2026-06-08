"""
One-shot applier for the live regime-aware sleeve allocator (Alpha-v4 P3).

DB values override schema defaults and are read live by the agents (no restart).

    python -m scripts.set_allocator_config --show                 # print current values
    python -m scripts.set_allocator_config --enable               # enable (scheme=equal)
    python -m scripts.set_allocator_config --enable --scheme vol  # enable + set scheme
    python -m scripts.set_allocator_config --disable              # back to static budgets

Default ships DISABLED (sleeves use static pm.trend_allocation_pct / pm.pead_size_mult).
Keep scheme='equal' until scripts/run_book_allocator.py selects vol/regime (on 2 sleeves
equal beats both). Enabling does NOT change behavior until the allocator has enough live
history (warmup) and a recompute runs (weekly, before the trend rebalance).
"""
from __future__ import annotations

import argparse

from app.database.session import get_session
from app.database.agent_config import get_agent_config, set_agent_config

WATCH = [
    "pm.allocator_enabled", "pm.allocator_scheme", "pm.allocator_vol_lookback",
    "pm.allocator_total_budget_pct", "pm.allocator_min_deployed_days",
    "pm.allocator_stale_days", "pm.allocator_trend_weight", "pm.allocator_pead_weight",
    "pm.allocator_last_computed",
]
_SCHEMES = ("equal", "vol", "regime")


def _show(db) -> None:
    print("Current allocator config:")
    for k in WATCH:
        print(f"  {k:32s} = {get_agent_config(db, k)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply live sleeve-allocator config")
    ap.add_argument("--enable", action="store_true", help="set pm.allocator_enabled=true")
    ap.add_argument("--disable", action="store_true", help="set pm.allocator_enabled=false")
    ap.add_argument("--scheme", choices=_SCHEMES, help="set pm.allocator_scheme")
    ap.add_argument("--show", action="store_true", help="print current values and exit")
    args = ap.parse_args()

    db = get_session()
    try:
        if args.show or not (args.enable or args.disable or args.scheme):
            _show(db)
            return
        if args.enable and args.disable:
            raise SystemExit("--enable and --disable are mutually exclusive")

        if args.enable:
            set_agent_config(db, "pm.allocator_enabled", "true")
            print("set pm.allocator_enabled = true")
        if args.disable:
            set_agent_config(db, "pm.allocator_enabled", "false")
            print("set pm.allocator_enabled = false")
        if args.scheme:
            set_agent_config(db, "pm.allocator_scheme", args.scheme)
            print(f"set pm.allocator_scheme = {args.scheme}")

        print("\nAfter:")
        _show(db)
        enabled = str(get_agent_config(db, "pm.allocator_enabled")).lower() == "true"
        scheme = get_agent_config(db, "pm.allocator_scheme")
        if not enabled:
            print("\nAllocator DISABLED — sleeves use static budgets (today's behavior).")
        else:
            print(f"\nAllocator ENABLED (scheme={scheme}). It recomputes weekly before the "
                  "trend rebalance; until each sleeve has pm.allocator_min_deployed_days of "
                  "live history it stays in WARMUP and the sleeves keep their static budgets. "
                  "Run scripts/run_book_allocator.py to check the gate before choosing vol/regime.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
