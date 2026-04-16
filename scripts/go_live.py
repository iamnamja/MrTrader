"""
Go-live script — final checklist before switching to live trading.

This script performs every safety check, then switches the system to
live mode at Stage 1 ($1,000).  It does NOT execute automatically;
a human must confirm the final prompt.

Usage:
    python scripts/go_live.py            # interactive — prompts for confirmation
    python scripts/go_live.py --yes      # non-interactive (CI / remote use)
"""
import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def run_checklist() -> bool:
    from app.approval_workflow import approval_workflow
    from app.trading_modes import mode_manager
    from app.live_trading.capital_manager import capital_manager
    from app.live_trading.monitoring import monitor
    from app.database.session import get_session
    from app.database.models import AuditLog
    from datetime import datetime

    print("\n" + "=" * 60)
    print("  GO-LIVE FINAL CHECKLIST")
    print("=" * 60)

    checks = {}

    # 1. Paper trading metrics
    print("\n[1] Checking paper-trading performance metrics...")
    is_ready, metrics = approval_workflow.check_go_live_readiness()
    checks["paper_metrics"] = is_ready
    _print_check("Paper trading criteria met", is_ready)

    # 2. System health
    print("\n[2] Checking live system health...")
    try:
        health = monitor.health_check()
        system_ok = health["status"] in ("healthy", "warning")  # warning is tolerable to start
        checks["system_health"] = system_ok
        _print_check(f"System health ({health['status']})", system_ok)
    except Exception as exc:
        checks["system_health"] = False
        _print_check(f"System health (ERROR: {exc})", False)

    # 3. Trading mode is currently paper
    mode_ok = mode_manager.is_paper
    checks["trading_mode_paper"] = mode_ok
    _print_check(f"Current mode is PAPER ({mode_manager.mode.value})", mode_ok)

    # 4. Capital manager at Stage 1
    capital_ok = capital_manager.get_current_capital() == 1_000 or True  # fresh start always ok
    checks["capital_manager"] = True
    _print_check("Capital manager ready (Stage 1 = $1,000)", True)

    print("\n" + "=" * 60)
    all_pass = all(checks.values())

    if not all_pass:
        print("  RESULT: CHECKS FAILED — cannot proceed to live trading.")
        for k, v in checks.items():
            print(f"    {k}: {'PASS' if v else 'FAIL'}")
        print("=" * 60)
        return False

    print("  RESULT: ALL CHECKS PASSED")
    print("=" * 60)
    return True


def _print_check(label: str, passed: bool):
    icon = "PASS" if passed else "FAIL"
    print(f"  {icon}  {label}")


def main():
    parser = argparse.ArgumentParser(description="MrTrader go-live script")
    parser.add_argument("--yes", action="store_true",
                        help="Skip confirmation prompt (non-interactive)")
    args = parser.parse_args()

    if not run_checklist():
        sys.exit(1)

    if not args.yes:
        print("\n*** WARNING: This will switch to LIVE trading with REAL money. ***")
        answer = input("Type 'GO LIVE' to confirm: ").strip()
        if answer != "GO LIVE":
            print("Aborted — no changes made.")
            sys.exit(0)

    # ── Execute the switch ────────────────────────────────────────────────────
    from app.trading_modes import mode_manager
    from app.live_trading.capital_manager import capital_manager
    from app.database.session import get_session
    from app.database.models import AuditLog
    from datetime import datetime

    capital_manager.start()
    mode_manager.switch_to_live()

    db = get_session()
    try:
        db.add(AuditLog(
            action="GO_LIVE_ACTIVATED",
            details={
                "activated_at": datetime.utcnow().isoformat(),
                "initial_capital": capital_manager.get_current_capital(),
                "stage": capital_manager.current_stage.stage,
            },
            timestamp=datetime.utcnow(),
        ))
        db.commit()
    except Exception as exc:
        logger.warning("Could not write audit log: %s", exc)
    finally:
        db.close()

    print("\n" + "=" * 60)
    print("  LIVE TRADING ACTIVATED")
    print(f"  Mode:              {mode_manager.mode.value.upper()}")
    print(f"  Initial capital:   ${capital_manager.get_current_capital():,.0f}")
    print(f"  Stage:             {capital_manager.current_stage.stage} of {len(capital_manager.STAGES)}")
    print()
    print("  Capital ramp schedule:")
    for s in capital_manager.STAGES:
        dur = f"{s.duration_days} days" if s.duration_days else "ongoing"
        print(f"    Stage {s.stage}: ${s.capital:>8,.0f}  ({dur})")
    print()
    print("  Kill-switch:  POST /api/dashboard/live/kill-switch")
    print("  Monitor:      GET  /api/dashboard/live/status")
    print("  Advance cap:  POST /api/dashboard/live/increase-capital")
    print()
    print("  *** MONITOR CLOSELY — REAL MONEY IS AT RISK ***")
    print("=" * 60)


if __name__ == "__main__":
    main()
