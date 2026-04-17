#!/usr/bin/env python
"""
Pre-flight live trading readiness check.

Usage:
    python scripts/check_live_readiness.py

Exits 0 if all blockers pass, 1 if any blocker fails.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.live_trading.readiness import ReadinessChecker

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"
CHECK = "✓"
CROSS = "✗"
WARN = "⚠"


def main():
    print(f"\n{BOLD}MrTrader — Live Trading Readiness Check{RESET}")
    print("=" * 50)

    checker = ReadinessChecker()
    report = checker.run()

    print(f"\nTimestamp : {report['timestamp']}")
    print(f"Summary   : {report['summary']}")
    print()

    for item in report["all_checks"]:
        if item["passed"]:
            icon = f"{GREEN}{CHECK}{RESET}"
            color = GREEN
        elif item["check"] in ("smtp_configured", "slack_configured"):
            icon = f"{YELLOW}{WARN}{RESET}"
            color = YELLOW
        else:
            icon = f"{RED}{CROSS}{RESET}"
            color = RED

        val = f" [{item['value']}]" if item["value"] is not None else ""
        print(f"  {icon}  {item['check']}{val}")
        print(f"       {color}{item['detail']}{RESET}")
        print()

    print("=" * 50)
    if report["ready"]:
        print(f"{GREEN}{BOLD}✓ READY FOR LIVE TRADING{RESET}")
        print(f"\nNext steps:")
        print("  1. Change TRADING_MODE=live in your .env")
        print("  2. Update ALPACA_BASE_URL=https://api.alpaca.markets")
        print("  3. Restart the app")
        print("  4. Monitor closely for the first hour")
    else:
        print(f"{RED}{BOLD}✗ NOT READY — fix the blockers above first{RESET}")
        print(f"\n{len(report['blockers'])} blocker(s) must be resolved:")
        for b in report["blockers"]:
            print(f"  • {b['check']}: {b['detail']}")

    if report["warnings"]:
        print(f"\n{YELLOW}Warnings (non-blocking):{RESET}")
        for w in report["warnings"]:
            print(f"  ⚠  {w['check']}: {w['detail']}")

    print()
    sys.exit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
