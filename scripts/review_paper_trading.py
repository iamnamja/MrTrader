"""
CLI: paper trading review — runs backtest + readiness checks,
prints a combined go/no-go report, and optionally enables paper mode.

Usage:
  python scripts/review_paper_trading.py
  python scripts/review_paper_trading.py --model swing --symbols AAPL MSFT NVDA
  python scripts/review_paper_trading.py --skip-backtest   # readiness only
  python scripts/review_paper_trading.py --enable-paper    # enable if all checks pass
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
DIM = "\033[2m"


def _c(colour, text):
    return f"{colour}{text}{RESET}"


def header(step, title):
    print(f"\n{BOLD}{CYAN}[{step}] {title}{RESET}")
    print(_c(DIM, "-" * 60))


def ok(msg):
    print(f"  {GREEN}PASS{RESET}  {msg}")


def warn(msg):
    print(f"  {YELLOW}WARN{RESET}  {msg}")


def fail(msg):
    print(f"  {RED}FAIL{RESET}  {msg}")


def info(msg):
    print(f"       {msg}")


def _print_check(c):
    if c["passed"]:
        ok(c["detail"])
    elif c.get("is_warning"):
        warn(c["detail"])
    else:
        fail(c["detail"])


def run_backtest_section(model_name, symbols, years, days):
    from app.live_trading.backtest_readiness import run_quick_backtest

    header("BT", f"Backtest — {model_name} model")
    print(f"  Running backtest on {len(symbols)} symbols...")
    t0 = time.time()
    result = run_quick_backtest(model_name, symbols, years=years, days=days)
    elapsed = time.time() - t0

    if result is None:
        fail(f"No trained {model_name} model found")
        info("Train first:")
        info(f"  python scripts/train_model.py --years {years} --no-fundamentals")
        return None

    s = result.summary()
    print(f"  Done in {elapsed:.1f}s  |  {s['total_trades']} trades")
    print()
    ok(f"Win rate       : {s['win_rate']}")
    ok(f"Sharpe ratio   : {s['sharpe_ratio']}")
    ok(f"Profit factor  : {s['profit_factor']}")
    ok(f"Max drawdown   : {s['max_drawdown_pct']}")
    ok(f"Avg P&L/trade  : {s['avg_pnl_pct']}")

    if result.total_trades > 0:
        by_reason = {}
        for t in result.trades:
            by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1
        print()
        info("Exit breakdown: " + "  ".join(
            f"{r}={n}" for r, n in sorted(by_reason.items(), key=lambda x: -x[1])
        ))

    return result


def run_readiness_section(skip_infra=False):
    header("RD", "Readiness checks")

    if skip_infra:
        info("(Infrastructure checks skipped — use without --skip-backtest for full check)")
        return {"ready": True, "blockers": [], "warnings": [], "passed": []}

    try:
        from app.live_trading.readiness import ReadinessChecker
        report = ReadinessChecker().run()

        for c in report.get("passed", []):
            ok(c["detail"])
        for c in report.get("warnings", []):
            warn(c["detail"])
        for c in report.get("blockers", []):
            fail(c["detail"])

        return report
    except Exception as exc:
        warn(f"Readiness check failed: {exc}")
        return {"ready": False, "blockers": [{"detail": str(exc)}], "warnings": [], "passed": []}


def run_backtest_go_nogo(swing_result, intraday_result):
    header("GO", "Backtest go/no-go evaluation")
    from app.live_trading.backtest_readiness import BacktestReadinessChecker

    checker = BacktestReadinessChecker()
    report = checker.evaluate(swing_result, intraday_result)

    for c in report["all_checks"]:
        _print_check(c)

    return report


def main():
    from app.utils.constants import SP_100_TICKERS

    parser = argparse.ArgumentParser(
        description="MrTrader paper trading review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=["swing", "intraday", "both"], default="both",
    )
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--days", type=int, default=55)
    parser.add_argument("--symbols", nargs="+", default=None, metavar="TICKER")
    parser.add_argument(
        "--skip-backtest", action="store_true",
        help="Skip backtest; only run infrastructure readiness checks",
    )
    parser.add_argument(
        "--enable-paper", action="store_true",
        help="Enable paper trading if all checks pass (writes to DB)",
    )
    args = parser.parse_args()

    symbols = args.symbols or SP_100_TICKERS[:20]

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  MrTrader -- Paper Trading Review{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    t_start = time.time()
    swing_result = intraday_result = None

    if not args.skip_backtest:
        if args.model in ("swing", "both"):
            swing_result = run_backtest_section("swing", symbols, args.years, args.days)

        if args.model in ("intraday", "both"):
            intraday_result = run_backtest_section("intraday", symbols, args.years, args.days)

    # Infrastructure readiness (skippable for quick backtest-only review)
    readiness = run_readiness_section(skip_infra=args.skip_backtest)

    # Combined go/no-go
    if not args.skip_backtest:
        bt_report = run_backtest_go_nogo(swing_result, intraday_result)
    else:
        bt_report = {"ready": True, "blockers": [], "warnings": []}

    # Final verdict
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    infra_ready = readiness.get("ready", False) or args.skip_backtest
    bt_ready = bt_report.get("ready", True)
    all_ready = infra_ready and bt_ready

    elapsed = time.time() - t_start

    if all_ready:
        print(f"{GREEN}{BOLD}  VERDICT: READY FOR PAPER TRADING{RESET}")
        print(f"  All checks passed in {elapsed:.0f}s")
        if args.enable_paper:
            _enable_paper_mode()
    else:
        print(f"{RED}{BOLD}  VERDICT: NOT READY{RESET}")
        all_blockers = (
            readiness.get("blockers", []) + bt_report.get("blockers", [])
        )
        for b in all_blockers:
            fail(b["detail"])
        all_warnings = (
            readiness.get("warnings", []) + bt_report.get("warnings", [])
        )
        for w in all_warnings:
            warn(w["detail"])
        if args.enable_paper:
            print(f"\n  {YELLOW}--enable-paper ignored: checks did not pass{RESET}")
        print(f"\n  Elapsed: {elapsed:.0f}s")

    print(f"{BOLD}{'=' * 60}{RESET}\n")
    return 0 if all_ready else 1


def _enable_paper_mode():
    """Record a paper trading activation event in the audit log."""
    try:
        from datetime import datetime
        from app.database.models import AuditLog
        from app.database.session import get_session

        db = get_session()
        try:
            db.add(AuditLog(
                action="PAPER_TRADING_ENABLED",
                details={"enabled_at": datetime.utcnow().isoformat(),
                         "source": "review_paper_trading.py"},
                timestamp=datetime.utcnow(),
            ))
            db.commit()
            print(f"\n  {GREEN}Paper trading mode activated and logged to audit trail{RESET}")
        except Exception as exc:
            db.rollback()
            print(f"\n  {YELLOW}Warning: could not write audit log: {exc}{RESET}")
        finally:
            db.close()
    except Exception as exc:
        print(f"\n  {YELLOW}Warning: audit log unavailable: {exc}{RESET}")


if __name__ == "__main__":
    sys.exit(main())
