"""
Paper trading monitor — check readiness for go-live.

Usage:
    python scripts/monitor_paper_trading.py          # check only
    python scripts/monitor_paper_trading.py --watch  # check every 60s
"""
import os
import sys
import time
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.approval_workflow import approval_workflow, GO_LIVE_CRITERIA

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s — %(message)s",
)


def _bar(value: float, max_val: float, width: int = 20, reverse: bool = False) -> str:
    """Simple ASCII progress bar."""
    ratio = min(value / max_val, 1.0) if max_val else 0.0
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}]"


def print_report(is_ready: bool, metrics: dict):
    c = GO_LIVE_CRITERIA
    print("\n" + "=" * 65)
    print("  PAPER TRADING — GO-LIVE READINESS REPORT")
    print("=" * 65)

    rows = [
        ("Duration",       f"{metrics['duration_days']} days",
         metrics["duration_days"] >= c["min_days"],
         f"need {c['min_days']} days"),
        ("Total trades",   str(metrics["total_trades"]),
         metrics["total_trades"] >= c["min_trades"],
         f"need {c['min_trades']}"),
        ("Win rate",       f"{metrics['win_rate_pct']:.1f}%",
         metrics["win_rate_pct"] >= c["min_win_rate"],
         f"need {c['min_win_rate']}%"),
        ("Total return",   f"{metrics['total_return_pct']:.2f}%",
         metrics["total_return_pct"] >= c["min_return_pct"],
         f"need {c['min_return_pct']}%"),
        ("Max drawdown",   f"{metrics['max_drawdown_pct']:.2f}%",
         metrics["max_drawdown_pct"] <= c["max_drawdown_pct"],
         f"max {c['max_drawdown_pct']}%"),
        ("Sharpe ratio",   f"{metrics['sharpe_ratio']:.3f}",
         metrics["sharpe_ratio"] >= c["sharpe_ratio"],
         f"need {c['sharpe_ratio']}"),
    ]

    for label, value, passed, note in rows:
        status = "PASS" if passed else "FAIL"
        print(f"  {label:<20} {value:<12} {status:<6}  ({note})")

    print("-" * 65)
    verdict = "PROCEED TO LIVE TRADING" if is_ready else "CONTINUE PAPER TRADING"
    print(f"  Verdict:  {verdict}")
    print("=" * 65)
    print()


def check_once() -> bool:
    is_ready, metrics = approval_workflow.check_go_live_readiness()
    print_report(is_ready, metrics)
    return is_ready


def main():
    parser = argparse.ArgumentParser(description="Paper trading readiness monitor")
    parser.add_argument("--watch", action="store_true",
                        help="Re-check every 60 seconds")
    parser.add_argument("--interval", type=int, default=60,
                        help="Watch interval in seconds (default 60)")
    args = parser.parse_args()

    if args.watch:
        print(f"Watching paper trading metrics every {args.interval}s — Ctrl+C to stop")
        try:
            while True:
                check_once()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        is_ready = check_once()
        sys.exit(0 if is_ready else 1)


if __name__ == "__main__":
    main()
