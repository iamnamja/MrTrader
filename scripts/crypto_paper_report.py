"""
crypto_paper_report.py — show the P3-1 crypto trend LIVE-PAPER OOS track.

Recomputes the rules-based crypto-trend book on live Alpaca closes, freezes the forward
out-of-sample slice (from inception), and prints Sharpe-to-date vs the 0.64 backtest.
Report-only — no orders, no capital.

Usage:
    python -m scripts.crypto_paper_report
    python -m scripts.crypto_paper_report --email
"""
from __future__ import annotations

import argparse

from app.live_trading import crypto_paper_track as cpt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P3-1 crypto live-paper OOS report")
    ap.add_argument("--email", action="store_true", help="also enqueue the weekly email")
    args = ap.parse_args(argv)

    s = cpt.run_crypto_paper_track(force=True)
    print("\n" + "=" * 70)
    print("P3-1 CRYPTO TREND — LIVE-PAPER OOS TRACK  (report-only, no capital)")
    print("=" * 70)
    print(f"  status              : {s.get('status')}")
    print(f"  enabled             : {s.get('enabled')}")
    print(f"  OOS inception       : {s.get('inception')}")
    print(f"  OOS trading days    : {s.get('n_oos_days')}")
    sharpe = s.get("oos_sharpe")
    print(f"  OOS Sharpe-to-date  : {sharpe:+.3f}" if isinstance(sharpe, (int, float))
          else f"  OOS Sharpe-to-date  : {sharpe}")
    print(f"  backtest expectation: +{s.get('backtest_sharpe')}  (365-ann)")
    cum = s.get("oos_cum")
    print(f"  OOS cumulative ret  : {cum:+.2%}" if isinstance(cum, (int, float))
          else f"  OOS cumulative ret  : {cum}")
    if s.get("n_oos_days", 0) < 15:
        print("\n  (BUILDING — needs ~15+ OOS days before the Sharpe is meaningful.)")
    print("=" * 70 + "\n")

    if args.email:
        cpt.weekly_email(s)
        print("weekly email enqueued.")
    return 0 if s.get("status") in ("ok", "dormant") else 1


if __name__ == "__main__":
    raise SystemExit(main())
