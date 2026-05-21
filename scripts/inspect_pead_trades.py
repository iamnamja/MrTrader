"""
Manual trade inspection: spot-check PEAD trades for look-ahead or data integrity issues.

For each sampled trade, verifies:
1. Entry date is AFTER the earnings announcement date (not same-day)
2. Surprise was above threshold at entry (signal was valid on entry day)
3. Days-since-earnings at entry is within max_days_after window

Usage:
    python scripts/inspect_pead_trades.py
"""
import sys
import logging
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING)

INSPECT_N = 20    # number of trades to inspect
LONG_THRESHOLD = 0.05
MAX_DAYS_AFTER = 3


def main():
    import pandas as pd
    import yfinance as yf
    from app.data.fmp_provider import get_earnings_features_at
    from app.ml.pead_scorer import PEADScorer

    scorer = PEADScorer(long_threshold=LONG_THRESHOLD, max_days_after=MAX_DAYS_AFTER, long_short=True)

    # Use a fixed recent test window (Fold 5 of the WF)
    test_start = date(2025, 6, 1)
    test_end = date(2026, 5, 20)

    # Pull a sample of symbols that had PEAD signals in this period
    from app.utils.constants import RUSSELL_1000_TICKERS
    sample_syms = list(RUSSELL_1000_TICKERS)[:80]  # first 80 for speed

    print(f"Inspecting PEAD signals {test_start} -> {test_end}")
    print("=" * 70)

    # Download bars for sample
    bars = {}
    for sym in sample_syms:
        try:
            df = yf.download(sym, start=str(test_start - timedelta(days=10)),
                             end=str(test_end), progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            if len(df) >= 5:
                bars[sym] = df
        except Exception:
            pass

    print(f"Loaded {len(bars)} symbols\n")

    trades_found = []
    trading_days = pd.bdate_range(start=test_start, end=test_end)

    for day in trading_days:
        d = day.date()
        day_bars = {s: df for s, df in bars.items() if not df.empty}
        signals = scorer(d, day_bars)
        for sym, conf, direction in signals:
            feats = get_earnings_features_at(sym, d)
            if feats is None:
                continue
            # Compute report_date from PIT-filtered view (same data scorer sees)
            days_since_f = feats.get("fmp_days_since_earnings", 0)
            report_date = d - timedelta(days=int(days_since_f))

            trades_found.append({
                "entry_day": d,
                "symbol": sym,
                "direction": direction,
                "confidence": conf,
                "report_date": report_date,
                "days_since": int(days_since_f),
                "surprise": feats.get("fmp_surprise_1q", 0),
            })

            if len(trades_found) >= INSPECT_N:
                break
        if len(trades_found) >= INSPECT_N:
            break

    if not trades_found:
        print("No PEAD signals found in sample — check universe or threshold")
        return 1

    issues = []
    print(f"{'#':<3} {'Symbol':<6} {'Dir':<5} {'Entry':<12} {'Report':<12} "
          f"{'Days':<6} {'Surprise':>9} {'OK?':<5}")
    print("-" * 70)
    for i, t in enumerate(trades_found, 1):
        # Check 1: entry must be AFTER report (days_since >= 1)
        ok1 = t["days_since"] >= 1
        # Check 2: days within window
        ok2 = 1 <= t["days_since"] <= MAX_DAYS_AFTER
        # Check 3: surprise above threshold
        ok3 = (abs(t["surprise"]) >= LONG_THRESHOLD)
        ok = ok1 and ok2 and ok3
        if not ok:
            issues.append((i, t, ok1, ok2, ok3))
        flag = "OK" if ok else "FAIL"
        print(f"{i:<3} {t['symbol']:<6} {t['direction']:<5} {str(t['entry_day']):<12} "
              f"{str(t['report_date']):<12} {t['days_since']:<6} "
              f"{t['surprise']:>9.3f} {flag}")

    print()
    if issues:
        print(f"ISSUES FOUND ({len(issues)} of {len(trades_found)} trades):")
        for i, t, ok1, ok2, ok3 in issues:
            reasons = []
            if not ok1:
                reasons.append(f"days_since={t['days_since']} (must be >=1, entry AFTER report)")
            if not ok2:
                reasons.append(f"days_since={t['days_since']} outside [1,{MAX_DAYS_AFTER}]")
            if not ok3:
                reasons.append(f"surprise={t['surprise']:.3f} below threshold {LONG_THRESHOLD}")
            print(f"  Trade {i} {t['symbol']}: {', '.join(reasons)}")
        return 1

    print(f"MANUAL INSPECTION PASSED -- {len(trades_found)} trades checked, 0 issues")
    print(f"  - All entries are 1-{MAX_DAYS_AFTER} days after earnings announcement")
    print(f"  - All surprises exceed |{LONG_THRESHOLD}| threshold")
    print("  - No same-day look-ahead detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
