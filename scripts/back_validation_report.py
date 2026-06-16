"""P1-4 — on-demand trend intended-vs-actual back-validation report.

Prints the trailing tracking metrics + graduation verdict from the daily snapshots +
rebalance intents recorded by app.live_trading.back_validation. Report-only.

  python -m scripts.back_validation_report [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
from datetime import date

from app.live_trading.back_validation import compute_report, MIN_DAYS_FOR_VERDICT


def _fmt(x, pct=False, nd=3):
    if x is None:
        return "n/a"
    return f"{x*100:+.2f}%" if pct else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    start = date.fromisoformat(a.start) if a.start else None
    end = date.fromisoformat(a.end) if a.end else None

    r = compute_report(start=start, end=end)
    print("=" * 72)
    print(f"TREND BACK-VALIDATION (intended vs actual)   verdict: {r.verdict}")
    print("=" * 72)
    print(f"  window           : {r.window_start} .. {r.window_end}  ({r.n_days} day-pairs)")
    if r.note:
        print(f"  note             : {r.note}")
    if r.verdict in ("BUILDING", "ERROR"):
        if r.verdict == "BUILDING":
            print(f"  building history — need >= {MIN_DAYS_FOR_VERDICT} day-pairs for a verdict.")
        else:
            print("  ERROR — instrument failed to compute (see logs).")
        return
    print(f"  intended-vs-actual corr : {_fmt(r.corr)}")
    print(f"  tracking error          : {_fmt(r.tracking_error_ann, pct=True)} annualized")
    print(f"  drift (actual-intended) : {_fmt(r.drift_ann, pct=True)} annualized")
    print(f"  execution drag          : {_fmt(r.slippage_drag_bps_day, nd=2)} bps/day")
    print(f"  cum actual / intended   : {_fmt(r.actual_cum_return, pct=True)} / "
          f"{_fmt(r.intended_cum_return, pct=True)}  (NAV contribution)")
    print(f"  governor days           : {r.governor_days}   total rebalance blocks: {r.total_blocked}")


if __name__ == "__main__":
    main()
