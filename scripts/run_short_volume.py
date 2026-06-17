"""
run_short_volume.py — Alpha-v9 P3-5: the pre-registered aggregate short-volume timing test.

Ensures the FINRA daily short-volume panel is cached (incrementally downloads any missing
days), fetches SPY closes over the same window, and runs the FROZEN verdict
(app/research/short_volume.py). Report-only — promotes nothing.

Usage:
    python -m scripts.run_short_volume
    python -m scripts.run_short_volume --no-refresh        # use cache as-is, skip download
    python -m scripts.run_short_volume --email
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from app.data import finra_short_volume as fsv
from app.research import short_volume as sv


def _spy_closes(start, end):
    from app.data.yfinance_provider import YFinanceProvider
    df = YFinanceProvider().get_daily_bars("SPY", start, end)
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    return df["close"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P3-5 aggregate short-volume timing test")
    ap.add_argument("--no-refresh", action="store_true",
                    help="use the cached panel as-is (skip the incremental download)")
    ap.add_argument("--max-days", type=int, default=None,
                    help="bound this run's downloads (resume-friendly backfill)")
    ap.add_argument("--email", action="store_true", help="send a phase_complete email")
    args = ap.parse_args(argv)

    if not args.no_refresh:
        print("Refreshing FINRA short-volume panel (incremental)...")
        fsv.build_panel(max_days=args.max_days)
    status = fsv.cache_status()
    print(f"Panel: {status['n_days']} days  ({status['first']} -> {status['last']})")

    ratio = fsv.load_aggregate_ratio()
    if ratio.empty:
        print("No cached short-volume data — aborting.")
        return 1

    start = ratio.index.min().date()
    end = datetime.now(timezone.utc).date()
    spy = _spy_closes(start, end)

    v = sv.short_volume_verdict(ratio, spy)
    print("\n" + "=" * 74)
    print(f"P3-5 AGGREGATE SHORT-VOLUME TIMING  ({v.registration_id})")
    print("=" * 74)
    print(f"  evaluated days       : {v.n_days}")
    print(f"  params               : window={int(v.params['window'])} "
          f"z>{v.params['z_threshold']} cost={v.params['cost_bps']}bps/side")
    print(f"  overlay Sharpe (net) : {v.overlay_sharpe:+.3f}")
    print(f"  overlay HAC p (1-sided): {v.overlay_hac_p:.3f}")
    print(f"  SPY buy-hold Sharpe  : {v.buyhold_sharpe:+.3f}")
    print(f"  opposite-direction SR: {v.opposite_overlay_sharpe:+.3f}  (diagnostic only)")
    print("  -- robustness (standalone edge vs timed SPY beta) --")
    print(f"  residual alpha vs SPY: {v.incr_alpha_ann:+.2%}/yr  HAC t {v.incr_alpha_t:+.2f}  "
          f"(beta {v.beta_spy:.2f})")
    print(f"  sub-period delta vs buy-hold: H1 {v.h1_delta:+.3f}  H2 {v.h2_delta:+.3f}  "
          f"{'(STABLE)' if v.h1_delta > 0 and v.h2_delta > 0 else '(UNSTABLE)'}")
    print(f"  standalone edge?     : {v.standalone_edge}")
    fwd = v.fwd_ret_by_z_tercile
    print("  next-day SPY ret by short-vol z tercile (diagnostic):")
    print(f"    low z  : {fwd.get('low', float('nan')):+.4%}")
    print(f"    mid z  : {fwd.get('mid', float('nan')):+.4%}")
    print(f"    high z : {fwd.get('high', float('nan')):+.4%}   "
          f"(informed-short prior: high < low)")
    print("\n" + "-" * 74)
    print(f"VERDICT: {v.verdict}")
    print(f"  {v.reason}")
    print("-" * 74 + "\n")

    if args.email:
        try:
            from app.notifications import notifier
            notifier.enqueue("phase_complete", {
                "phase": "P3-5 (aggregate short-volume timing)",
                "tasks_done": f"Built the FINRA daily short-volume provider + pre-registered "
                              f"aggregate-timing test on {v.n_days} days.",
                "outcome": f"{v.verdict}: {v.reason}",
                "next_phase": "cross-sectional short-volume post-Norgate (survivorship-free)",
                "notes": f"overlay Sharpe {v.overlay_sharpe:+.2f} (HAC p {v.overlay_hac_p:.3f}) "
                         f"vs buy-hold {v.buyhold_sharpe:+.2f}",
            })
            print("phase_complete email enqueued.")
        except Exception as exc:
            print(f"email enqueue failed (non-fatal): {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
