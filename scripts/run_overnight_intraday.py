"""
run_overnight_intraday.py — Alpha-v9 P3-3: the pre-registered overnight-vs-intraday
net-of-cost test.

Fetches deep-history daily OHLC for the liquid ETF universe, decomposes each name's
return into its overnight (close->open) and intraday (open->close) legs, and runs the
FROZEN net-of-cost verdict (see app/research/overnight_intraday.py). Report-only —
promotes nothing.

Usage:
    python -m scripts.run_overnight_intraday
    python -m scripts.run_overnight_intraday --start 2002-01-01 --symbols SPY,QQQ,IWM
    python -m scripts.run_overnight_intraday --email
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

from app.research import overnight_intraday as oi


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _fetch(symbols, start: date):
    from app.data.yfinance_provider import YFinanceProvider
    bulk = YFinanceProvider().get_daily_bars_bulk(list(symbols), start, _today())
    out = {}
    for sym in symbols:
        df = bulk.get(sym)
        if df is None or df.empty:
            print(f"  WARNING: no bars for {sym} — skipping")
            continue
        d = df.copy()
        d.columns = [str(c).lower() for c in d.columns]
        if "open" in d.columns and "close" in d.columns:
            out[sym] = d
        else:
            print(f"  WARNING: {sym} bars lack open/close — skipping")
    return out


def _print_report(v: oi.OvernightIntradayVerdict) -> None:
    print("\n" + "=" * 78)
    print(f"P3-3 OVERNIGHT vs INTRADAY DECOMPOSITION  ({v.registration_id})")
    print("=" * 78)
    print(f"Universe: {', '.join(v.universe)}")
    print(f"Realistic cost: {v.realistic_cost_bps_per_side:.1f} bps/side "
          f"(= {2 * v.realistic_cost_bps_per_side:.1f} bps/day round-trip)\n")

    def _row(leg: oi.LegStats):
        print(f"  {leg.label:20s}  n={leg.n_days:5d}  "
              f"gross Sharpe {leg.gross_sharpe:+.2f}  gross CAGR {leg.gross_cagr:+.2%}  "
              f"annVol {leg.ann_vol:.1%}  ||  NET Sharpe {leg.net_sharpe:+.2f}  "
              f"NET CAGR {leg.net_cagr:+.2%}")

    print("Equal-weight universe legs:")
    _row(v.overnight)
    _row(v.intraday)

    print("\nCost cliff — overnight NET Sharpe by cost (bps/side):")
    for c in sorted(v.cost_cliff):
        print(f"    {c:4.1f} bps/side ({2 * c:4.1f}/day) -> net Sharpe {v.cost_cliff[c]:+.2f}")

    print("\nPer-symbol (net @ realistic cost):")
    for sym, m in v.per_symbol.items():
        print(f"    {sym:5s}  overnight net SR {m['overnight_net_sharpe']:+.2f}  "
              f"intraday net SR {m['intraday_net_sharpe']:+.2f}  "
              f"(overnight gross CAGR {m['overnight_gross_cagr']:+.2%}, n={m['n_days']})")

    print("\n" + "-" * 78)
    print(f"VERDICT: {v.verdict}")
    print(f"  {v.reason}")
    print("-" * 78 + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P3-3 overnight-vs-intraday net-of-cost test")
    ap.add_argument("--start", default="2007-01-01", help="history start (YYYY-MM-DD)")
    ap.add_argument("--symbols", default=",".join(oi.DEFAULT_UNIVERSE),
                    help="comma-separated universe")
    ap.add_argument("--cost", type=float, default=oi.REALISTIC_COST_BPS_PER_SIDE,
                    help="realistic cost in bps per side")
    ap.add_argument("--email", action="store_true", help="send a phase_complete email")
    args = ap.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    print(f"Fetching daily OHLC for {len(symbols)} symbols from {start} ...")
    bars = _fetch(symbols, start)
    if not bars:
        print("No usable bars fetched — aborting.")
        return 1

    v = oi.overnight_intraday_verdict(bars, realistic_cost=args.cost)
    _print_report(v)

    if args.email:
        try:
            from app.notifications import notifier
            notifier.enqueue("phase_complete", {
                "phase": "P3-3 (overnight vs intraday)",
                "tasks_done": f"Decomposed {len(bars)} ETFs into overnight/intraday legs; "
                              f"ran the frozen net-of-cost test.",
                "outcome": f"{v.verdict}: {v.reason}",
                "next_phase": "enable crypto live-paper",
                "notes": f"overnight net Sharpe {v.overnight.net_sharpe:+.2f} vs intraday "
                         f"{v.intraday.net_sharpe:+.2f} @ {args.cost}bps/side",
            })
            print("phase_complete email enqueued.")
        except Exception as exc:
            print(f"email enqueue failed (non-fatal): {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
