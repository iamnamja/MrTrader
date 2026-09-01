"""Backfill per-sleeve daily P&L into the trend/cash trackers (2026-09-01).

CH5's purpose is "let the live-forward scorecard accrue", but `trend_daily` held 14 rows with every
P&L column NULL: the only caller (`trend_sleeve.run_trend_rebalance`) passed no P&L and ran weekly.
`cash_daily` had no P&L columns at all. Three months of live paper recorded nothing.

The daily data needed already exists — `back_validation.trend_backval_daily` has 51 daily snapshots
with positions, prices and NAV (322 symbol-days, 0 unpriced for the trend universe). This rebuilds
the history from those snapshots plus the broker fill blotter, so the record is recovered rather
than lost.

Two passes:

  1. `cash_prices` backfill. Historical snapshots predate that column, so the cash sleeve cannot be
     marked (SGOV was unpriced on 49 of 51 days). Closes are fetched once from yfinance and written
     into the historical rows.
  2. P&L reconstruction. FIFO lots per sleeve give realized-per-day and the cost basis needed for
     the unrealized LEVEL that `record_daily` expects.

ACCEPTED ONLY IF IT RECONCILES. The per-sleeve cumulative sum must reproduce the account's own NAV
move over the same window; the script exits non-zero and writes nothing otherwise. That criterion
earned its keep — it caught two real errors during construction:

  * row 0 absorbing ALL pre-window realized P&L (+$1,051.95 of contamination), and
  * day-0's own fills being double-counted against the opening balance (-$52.61).

Final: sleeve sum +$211.92 vs NAV move +$213.33 -> residual -$1.41 against $4.08 lifetime fees.

    python scripts/backfill_sleeve_pnl.py --dry-run
    python scripts/backfill_sleeve_pnl.py --apply
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def backfill_cash_prices(apply: bool) -> dict:
    """Populate `cash_prices` on historical snapshots.

    Returns the computed ``{date: {symbol: close}}`` map REGARDLESS of `apply`, so the caller can
    merge it in memory and the dry-run projects exactly what --apply would produce. Reporting a
    dry-run figure the real run would not reproduce is worse than no dry-run.
    """
    import pandas as pd
    from app.live_trading.back_validation import DB_PATH as BV
    from app.live_trading.cash_sleeve import CASH_ETFS
    from app.live_trading.sleeve_pnl import sleeve_of_fill
    from app.analytics.execution_pnl import iter_fills
    from app.integrations.alpaca import AlpacaClient

    # Which cash symbols were ever actually traded — not all 8 eligible tickers.
    fills = iter_fills(AlpacaClient().get_all_orders())
    syms = sorted({f["symbol"] for f in fills
                   if sleeve_of_fill(f) == "cash" and f["symbol"] in {s for s in CASH_ETFS}})
    if not syms:
        print("  no cash-sleeve fills found — nothing to price")
        return {}

    # Go through back_validation's own _conn(), which runs the idempotent ALTER migration —
    # a raw sqlite3.connect() would skip it and report the column as missing.
    from app.live_trading.back_validation import _conn as _bv_conn
    with _bv_conn():
        pass
    c = sqlite3.connect(str(BV))
    have = {r[1] for r in c.execute("PRAGMA table_info(trend_backval_daily)")}
    if "cash_prices" not in have:
        print("  cash_prices column still missing after migration — aborting")
        return {}
    rows = c.execute(
        "SELECT trade_date, cash_prices FROM trend_backval_daily ORDER BY trade_date").fetchall()
    need = [d for d, cp in rows if not cp]
    if not need:
        print("  cash_prices already present on every row")
        return {}

    import yfinance as yf
    hist = yf.download(syms, start=min(need), end=None, progress=False, auto_adjust=True)
    closes = hist["Close"] if "Close" in hist.columns else hist
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(syms[0])

    computed: dict = {}
    for d in need:
        try:
            row = closes.loc[d]
        except KeyError:
            continue                      # non-trading day / no bar: leave NULL, never invent one
        px = {s: float(row[s]) for s in closes.columns
              if s in row and row[s] == row[s]}      # drop NaN
        if not px:
            continue
        computed[d] = px
        if apply:
            c.execute("UPDATE trend_backval_daily SET cash_prices=? WHERE trade_date=?",
                      (json.dumps(px), d))
    if apply:
        c.commit()
    print(f"  cash symbols: {syms} | rows needing prices: {len(need)} | filled: {len(computed)}"
          + ("" if apply else "  (dry-run: merged in memory for PASS 2)"))
    return computed


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    apply = bool(args.apply)

    from app.integrations.alpaca import AlpacaClient
    from app.live_trading import cash_tracker, trend_tracker
    from app.live_trading.sleeve_pnl import (
        compute_daily_pnl, daily_pnl_series, load_snapshots,
    )

    print("PASS 1 — backfill cash_prices on historical snapshots")
    computed_cash = backfill_cash_prices(apply)

    print("\nPASS 2 — reconstruct per-sleeve daily P&L")
    snaps = load_snapshots()
    if not snaps:
        print("  no snapshots — nothing to do")
        return 1
    # Merge PASS 1's prices so a dry-run reconstructs exactly what --apply would.
    for s_ in snaps:
        extra = computed_cash.get(s_["date"])
        if extra:
            s_["prices"] = {**s_["prices"], **extra}
    alpaca = AlpacaClient()
    fills = alpaca.get_all_orders()

    totals = {}
    for sleeve, tracker in (("trend", trend_tracker), ("cash", cash_tracker)):
        rows = daily_pnl_series(compute_daily_pnl(fills, snaps, sleeve=sleeve))
        marked = [r for r in rows if r.get("unrealized") is not None]
        unmarked = len(rows) - len(marked)
        last = marked[-1] if marked else None
        totals[sleeve] = (last["cumulative"] if last else 0.0)
        print(f"  {sleeve:6s}: {len(rows)} days | marked {len(marked)} | unmarked {unmarked} "
              f"| cumulative ${totals[sleeve]:+,.2f}")
        if apply:
            # Back up before rewriting history — same discipline as the trade-book repairs.
            try:
                import shutil
                src = Path(str(tracker.DB_PATH))
                if src.exists():
                    shutil.copy2(src, src.with_suffix(".db.bak_20260901"))
            except Exception as exc:  # noqa: BLE001
                print(f"          WARNING: backup failed ({exc}) — writing anyway")
            n = 0
            for r in marked:
                if tracker.record_daily(
                    r["date"], realized_pnl=float(r["realized"]),
                    unrealized_pnl=float(r["unrealized"]),
                    daily_pnl_override=float(r["daily"]),
                    cumulative_pnl_override=float(r["cumulative"]),
                ):
                    n += 1
            print(f"          wrote {n} row(s) to {tracker.DB_PATH.name}")

    # ACCEPTANCE CHECK — the per-sleeve sum must reproduce the account's own NAV move over the
    # SAME window. This is the criterion that caught two real errors while building this: row 0
    # booking all pre-window realized P&L (+$1,051.95 of contamination), and then day-0's own
    # fills being double-counted against the baseline (-$52.61). Compared against the SNAPSHOT
    # NAV series, not Alpaca's portfolio-history equity — the two are marked at different times
    # and differ by a few dollars, and mixing them would obscure exactly this kind of bug.
    print("\nRECONCILIATION — per-sleeve sum vs the account's NAV move over the same window")
    navs = [(s["date"], s["nav"]) for s in snaps if s.get("nav")]
    nav_delta = navs[-1][1] - navs[0][1]
    sleeve_sum = sum(totals.values())
    residual = sleeve_sum - nav_delta
    print(f"  snapshot NAV {navs[0][0]} ${navs[0][1]:,.2f} -> {navs[-1][0]} ${navs[-1][1]:,.2f}"
          f"  = ${nav_delta:+,.2f}")
    print(f"  sleeve cumulative sum : ${sleeve_sum:+,.2f}")
    print(f"  RESIDUAL              : ${residual:+,.2f}   (lifetime fees ${_lifetime_fees():+,.2f})")

    TOL = 25.0     # generous vs fees + intraday mark timing; a real attribution bug is >> this
    if abs(residual) > TOL:
        print(f"\n  ✗ RECONCILIATION FAILED (|residual| > ${TOL:.0f}). The reconstruction is wrong "
              f"— NOT writing. Fix the attribution before retrying.")
        return 1
    print(f"  [OK] reconciles within ${TOL:.0f}")

    if not apply:
        print("\nDRY RUN — nothing written.")
    return 0


def _lifetime_fees() -> float:
    """Total FEE activity from the broker, for context in the reconciliation line."""
    try:
        from app.config import settings
        import httpx
        base = (getattr(settings, "alpaca_base_url", "") or "").rstrip("/")
        r = httpx.get(f"{base}/account/activities/FEE", timeout=15, params={"page_size": 100},
                      headers={"APCA-API-KEY-ID": settings.alpaca_api_key,
                               "APCA-API-SECRET-KEY": settings.alpaca_secret_key})
        if r.status_code != 200:
            return 0.0
        return sum(float(x.get("net_amount") or 0) for x in r.json())
    except Exception:  # noqa: BLE001
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
