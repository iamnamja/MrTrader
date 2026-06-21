"""
run_book_state_report.py — Alpha-v10 R0.4b: the consolidated risk-surface report (SHADOW, read-only).

Runs the R0 measurement layer against the LIVE broker account(s) and prints the consolidated
book-state: positions, per-venue accounts, gross/net, the netted FACTOR-EXPOSURE vector, cash-equiv,
and any unmapped / stale-price flags. READ-ONLY — the adapter is structurally incapable of trading;
this answers "what do I hold / net equity beta / margin / reconciled?" before any of it ever gates a
live order (R0.5). IBKR is added as a second adapter in R1.

    PYTHONPATH=. venv/Scripts/python scripts/run_book_state_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    from app.live_trading.broker_adapter import AlpacaReadOnlyAdapter
    from app.live_trading import book_state as bs
    from app.live_trading.risk_policy import RISK_POLICY_V1 as P

    adapters = [AlpacaReadOnlyAdapter()]            # + IBKRReadOnlyAdapter() in R1
    for ad in adapters:
        h = ad.health()
        print(f"[{ad.venue}] connected={h.connected} clock_ok={h.clock_ok} {h.detail}")
    book = bs.build_book_state(adapters)

    print("\n=== CONSOLIDATED BOOK (read-only / shadow) ===")
    print(f"  total NAV            {book.total_nav:>14,.0f}")
    print(f"  gross notional       {book.gross_notional:>14,.0f}   "
          f"({book.gross_ex_cash_frac:.1%} of NAV; policy cap {P.max_gross_ex_cash:.0%})")
    print(f"  net notional         {book.net_notional:>14,.0f}")
    print(f"  cash-equivalents     {book.cash_equiv_value:>14,.0f}   (excluded from gross)")
    for venue, acct in book.accounts.items():
        print(f"  [{venue}] cash {acct.cash:,.0f}  buying_power {acct.buying_power:,.0f}  "
              f"margin_used {acct.margin_used}")

    print("\n=== NETTED FACTOR EXPOSURES (across venues) ===")
    if book.factor_exposures:
        for k, v in sorted(book.factor_exposures.items()):
            unit = bs.FACTOR_UNITS.get(k, "")
            extra = ""
            if k == bs.EQUITY_BETA and book.total_nav:
                extra = f"   ({v / book.total_nav:+.2f} x NAV; policy cap {P.max_net_equity_beta:.2f})"
            print(f"  {k:14s} {v:>+14,.0f}  [{unit}]{extra}")
    else:
        print("  (none)")

    if book.unmapped_factor_instruments:
        print(f"\n  ! UNMAPPED (no factor map -> fail-closed at the gate): "
              f"{book.unmapped_factor_instruments}")
    if book.stale_price_instruments:
        print(f"  ! STALE PRICE (zero mark on a live position): {book.stale_price_instruments}")

    print("\n=== POSITIONS ===")
    for p in sorted(book.positions, key=lambda x: -abs(x.market_value)):
        tag = "cash-eq" if p.is_cash_equivalent else ("UNMAPPED" if not p.mapped else "")
        print(f"  [{p.venue}] {p.broker_symbol:6s} qty={p.quantity:>10.2f}  "
              f"mv={p.market_value:>12,.0f}  notional={p.notional:>12,.0f}  {tag}")

    print("\n(read-only shadow — controls nothing; validates the R0.3/R0.4 surface on live data)")


if __name__ == "__main__":
    main()
