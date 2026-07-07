"""FB0 (Futures Breadth Program) — how much breadth, and WHICH markets, restore the carry+xSMOM book's
Track-B t (diversification vs live ETF-trend) from the 16-market −0.20 toward the 76-market 2.61?

Loads the full-universe panels ONCE, then evaluates the book on column-subsets (each market's carry/mom
signal is per-market; the cross-section forms inside carry_backtest, so subsetting columns = the book on
that sub-universe). Outputs: anchors, a t-vs-#markets curve (random subsets), an asset-class bucket
analysis (16 + each family), and a greedy efficient-expansion path (minimum markets to clear t>2)."""
import numpy as np
import pandas as pd

from app.live_trading import instrument_master as im
from app.research import futures_carry as fc
from app.research import futures_data as fd
from app.research import futures_factors as ff
from app.research import futures_roll as frl
from app.research.null_zoo import track_b_stat
from scripts.walkforward.sleeves import live_trend_book_returns

CFG = fc.CarryConfig(roll_cost_bps=3.0)

# Asset-class buckets (by futures code; markets not listed fall in "other").
BUCKETS = {
    "us_equity_idx": ["ES", "NQ", "RTY", "EMD", "MES", "MNQ", "M2K", "MYM", "YM"],
    "intl_equity_idx": ["FDAX", "FESX", "FCE", "FSMI", "FTDX", "HSI", "MHI", "NKD", "NIY", "SXF",
                        "SCN", "SSG", "KOS", "YAP", "SNK", "NKY"],  # noqa: E128
    "us_rates": ["ZN", "ZB", "ZF", "ZT", "TN", "UB", "ZQ", "SR3", "SO3"],
    "intl_rates": ["FGBL", "FGBM", "FGBS", "FGBX", "FBTP", "FOAT", "CGB", "LFT", "YIB", "YIR", "YXT",
                   "YYT", "SJB", "AFB", "AWM", "CRA", "HTW"],
    "fx": ["6E", "6J", "6A", "6B", "6C", "6M", "6N", "6S", "DX"],
    "energy": ["CL", "NG", "HO", "RB", "BRN", "GAS", "EUA", "WBS", "LLG", "LRC", "LSU", "LCC", "LWB"],
    "metals": ["GC", "SI", "HG", "PA", "PL", "GD", "MET"],
    "grains": ["ZC", "ZS", "ZL", "ZM", "ZW", "ZO", "KE", "MWE", "RS"],
    "softs": ["CC", "CT", "KC", "SB", "OJ", "LBR"],
    "livestock": ["LE", "HE", "GF"],
    "crypto": ["BTC", "ETH", "MBT"],
    "vol": ["VX"],
}
IBKR16 = sorted({str(inst.root).upper() for inst in im.futures_instruments().values()})

_PANELS = {}


def _load():
    if _PANELS:
        return _PANELS
    uni = fd.liquid_universe()
    rets = fd.returns_panel(uni)
    _PANELS.update(uni=uni, rets=rets, carry=fc.carry_panel(uni),
                   prices=fd.synthetic_price_panel(uni),   # momentum is RECOMPUTED per subset (below)
                   roll=frl.roll_days_panel(uni, index=rets.index),
                   base=live_trend_book_returns())
    return _PANELS


def book_t(cols):
    """Track-B t of the carry+xsmom book restricted to `cols` (markets present in the panels)."""
    p = _load()
    cols = [c for c in cols if c in p["rets"].columns]
    if len(cols) < CFG.min_xs_width:
        return float("nan")
    # Re-derive the SUBSET's own index: drop rows where none of these markets trade, so the rebalance
    # grid + vol windows match returns_panel(cols) exactly (the faithful "trade only these" book).
    # xs_momentum uses POSITIONAL shifts, so it MUST be recomputed on the subset's own price grid
    # (slicing a full-universe momentum panel would span a different calendar lookback — Opus FB0 review).
    r = p["rets"][cols].dropna(how="all")
    mo = ff.xs_momentum_signal(p["prices"][cols].dropna(how="all"))
    c = fc.carry_backtest(r, p["carry"][cols], CFG, roll_days=p["roll"][cols])
    x = ff.xs_factor_backtest(r, mo, CFG, roll_days=p["roll"][cols])
    j = pd.concat([c.rename("c"), x.rename("x")], axis=1, join="inner").dropna()
    if j.empty:
        return float("nan")
    t, _ = track_b_stat((0.5 * j["c"] + 0.5 * j["x"]), p["base"])
    return t


def main():
    p = _load()
    full = [c.upper() for c in p["rets"].columns]
    ibkr = [m for m in full if m in IBKR16]
    print(f"panels: {len(full)} markets | IBKR-tradeable present: {len(ibkr)}")
    print(f"\nANCHORS:  t(IBKR-{len(ibkr)}) = {book_t(ibkr):+.2f}   t(FULL-{len(full)}) = {book_t(full):+.2f}   (bar t>2)")

    # (b) t-vs-#markets curve — random subsets, avg over samples
    rng = np.random.default_rng(0)
    print("\nBREADTH CURVE (random subsets, mean±sd of Track-B t over 12 samples):")
    for k in [16, 24, 32, 40, 48, 56, 64, len(full)]:
        if k > len(full):
            continue
        ts = [book_t(list(rng.choice(full, size=k, replace=False))) for _ in range(12)]
        ts = [t for t in ts if t == t]
        if ts:
            print(f"  k={k:>3}: t = {np.mean(ts):+.2f} ± {np.std(ts):.2f}")

    # (c) bucket analysis — IBKR-16 + each family present in the universe
    print("\nBUCKET ANALYSIS (IBKR-16 + one family; dt vs IBKR-16 baseline):")
    base_t = book_t(ibkr)
    rows = []
    for name, mkts in BUCKETS.items():
        add = [m for m in mkts if m in full and m not in ibkr]
        if not add:
            continue
        t = book_t(ibkr + add)
        rows.append((name, len(add), t, t - base_t))
    for name, nadd, t, dt in sorted(rows, key=lambda r: -r[3]):
        print(f"  +{name:<16} (+{nadd:>2}): t = {t:+.2f}  (dt {dt:+.2f})")

    # NOTE: a GREEDY forward-selection (add the single in-sample-best market each step) was evaluated and
    # DELIBERATELY REMOVED — maximizing in-sample Track-B t over ~60 candidates/step is selection-on-the-
    # outcome (overfit, no deflation). It "cleared t>2 at n=19" by picking intl_equity/intl_rates markets
    # the bucket analysis rates NEUTRAL-TO-HARMFUL — a direct tell of overfit. The random-subset curve
    # above is the only selection-free estimate of the required breadth (~48). (Opus FB0 review.)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
