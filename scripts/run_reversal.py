"""
Validate the short-term cross-sectional reversal sleeve (Alpha-v4 P4, 3rd premium).

Mirrors scripts/run_tsmom.py + scripts/trend_residual_alpha.py: standalone PIT backtest
on the cached R1K universe (survivorship-safe via point-in-time index membership), a
punitive one-way cost sweep (the make-or-break for reversal), CAPM/HAC beta-isolation
(should be ~0 beta by construction), correlation to the live sleeves (PEAD + trend), and
the 3-sleeve vs 2-sleeve book marginal contribution. Ends with a KEEP/KILL verdict.

Run:  PYTHONIOENCODING=utf-8 python -m scripts.run_reversal
"""
from __future__ import annotations

import glob
import logging
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from app.strategy.reversal import ReversalConfig, reversal_backtest
from scripts.walkforward.attribution import capm_alpha

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("reversal")

_ROOT = Path(__file__).resolve().parents[1]
CACHE = _ROOT / "data" / "price_cache"
MEMBERSHIP = _ROOT / "data" / "universe" / "russell1000_membership.parquet"
START = "2007-01-01"
ANN = 252
# Broad ETFs that live in price_cache — exclude from a SINGLE-NAME reversal cross-section.
_BENCH = {"SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "GLD", "DBC", "UUP", "EFA", "EEM",
          "HYG", "LQD", "VXX", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
          "XLB", "XLRE", "XLC"}
CRISES = [
    ("GFC 2008", "2008-09-01", "2009-03-31"),
    ("COVID crash", "2020-02-19", "2020-03-23"),
    ("2022 bear", "2022-01-01", "2022-10-31"),
]


def _load_price_volume(start: str):
    closes, vols = {}, {}
    for f in glob.glob(str(CACHE / "*.parquet")):
        sym = os.path.basename(f)[:-8]
        if sym in _BENCH:
            continue
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if "close" not in df.columns or "volume" not in df.columns:
            continue
        df = df[df.index >= pd.Timestamp(start)]
        if len(df) < 260:
            continue
        closes[sym] = df["close"]
        vols[sym] = df["volume"]
    P = pd.DataFrame(closes).sort_index()
    V = pd.DataFrame(vols).reindex_like(P)
    log.info("Loaded %d single-name symbols, %s -> %s (%d days)",
             P.shape[1], P.index.min().date(), P.index.max().date(), len(P))
    return P, V


def _membership_mask(index, columns):
    """PIT index-membership mask (survivorship-safe): True where a name was a member."""
    if not MEMBERSHIP.exists():
        log.warning("No membership parquet — using ALL cached names (SURVIVORSHIP-BIASED)")
        return None
    m = pd.read_parquet(MEMBERSHIP)
    mask = pd.DataFrame(False, index=index, columns=columns)
    cols = set(columns)
    for _, row in m.iterrows():
        t = row["ticker"]
        if t not in cols:
            continue
        a = pd.Timestamp(row["added"]) if pd.notna(row.get("added")) else index[0]
        r = (pd.Timestamp(row["removed"]) if pd.notna(row.get("removed"))
             else index[-1] + pd.Timedelta(days=1))
        mask.loc[(index >= a) & (index < r), t] = True
    cov = float(mask.sum(axis=1).mean())
    log.info("PIT membership mask: avg %.0f eligible names/day", cov)
    return mask


def _spy_returns(index):
    spy = pd.read_parquet(CACHE / "SPY.parquet")["close"]
    spy.index = pd.to_datetime(spy.index)
    return spy.pct_change().dropna().reindex(index).dropna()


def _sharpe(r):
    return float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else 0.0


def _win_ret(r, a, b):
    w = r.loc[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]
    return float((1.0 + w).prod() - 1.0), len(w)


def main() -> int:
    P, V = _load_price_volume(START)
    elig = _membership_mask(P.index, P.columns)

    base = reversal_backtest(P, V, ReversalConfig(cost_bps=10.0), eligible=elig)
    r = base.returns
    r_spy = _spy_returns(r.index)
    m = capm_alpha(r, r_spy)
    s = base.summary()

    print("\n" + "=" * 80)
    print("  SHORT-TERM REVERSAL SLEEVE — dollar-neutral cross-sectional (R1K, PIT membership)")
    print(f"  {P.index.min().date()} -> {P.index.max().date()} | lookback=5 skip=1 daily | "
          f"top-{ReversalConfig().liquidity_top_n} liquid | 10bps one-way")
    print("=" * 80)
    print(f"  net Sharpe:        {s['sharpe']:+.3f}   (raw before beta-hedge)")
    print(f"  CAGR/vol/maxDD:    {s['cagr']*100:+.1f}% / {s['ann_vol']*100:.1f}% / {s['max_drawdown']*100:+.1f}%")
    print(f"  ann turnover:      {s['ann_turnover']:.0f}x   avg names L/S: {s['avg_n_long']:.0f}/{s['avg_n_short']:.0f}")
    print(f"  SPY beta:          {m['beta']:+.3f}   (dollar-neutral -> expect ~0)")
    print(f"  alpha t (HAC):     {m['t_alpha_hac']:+.2f}   beta-hedged Sharpe: {m['resid_sharpe']:+.3f}")
    print("=" * 80)

    print("  COST SENSITIVITY (one-way bps) — the make-or-break for reversal")
    print(f"  {'bps':>5}{'netSR':>8}{'t_HAC':>7}{'hedgedSR':>9}{'turnover':>10}")
    for c in (2.0, 5.0, 10.0, 20.0, 30.0):
        bt = reversal_backtest(P, V, ReversalConfig(cost_bps=c), eligible=elig)
        mc = capm_alpha(bt.returns, r_spy.reindex(bt.returns.index))
        print(f"  {c:>5.0f}{bt.sharpe:>+8.3f}{mc['t_alpha_hac']:>+7.2f}{mc['resid_sharpe']:>+9.3f}"
              f"{bt.summary()['ann_turnover']:>10.0f}")
    print("=" * 80)

    print("  REBALANCE-FREQUENCY SWEEP (cut turnover/cost) @ 10bps")
    print(f"  {'rebal_d':>8}{'netSR':>8}{'turnover':>10}")
    for rb in (1, 2, 3, 5):
        bt = reversal_backtest(P, V, ReversalConfig(cost_bps=10.0, rebalance_days=rb), eligible=elig)
        print(f"  {rb:>8}{bt.sharpe:>+8.3f}{bt.summary()['ann_turnover']:>10.0f}")
    print("=" * 80)

    print("  PER-YEAR net Sharpe (10bps)")
    by = r.groupby(r.index.year).apply(_sharpe)
    print("  " + "  ".join(f"{y}:{v:+.2f}" for y, v in by.items()))
    print(f"  positive years: {int((by > 0).sum())}/{len(by)}")
    print("  CRISIS windows (total return):")
    for label, a, b in CRISES:
        tr, n = _win_ret(r, a, b)
        print(f"    {label:<14}{tr*100:>+7.1f}%  ({n}d)")
    print("=" * 80)

    # ── Diversification + book marginal contribution vs the live sleeves ──────────
    try:
        from scripts.run_book_allocator import _sleeve_returns
        from app.strategy.sleeve_allocator import build_book
        live = _sleeve_returns()  # DataFrame[['pead','trend']] on the overlap
        book3 = pd.concat([live, r.rename("reversal")], axis=1).dropna()
        print(f"  OVERLAP with live sleeves: {book3.index.min().date()} -> {book3.index.max().date()} "
              f"({len(book3)} days)")
        corr = book3.corr()
        print("  Correlations:")
        print(f"    reversal vs PEAD  : {corr.loc['reversal','pead']:+.3f}")
        print(f"    reversal vs trend : {corr.loc['reversal','trend']:+.3f}")
        b2 = build_book(book3[["pead", "trend"]], "equal").summary()
        b3 = build_book(book3[["pead", "trend", "reversal"]], "equal").summary()
        print("  Equal-capital book (marginal contribution of adding reversal):")
        print(f"    2-sleeve {{pead,trend}}        Sharpe {b2['sharpe']:+.3f}  maxDD {b2['max_drawdown']*100:+.1f}%")
        print(f"    3-sleeve {{pead,trend,reversal}} Sharpe {b3['sharpe']:+.3f}  maxDD {b3['max_drawdown']*100:+.1f}%")
        book_improves = (b3["sharpe"] > b2["sharpe"]) or (b3["max_drawdown"] > b2["max_drawdown"])
        uncorr = abs(corr.loc["reversal", "pead"]) < 0.3 and abs(corr.loc["reversal", "trend"]) < 0.3
    except Exception as exc:
        log.warning("book-level overlap skipped: %s", exc)
        book_improves = uncorr = None

    # ── Verdict (gate) ───────────────────────────────────────────────────────────
    bt10 = base
    m10 = capm_alpha(bt10.returns, r_spy)
    survives_cost = bt10.sharpe >= 0.30 and m10["t_alpha_hac"] >= 2.0
    print("=" * 80)
    print("  VERDICT GATE: KEEP iff net Sharpe@10bps >= 0.30 AND alpha t(HAC) >= 2 "
          "AND |corr| < 0.3 to both sleeves AND book improves.")
    print(f"    survives realistic cost (>=0.30 / t>=2): {survives_cost}")
    print(f"    genuinely uncorrelated (<0.3 both):      {uncorr}")
    print(f"    improves the book (Sharpe or DD):        {book_improves}")
    keep = bool(survives_cost and uncorr and book_improves)
    print(f"  --> {'KEEP (validated 3rd sleeve)' if keep else 'KILL / benchmark-only (does not earn it)'}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
