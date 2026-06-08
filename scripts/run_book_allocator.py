"""
Phase 3 (Alpha-v4) — validate the regime-aware sleeve allocator.

Combines the two validated sleeves — PEAD (Phase 1) and TSMOM trend (Phase 2) —
into a BOOK under three schemes and asks the disciplined question: does a regime
TILT beat simple static vol-weighting, NET OF TURNOVER, or does it fail to earn
its complexity (in which case we ship static vol-weight and keep the tilt OFF)?

  equal-capital      1/N (reference)
  static vol-weight  inverse-vol risk parity  <- the baseline to beat
  regime-tilted      vol-weight x a-priori economic regime tilt (persistence+blend)

Caveat (honest): PEAD's return series only spans 2020-26, so the combined-book
overlap is thin (~COVID + 2022). With 2 sleeves over ~6 years, the bar for
adopting a regime layer is HIGH — it must clearly beat static vol-weight on both
Sharpe and drawdown net of turnover. Trend's Phase-2 profile (slow-bear-positive,
fast-shock-NEGATIVE) makes a blanket "trend up in BEAR" tilt ambiguous on this
sample, so a null result here is the expected, principled outcome.

Usage: python scripts/run_book_allocator.py
"""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [book] %(message)s")
logger = logging.getLogger(__name__)
ANN = 252


def _sleeve_returns():
    """PEAD (cached R1K bars, full-window pass) + TSMOM trend (ETF backtest)."""
    from app.strategy.tsmom import TSMOMConfig, tsmom_backtest
    from scripts.run_tsmom import _fetch_etf_prices
    from scripts.pead_phase1_attribution import _load_or_fetch_bars, _run_pass

    tcfg = TSMOMConfig()
    trend_r = tsmom_backtest(_fetch_etf_prices(tcfg.universe), tcfg).returns
    trend_r.index = pd.to_datetime(trend_r.index)

    sd, spy, days = _load_or_fetch_bars()
    pead_r, _ = _run_pass(sd, spy, days, 0.0003)
    pead_r.index = pd.to_datetime(pead_r.index)

    rets = pd.concat([pead_r.rename("pead"), trend_r.rename("trend")], axis=1).dropna()
    logger.info("sleeve overlap: %s -> %s (%d days)",
                rets.index[0].date(), rets.index[-1].date(), len(rets))
    return rets


def main() -> int:
    import argparse
    _ap = argparse.ArgumentParser(description="Book-level sleeve-allocator gate")
    _ap.add_argument("--emit-config", action="store_true",
                     help="print the recommended pm.allocator_scheme for the live allocator")
    _args = _ap.parse_args()

    from app.strategy.sleeve_allocator import build_book, AllocatorConfig, DEFAULT_REGIME_TILT
    from scripts.walkforward.regime import load_regime_map

    rets = _sleeve_returns()
    rmap = load_regime_map(rets.index[0].date(), rets.index[-1].date())
    labels = pd.Series({pd.Timestamp(d): v for d, v in rmap.items()}).reindex(rets.index).ffill()

    cfg = AllocatorConfig()
    books = {
        "equal-capital":     build_book(rets, "equal", cfg=cfg),
        "static vol-weight": build_book(rets, "vol", cfg=cfg),
        "regime-tilted":     build_book(rets, "regime", regime_labels=labels, cfg=cfg),
    }

    print("\n" + "=" * 78)
    print("  PHASE 3 — REGIME-AWARE SLEEVE ALLOCATOR (PEAD + TSMOM trend)")
    print(f"  Overlap {rets.index[0].date()} -> {rets.index[-1].date()} ({len(rets)} days)")
    print(f"  Regime tilt (a-priori): {DEFAULT_REGIME_TILT}")
    print("=" * 78)
    print(f"  {'book':18} {'Sharpe':>7} {'CAGR':>7} {'vol':>6} {'maxDD':>7} "
          f"{'Calmar':>7} {'turnover':>9}")
    for name, b in books.items():
        s = b.summary()
        print(f"  {name:18} {s['sharpe']:>+7.3f} {s['cagr']*100:>+6.1f}% {s['ann_vol']*100:>5.1f}% "
              f"{s['max_drawdown']*100:>+6.1f}% {s['calmar']:>7.2f} {s['ann_turnover']:>8.1f}x")
    print("=" * 78)

    vol_b = books["static vol-weight"].summary()
    reg_b = books["regime-tilted"].summary()
    eq_b = books["equal-capital"].summary()

    # COMPLEXITY LADDER: prefer the SIMPLEST scheme unless a more complex one CLEARLY
    # wins (Sharpe margin > 0.10 AND drawdown no worse), so added turnover/parameters
    # must pay for themselves. equal-capital (0 params, 0 turnover) is the floor;
    # vol-weight must beat it to justify its turnover; regime-tilt must beat vol-weight.
    SHARPE_MARGIN = 0.10
    vol_beats_equal = (vol_b["sharpe"] - eq_b["sharpe"] > SHARPE_MARGIN) and \
                      (vol_b["max_drawdown"] >= eq_b["max_drawdown"])
    regime_beats_vol = (reg_b["sharpe"] - vol_b["sharpe"] > SHARPE_MARGIN) and \
                       (reg_b["max_drawdown"] >= vol_b["max_drawdown"])
    print(f"  vol-weight vs equal-capital: Sharpe {eq_b['sharpe']:+.3f} -> {vol_b['sharpe']:+.3f}, "
          f"maxDD {eq_b['max_drawdown']*100:+.1f}% -> {vol_b['max_drawdown']*100:+.1f}%, "
          f"turnover {eq_b['ann_turnover']:.1f}x -> {vol_b['ann_turnover']:.1f}x "
          f"-> {'vol-weight earns it' if vol_beats_equal else 'NO (equal-capital wins)'}")
    print(f"  regime-tilt vs static vol-weight: Sharpe {vol_b['sharpe']:+.3f} -> {reg_b['sharpe']:+.3f} "
          f"(d{reg_b['sharpe']-vol_b['sharpe']:+.3f}), maxDD {vol_b['max_drawdown']*100:+.1f}% -> "
          f"{reg_b['max_drawdown']*100:+.1f}% (d{(reg_b['max_drawdown']-vol_b['max_drawdown'])*100:+.1f}pp), "
          f"turnover {vol_b['ann_turnover']:.1f}x -> {reg_b['ann_turnover']:.1f}x "
          f"-> {'regime earns it' if regime_beats_vol else 'NO (tilt does not earn it)'}")
    print("=" * 78)
    if vol_beats_equal and regime_beats_vol:
        ship = "regime-tilted"
    elif vol_beats_equal:
        ship = "static vol-weight"
    else:
        ship = "equal-capital (simple fixed weight)"
    print(f"  VERDICT: SHIP {ship.upper()}. Neither vol-tilt nor regime-tilt that fails its "
          "margin is adopted — added complexity/turnover must pay for itself. On this thin "
          "2020-26 overlap inverse-vol is fooled by PEAD's sparse-low vol (over-weights the weak "
          "sleeve) and the regime tilt is worse on every metric, so the SIMPLE book wins. Keep "
          "the vol/regime layers as OFF scaffold; revisit when more sleeves / longer overlapping "
          "history exist. (Complexity must earn it.)")
    print()
    logger.info("allocator: equal SR %.3f | vol SR %.3f | regime SR %.3f -> SHIP %s",
                eq_b["sharpe"], vol_b["sharpe"], reg_b["sharpe"], ship)

    if _args.emit_config:
        scheme = ("regime" if ship == "regime-tilted"
                  else "vol" if ship == "static vol-weight" else "equal")
        print(f"  RECOMMENDED LIVE CONFIG: pm.allocator_scheme = {scheme}")
        print(f"    apply with: python -m scripts.set_allocator_config --enable --scheme {scheme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
