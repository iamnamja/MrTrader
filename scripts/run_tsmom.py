"""
Phase 2 (Alpha-v4) — validate the TSMOM / trend-following ETF sleeve.

Fetches a long ETF history (~2007 -> now, to capture 2008/2011/2015/2018/2020/
2022), runs the vectorized PIT-safe sleeve (app/strategy/tsmom.py), and reports:

  (1) STANDALONE:   Sharpe, CAGR, vol, maxDD, Calmar, avg gross, ann turnover.
  (2) PER-REGIME:   Sharpe in BULL / NEUTRAL / BEAR (coarse3 PIT regime map).
  (3) CRISIS:       return through 2008 GFC / 2020 COVID / 2022 — is it the
                    crisis-positive diversifier the long-biased book needs?
  (4) vs PEAD:      correlation on the 2020-26 overlap + a vol-weighted COMBINED
                    book (PEAD + trend) — the Phase-2 EXIT GATE: keep iff the
                    sleeve cuts combined drawdown / raises book Sharpe (modest
                    standalone Sharpe is fine — the value is diversification).

The committed Phase-1 verdict makes (4) the point: PEAD alone is ~beta; the
question is whether trend turns PEAD-plus-trend into a better BOOK.

Usage: python scripts/run_tsmom.py
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [tsmom] %(message)s")
logger = logging.getLogger(__name__)

START = "2007-01-01"
ANN = 252
# Pre-declared crisis windows (equity drawdowns) — fixed BEFORE seeing results.
CRISES = {
    "2008 GFC": ("2008-09-01", "2009-03-31"),
    "2011 EU":  ("2011-07-01", "2011-10-31"),
    "2018 Q4":  ("2018-10-01", "2018-12-31"),
    "2020 COVID": ("2020-02-19", "2020-03-23"),
    "2022 bear": ("2022-01-01", "2022-10-31"),
}


def _fetch_etf_prices(universe, start=START):
    import yfinance as yf
    cols = {}
    for sym in universe + ["SPY"]:
        try:
            df = yf.download(sym, start=start, end=datetime.now().date().isoformat(),
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            if len(df) > 260:
                cols[sym] = df["close"]
        except Exception as e:
            logger.warning("fetch failed %s: %s", sym, e)
    prices = pd.DataFrame(cols).sort_index()
    prices.index = pd.to_datetime(prices.index)
    logger.info("ETF prices: %d instruments, %s -> %s (%d days)",
                prices.shape[1], prices.index[0].date(), prices.index[-1].date(), len(prices))
    return prices


def _sharpe(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(ANN)) if len(r) > 2 and r.std() > 0 else 0.0


def _maxdd(r):
    eq = (1 + r.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def _scale_to_vol(r, target=0.10):
    """Scale a return series to a target annualized vol (ex-ante, full-sample —
    a fair common-risk basis for comparing book combinations)."""
    v = r.std() * np.sqrt(ANN)
    return r * (target / v) if v > 0 else r


def main() -> int:
    from app.strategy.tsmom import TSMOMConfig, tsmom_backtest
    from scripts.walkforward.regime import load_regime_map

    cfg = TSMOMConfig()
    prices = _fetch_etf_prices(cfg.universe)
    res = tsmom_backtest(prices, cfg)
    r = res.returns

    print("\n" + "=" * 72)
    print("  PHASE 2 — TSMOM TREND SLEEVE (long-flat, inv-vol, weekly, ETF basket)")
    print("=" * 72)
    print(f"  Universe: {', '.join([c for c in cfg.universe if c in prices.columns])}")
    print(f"  Period:   {r.index[0].date()} -> {r.index[-1].date()}  ({len(r)} days)")
    print("  (1) STANDALONE")
    s = res.summary()
    print(f"      Sharpe {s['sharpe']:+.3f} | CAGR {s['cagr']*100:+.1f}% | vol "
          f"{s['ann_vol']*100:.1f}% | maxDD {s['max_drawdown']*100:.1f}% | "
          f"Calmar {s['calmar']:.2f} | avgGross {s['avg_gross']:.2f} | "
          f"turnover {s['ann_turnover']:.1f}x/yr")

    # (2) per-regime
    try:
        rmap = load_regime_map(r.index[0].date(), r.index[-1].date())
        lab = pd.Series({pd.Timestamp(d): v for d, v in rmap.items()}).reindex(r.index).ffill()
        print("  (2) PER-REGIME Sharpe")
        for reg in ["BULL", "NEUTRAL", "BEAR"]:
            rr = r[lab == reg]
            print(f"      {reg:8} n={len(rr):4}  Sharpe {_sharpe(rr):+.3f}  "
                  f"mean {rr.mean()*ANN*100:+.1f}%/yr")
    except Exception as e:
        logger.warning("regime breakdown failed: %s", e)

    # (3) crisis windows — trend vs SPY buy&hold
    spy_r = prices["SPY"].pct_change()
    print("  (3) CRISIS WINDOWS  (trend total return  vs  SPY buy&hold)")
    for name, (a, b) in CRISES.items():
        seg = r[(r.index >= a) & (r.index <= b)]
        spyseg = spy_r[(spy_r.index >= a) & (spy_r.index <= b)]
        if len(seg) < 5:
            print(f"      {name:11} (insufficient data)")
            continue
        tr = float((1 + seg).prod() - 1)
        sp = float((1 + spyseg.fillna(0)).prod() - 1)
        print(f"      {name:11} trend {tr*100:+6.1f}%   SPY {sp*100:+6.1f}%")

    # (4) vs PEAD: correlation + combined book on the overlap
    try:
        from scripts.pead_phase1_attribution import _load_or_fetch_bars, _run_pass
        sd, spy_p, days = _load_or_fetch_bars()
        pead_r, _ = _run_pass(sd, spy_p, days, 0.0003)
        pead_r.index = pd.to_datetime(pead_r.index)
        join = pd.concat([r.rename("trend"), pead_r.rename("pead")], axis=1).dropna()
        if len(join) > 60:
            corr = float(join["trend"].corr(join["pead"]))
            print(f"  (4) vs PEAD  (overlap {join.index[0].date()} -> {join.index[-1].date()}, "
                  f"{len(join)} days)")
            print(f"      correlation(trend, PEAD) = {corr:+.3f}")
            # Vol-weighted combined book (each scaled to 10% vol, 50/50).
            t_s = _scale_to_vol(join["trend"]); p_s = _scale_to_vol(join["pead"])
            combo = 0.5 * t_s + 0.5 * p_s
            print(f"      PEAD-only (10% vol):   Sharpe {_sharpe(p_s):+.3f}  maxDD {_maxdd(p_s)*100:+.1f}%")
            print(f"      Trend-only (10% vol):  Sharpe {_sharpe(t_s):+.3f}  maxDD {_maxdd(t_s)*100:+.1f}%")
            print(f"      COMBINED 50/50:        Sharpe {_sharpe(combo):+.3f}  maxDD {_maxdd(combo)*100:+.1f}%")
            book_better = (_sharpe(combo) > _sharpe(p_s)) and (_maxdd(combo) > _maxdd(p_s))
            print("  EXIT GATE: " + (
                "KEEP — trend improves the book (higher Sharpe AND shallower drawdown vs PEAD-alone)"
                if book_better else
                "REVIEW — combined book does not strictly beat PEAD-alone on both Sharpe and drawdown"))
    except Exception as e:
        logger.warning("PEAD comparison skipped: %s", e)

    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
