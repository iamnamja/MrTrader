"""
run_trend_broadening.py — Alpha-v6 P5 confirmatory run (the ONE frozen shot).

Runs the FROZEN broadened TSMOM spec (scripts/preregister_p5_trend_broadening.py:
16 ETFs + long-short + 10% book-vol overlay) against the current 10-ETF live sleeve
as the baseline, over ~2007->now (19y), on the SAME date window. Reports standalone
Sharpe + its t-stat (t ~= Sharpe*sqrt(years)) + maxDD/Calmar + per-crisis returns
for both, applies the frozen pass/park rule, and (with --record) writes the one-shot
R4 (run_at must post-date the prereg instant).

EXPLORATORY by default. Usage:
  python -m scripts.run_trend_broadening
  python -m scripts.run_trend_broadening --record
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.strategy.tsmom import TSMOMConfig, tsmom_backtest  # noqa: E402
from scripts.preregister_p5_trend_broadening import (  # noqa: E402
    BROADENED_SPEC, CORE_UNIVERSE, HYPOTHESIS_ID,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [p5] %(message)s")
logger = logging.getLogger(__name__)

START = "2007-01-01"
ANN = 252
T_PASS = 2.0
MAXDD_TOLERANCE = 0.02   # broadened maxDD may be at most this much worse (abs) than baseline
CRISES = {
    "2008 GFC": ("2008-09-01", "2009-03-31"), "2011 EU": ("2011-07-01", "2011-10-31"),
    "2018 Q4": ("2018-10-01", "2018-12-31"), "2020 COVID": ("2020-02-19", "2020-03-23"),
    "2022 bear": ("2022-01-01", "2022-10-31"),
}


def _fetch(universe):
    import yfinance as yf
    cols = {}
    for sym in sorted(set(universe + ["SPY"])):
        try:
            df = yf.download(sym, start=START, end=datetime.now(timezone.utc).date().isoformat(),
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            if len(df) > 260:
                cols[sym] = df["close"]
        except Exception as exc:
            logger.warning("fetch %s failed: %s", sym, exc)
    return pd.DataFrame(cols).sort_index()


def _metrics(res, window_returns=None):
    r = window_returns if window_returns is not None else res.returns
    n = len(r)
    yrs = max(n / ANN, 1e-9)
    sharpe = float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else 0.0
    tstat = float(sharpe * np.sqrt(yrs))
    eq = (1.0 + r.fillna(0.0)).cumprod()
    mdd = float((eq / eq.cummax() - 1.0).min())
    cagr = float(eq.iloc[-1] ** (1.0 / yrs) - 1.0) if n else 0.0
    return {"sharpe": round(sharpe, 3), "tstat": round(tstat, 2),
            "ann_vol": round(float(r.std() * np.sqrt(ANN)), 3),
            "maxdd": round(mdd, 3), "calmar": round(cagr / abs(mdd) if mdd < 0 else 0.0, 3),
            "cagr": round(cagr, 3), "n_days": n, "years": round(yrs, 1)}


def _crisis_returns(returns):
    out = {}
    for name, (s, e) in CRISES.items():
        w = returns[(returns.index >= s) & (returns.index <= e)]
        out[name] = round(float((1.0 + w.fillna(0.0)).prod() - 1.0), 4) if len(w) else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the P5 trend-broadening confirmatory")
    ap.add_argument("--record", action="store_true", help="record the one-shot R4")
    ap.add_argument("--run-at", default=None)
    args = ap.parse_args()

    broadened_universe = BROADENED_SPEC["universe"]
    prices = _fetch(broadened_universe)
    logger.info("loaded %d ETFs, %s -> %s", prices.shape[1],
                prices.index.min().date(), prices.index.max().date())

    broadened_cfg = TSMOMConfig(**{k: (tuple(v) if isinstance(v, list) and k == "lookbacks"
                                       else v) for k, v in BROADENED_SPEC.items()})
    baseline_cfg = TSMOMConfig(universe=CORE_UNIVERSE,
                               lookbacks=tuple(BROADENED_SPEC["lookbacks"]))  # live defaults

    broadened = tsmom_backtest(prices, broadened_cfg)
    baseline = tsmom_backtest(prices[[c for c in CORE_UNIVERSE if c in prices.columns]],
                              baseline_cfg)

    # Compare on the SHARED date window (fair head-to-head).
    common = broadened.returns.index.intersection(baseline.returns.index)
    b_m = _metrics(broadened, broadened.returns.loc[common])
    base_m = _metrics(baseline, baseline.returns.loc[common])

    # Frozen pass rule: broadened t>=2 AND beats baseline (higher Sharpe AND maxDD
    # not materially worse).
    t_ok = b_m["tstat"] >= T_PASS
    beats_sharpe = b_m["sharpe"] > base_m["sharpe"]
    dd_ok = b_m["maxdd"] >= base_m["maxdd"] - MAXDD_TOLERANCE   # less negative or within tol
    verdict = "PASS" if (t_ok and beats_sharpe and dd_ok) else "PARK"

    logger.info("=" * 70)
    logger.info("P5 VERDICT: %s  (t>=2:%s beats_sharpe:%s dd_ok:%s)",
                verdict, t_ok, beats_sharpe, dd_ok)
    logger.info("  BROADENED (16 ETF, L/S, book-vol 10%%): %s", b_m)
    logger.info("  BASELINE  (10 ETF live):                %s", base_m)
    logger.info("  broadened crises: %s", _crisis_returns(broadened.returns.loc[common]))
    logger.info("  baseline  crises: %s", _crisis_returns(baseline.returns.loc[common]))
    logger.info("=" * 70)

    result = {"verdict": verdict, "broadened": b_m, "baseline": base_m,
              "broadened_crises": _crisis_returns(broadened.returns.loc[common]),
              "baseline_crises": _crisis_returns(baseline.returns.loc[common]),
              "window": [str(common.min().date()), str(common.max().date())]}

    if args.record:
        from app.research.registry import RegistryIntegrityError, ResearchRegistry
        run_at = args.run_at or datetime.now(timezone.utc).isoformat()
        decision = "promote_paper" if verdict == "PASS" else "park"
        try:
            ResearchRegistry().record_result(HYPOTHESIS_ID, run_at=run_at,
                                             result=result, decision=decision)
            logger.info("RECORDED R4 for %s (run_at=%s, decision=%s)",
                        HYPOTHESIS_ID, run_at, decision)
        except RegistryIntegrityError as exc:
            logger.error("registry refused: %s", exc)
            return 1
    else:
        logger.info("EXPLORATORY — not recorded (pass --record to commit the one-shot).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
