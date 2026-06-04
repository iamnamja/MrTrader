"""
A1 beta isolation: does the long-only analyst-upgrade book have CAPM alpha, or is
its Sharpe just market beta? Runs the long-only AnalystRevisionScorer over the full
window in ONE AgentSimulator pass, then regresses daily book returns on SPY:

    r_book = alpha + beta * r_spy + eps

Significant positive alpha (t>2) => real long-side drift worth running beta-hedged.
alpha ~ 0 with high beta => the +0.894 long-only Sharpe was market beta (edge dead).

Usage: python scripts/analyst_beta_check.py
"""
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [beta_check] %(message)s")
logger = logging.getLogger(__name__)

TOTAL_YEARS = 6


def main() -> int:
    from app.utils.constants import RUSSELL_1000_TICKERS
    from app.backtesting.agent_simulator import AgentSimulator
    from app.ml.analyst_revision_scorer import AnalystRevisionScorer
    from scripts.walkforward.event_edge import EventEdgeStrategy

    strat = EventEdgeStrategy(
        scorer=AnalystRevisionScorer(lookback_days=30, max_days_after=5,
                                     min_net_momentum=1.0, long_short=False, vix_block_all=30.0),
        symbols=list(RUSSELL_1000_TICKERS),
        model_type="analyst_drift",
        transaction_cost_pct=0.0005,
        max_hold_bars_override=20,
    )
    end = datetime.now()
    start = end - timedelta(days=TOTAL_YEARS * 365 + 30)
    strat.fetch_data(start, end)

    # Single full-window long-only pass (no CPCV folds).
    sim = AgentSimulator(model=None, factor_scorer=strat.scorer,
                         transaction_cost_pct=0.0005, no_prefilters=True,
                         max_hold_bars_override=20)
    start_d = strat.all_days_sorted[0]
    end_d = strat.all_days_sorted[-1]
    result = sim.run({s: d for s, d in strat.symbols_data.items()},
                     start_date=start_d, end_date=end_d, spy_prices=strat.spy_prices)

    eq = getattr(result, "equity_curve", [])
    if len(eq) < 60:
        logger.error("equity curve too short (%d) — aborting", len(eq))
        return 1
    edf = pd.DataFrame(eq, columns=["date", "equity"]).set_index("date")
    edf.index = pd.to_datetime(edf.index)
    r_book = edf["equity"].pct_change().dropna()

    spy = strat.spy_prices.copy()
    spy.index = pd.to_datetime(spy.index)
    r_spy = spy.pct_change().reindex(r_book.index).dropna()
    r_book = r_book.reindex(r_spy.index).dropna()
    r_spy = r_spy.reindex(r_book.index)

    # OLS: r_book = alpha + beta * r_spy
    x = r_spy.to_numpy()
    y = r_book.to_numpy()
    X = np.column_stack([np.ones_like(x), x])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha_d, beta = float(coef[0]), float(coef[1])
    resid = y - X @ coef
    n = len(y)
    dof = n - 2
    s2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se_alpha = float(np.sqrt(s2 * XtX_inv[0, 0]))
    t_alpha = alpha_d / se_alpha if se_alpha > 0 else 0.0
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0

    ann = 252
    raw_sharpe = float(y.mean() / y.std() * np.sqrt(ann)) if y.std() > 0 else 0.0
    resid_sharpe = float(resid.mean() / resid.std() * np.sqrt(ann)) if resid.std() > 0 else 0.0
    alpha_ann = alpha_d * ann

    print("\n" + "=" * 64)
    print("  A1 ANALYST-DRIFT — CAPM BETA ISOLATION (long-only, full window)")
    print("=" * 64)
    print(f"  Days:            {n}")
    print(f"  Raw Sharpe:      {raw_sharpe:+.3f}")
    print(f"  Market beta:     {beta:+.3f}")
    print(f"  Annual alpha:    {alpha_ann*100:+.2f}%   (daily {alpha_d*1e4:+.2f} bps)")
    print(f"  Alpha t-stat:    {t_alpha:+.2f}   (>2 => real alpha beyond beta)")
    print(f"  Residual Sharpe: {resid_sharpe:+.3f}   (beta-removed)")
    print(f"  R^2 to SPY:      {r2:.3f}")
    print("=" * 64)
    verdict = ("REAL ALPHA (run beta-hedged)" if t_alpha > 2.0
               else "BETA-DRIVEN (alpha not significant) — edge not validated")
    print(f"  VERDICT: {verdict}\n")
    logger.info("beta-check: beta=%.3f alpha_ann=%.2f%% t_alpha=%.2f resid_sharpe=%.3f r2=%.3f -> %s",
                beta, alpha_ann * 100, t_alpha, resid_sharpe, r2, verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
