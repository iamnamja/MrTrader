"""
Phase 1 — PEAD honest reckoning (Alpha-v4).

After the is_trained guard fix gave PEAD full-coverage CPCV (KL-11), decide the
real question: is PEAD a genuine diversifier, or conditional market beta dressed
up as alpha? (87% of P&L lands in up-trends — the tell.)

Runs the EXACT committed PEAD scorer (build_pead_scorer) over the full window in
ONE AgentSimulator pass per slippage level, then:

  (1.3) CAPM beta isolation:  r_book = alpha + beta * r_spy + eps
        alpha t-stat (Newey-West HAC) is the decisive statistic. The A1
        analyst-drift analog died here (alpha t=0.20). If PEAD's residual-alpha
        t < ~1, the Sharpe is largely market beta and PEAD is a risk-on sleeve,
        not standalone alpha.

  (1.4) Gapper-slippage stress: re-run at entry slippage {3, 30, 50} bps. PEAD
        enters post-earnings GAPPERS at the next open — the most-violated fill
        assumption. Idealized paper fills understate real slippage exactly where
        it bites, so the realized ~0.40 SR is probably an over-estimate. We sweep
        a punitive blanket entry slippage (PEAD entries ARE the gappers) and watch
        how fast the edge decays.

Bars are cached to disk so the slippage variants share one download (the R1K
yfinance pull is the slow part).

Usage: python scripts/pead_phase1_attribution.py
"""
import logging
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [pead_attr] %(message)s")
logger = logging.getLogger(__name__)

TOTAL_YEARS = 6
ANN = 252
CACHE = ROOT / "data" / "cache" / f"pead_phase1_bars_{TOTAL_YEARS}y.pkl"

# Entry-slippage sweep (one-way, fraction). 0.0003 = the committed 3 bps default;
# 0.0030 / 0.0050 = the punitive 30 / 50 bps gapper stress.
SLIPPAGE_LEVELS = [0.0003, 0.0030, 0.0050]


def _load_or_fetch_bars():
    """Fetch the R1K daily bars once (the slow part) and cache to disk so each
    slippage variant reuses the SAME data."""
    from app.utils.constants import RUSSELL_1000_TICKERS
    from scripts.run_pead_cpcv import PEADStrategy

    if CACHE.exists():
        logger.info("Loading cached bars from %s", CACHE)
        with open(CACHE, "rb") as f:
            blob = pickle.load(f)
        return blob["symbols_data"], blob["spy_prices"], blob["all_days"]

    strat = PEADStrategy(scorer=None, symbols=list(RUSSELL_1000_TICKERS),
                         transaction_cost_pct=0.0005)
    end = datetime.now()
    start = end - timedelta(days=TOTAL_YEARS * 365 + 30)
    strat.fetch_data(start, end)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump({"symbols_data": strat.symbols_data,
                     "spy_prices": strat.spy_prices,
                     "all_days": strat.all_days_sorted}, f)
    logger.info("Cached %d symbols to %s", len(strat.symbols_data), CACHE)
    return strat.symbols_data, strat.spy_prices, strat.all_days_sorted


def _capm(r_book: pd.Series, r_spy: pd.Series, hac_lag: int = 5) -> dict:
    """OLS r_book = alpha + beta*r_spy with OLS and Newey-West HAC alpha t-stats."""
    r_spy = r_spy.reindex(r_book.index).dropna()
    r_book = r_book.reindex(r_spy.index).dropna()
    r_spy = r_spy.reindex(r_book.index)
    x = r_spy.to_numpy()
    y = r_book.to_numpy()
    n = len(y)
    X = np.column_stack([np.ones_like(x), x])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha_d, beta = float(coef[0]), float(coef[1])
    resid = y - X @ coef
    dof = n - 2
    s2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se_alpha_ols = float(np.sqrt(s2 * XtX_inv[0, 0]))
    t_alpha_ols = alpha_d / se_alpha_ols if se_alpha_ols > 0 else 0.0

    # Newey-West HAC sandwich on the full [1, x] design (robust to the
    # autocorrelation/heteroskedasticity of overlapping multi-week holds).
    bread = XtX_inv
    S = np.zeros((2, 2))
    u = resid
    Xr = X * u[:, None]  # n x 2 score contributions
    S += Xr.T @ Xr
    for L in range(1, hac_lag + 1):
        w = 1.0 - L / (hac_lag + 1.0)  # Bartlett kernel
        G = Xr[L:].T @ Xr[:-L]
        S += w * (G + G.T)
    cov_hac = bread @ S @ bread
    se_alpha_hac = float(np.sqrt(max(cov_hac[0, 0], 0.0)))
    t_alpha_hac = alpha_d / se_alpha_hac if se_alpha_hac > 0 else 0.0

    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    raw_sharpe = float(y.mean() / y.std() * np.sqrt(ANN)) if y.std() > 0 else 0.0
    # Beta-REMOVED (market-hedged) return stream = y - beta*x; its mean is alpha,
    # so its Sharpe is the honest "what survives after hedging out SPY" number.
    # (The raw OLS residual y-alpha-beta*x is mean-zero by construction → Sharpe~0,
    # which is uninformative; that's NOT what we want here.)
    hedged = y - beta * x
    resid_sharpe = float(hedged.mean() / hedged.std() * np.sqrt(ANN)) if hedged.std() > 0 else 0.0
    return {
        "n": n, "raw_sharpe": raw_sharpe, "beta": beta,
        "alpha_ann": alpha_d * ANN, "alpha_bps_d": alpha_d * 1e4,
        "t_alpha_ols": t_alpha_ols, "t_alpha_hac": t_alpha_hac,
        "resid_sharpe": resid_sharpe, "r2": r2,
    }


def _run_pass(symbols_data, spy_prices, all_days, entry_slippage_pct):
    """One full-window AgentSimulator pass of the committed PEAD book at a given
    entry slippage. Returns the daily book-return series."""
    from app.backtesting.agent_simulator import AgentSimulator
    from scripts.run_pead_cpcv import build_pead_scorer

    scorer = build_pead_scorer()
    sim = AgentSimulator(
        model=None, factor_scorer=scorer,
        transaction_cost_pct=0.0005, no_prefilters=True,
        entry_slippage_pct=entry_slippage_pct,
    )
    result = sim.run(symbols_data, start_date=all_days[0], end_date=all_days[-1],
                     spy_prices=spy_prices)
    eq = getattr(result, "equity_curve", [])
    if len(eq) < 60:
        raise RuntimeError(f"equity curve too short ({len(eq)})")
    edf = pd.DataFrame(eq, columns=["date", "equity"]).set_index("date")
    edf.index = pd.to_datetime(edf.index)
    return edf["equity"].pct_change().dropna(), int(result.total_trades)


def main() -> int:
    symbols_data, spy_prices, all_days = _load_or_fetch_bars()
    spy = spy_prices.copy()
    spy.index = pd.to_datetime(spy.index)
    r_spy_all = spy.pct_change().dropna()

    rows = []
    for slip in SLIPPAGE_LEVELS:
        logger.info("Full-window PEAD pass at entry slippage = %.0f bps ...", slip * 1e4)
        r_book, n_trades = _run_pass(symbols_data, spy_prices, all_days, slip)
        m = _capm(r_book, r_spy_all)
        m["slip_bps"] = slip * 1e4
        m["n_trades"] = n_trades
        rows.append(m)

    base = rows[0]  # 3 bps default = the committed-config reference
    print("\n" + "=" * 78)
    print("  PHASE 1 — PEAD BETA ISOLATION (CAPM) + GAPPER-SLIPPAGE STRESS")
    print("  Committed long-only config, full-window single pass.")
    print("=" * 78)
    print(f"  {'slip':>6} {'trades':>7} {'rawSR':>7} {'beta':>6} "
          f"{'alphaAnn':>9} {'t(OLS)':>7} {'t(HAC)':>7} {'residSR':>8} {'R2spy':>6}")
    for m in rows:
        print(f"  {m['slip_bps']:>5.0f} {m['n_trades']:>7} {m['raw_sharpe']:>+7.3f} "
              f"{m['beta']:>+6.2f} {m['alpha_ann']*100:>+8.2f}% {m['t_alpha_ols']:>+7.2f} "
              f"{m['t_alpha_hac']:>+7.2f} {m['resid_sharpe']:>+8.3f} {m['r2']:>6.3f}")
    print("=" * 78)

    # Verdict keys on the HAC alpha t-stat at the committed 3 bps config.
    t_hac = base["t_alpha_hac"]
    beta = base["beta"]
    if t_hac >= 2.0:
        verdict = "REAL ALPHA beyond market beta (HAC alpha t>=2) — run beta-hedged"
    elif t_hac >= 1.0:
        verdict = ("WEAK / INCONCLUSIVE (1<=HAC alpha t<2) — small risk-on sleeve at "
                   "most; pair with a crisis-positive sleeve, never a centerpiece")
    else:
        verdict = ("BETA-DRIVEN (HAC alpha t<1) — the Sharpe is largely market beta; "
                   "PEAD is conditional risk-on exposure, not standalone alpha")
    print(f"  VERDICT (committed 3bps, beta={beta:+.2f}): {verdict}")
    # Slippage decay: how much of the edge survives punitive gapper fills.
    if len(rows) >= 3:
        print(f"  Gapper-slippage decay: rawSR {rows[0]['raw_sharpe']:+.3f} (3bps) -> "
              f"{rows[1]['raw_sharpe']:+.3f} (30bps) -> {rows[2]['raw_sharpe']:+.3f} (50bps)")
    print()
    logger.info("PEAD attribution done: beta=%.2f t_hac=%.2f resid_sharpe=%.3f -> %s",
                beta, t_hac, base["resid_sharpe"], verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
