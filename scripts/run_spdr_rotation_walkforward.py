"""
Phase G.1 — SPDR Sector Rotation Floor Benchmark.

Naive baseline: rank 11 SPDR sector ETFs by 6-month momentum (1-month skip),
hold top-3 equal-weight, monthly rebalance, 5bps round-trip costs.

Also computes equal-weight 11-SPDR portfolio as a second floor.

5-fold, 6-year walk-forward, same gate as factor WF (avg Sharpe ≥ 0.80).

If this passes the gate → sector rotation becomes the primary strategy
and single-name picker work is deprioritized.

Usage:
    python scripts/run_spdr_rotation_walkforward.py
"""
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [spdr_wf] %(message)s")
logger = logging.getLogger(__name__)

SPDR_ETFS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"]
TOP_N = 3
MOM_MONTHS = 6          # 6-month lookback
MOM_SKIP_MONTHS = 1     # skip most recent month (standard Jegadeesh-Titman)
REBALANCE_DAY = 1       # rebalance on first trading day of each month
COST_BPS = 5            # one-way cost in basis points
N_FOLDS = 5
TOTAL_YEARS = 6
PURGE_DAYS = 10
GATE = {"min_avg_sharpe": 0.80, "min_fold_sharpe": -0.30}


@dataclass
class FoldResult:
    fold: int
    test_start: date
    test_end: date
    sharpe_top3: float
    sharpe_ew: float
    total_return_top3: float
    trades_top3: int


def _download_prices(symbols: list, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    logger.info("Downloading %d ETFs %s→%s", len(symbols), start, end)
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]].rename(columns={"Close": symbols[0]})
    closes.columns = [str(c) for c in closes.columns]
    return closes.ffill().dropna(how="all")


def _momentum_score(closes: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """6-month return with 1-month skip as of `as_of`."""
    idx = closes.index.get_loc(as_of) if as_of in closes.index else None
    if idx is None:
        pos = closes.index.searchsorted(as_of, side="right") - 1
        idx = max(0, pos)
    skip_idx = max(0, idx - int(MOM_SKIP_MONTHS * 21))
    start_idx = max(0, skip_idx - int(MOM_MONTHS * 21))
    c_now = closes.iloc[skip_idx]
    c_start = closes.iloc[start_idx]
    return (c_now / c_start.replace(0, np.nan) - 1).dropna()


def _simulate_strategy(
    closes: pd.DataFrame,
    start: date,
    end: date,
    top_n: int,
    equal_weight_all: bool = False,
) -> tuple[pd.Series, int]:
    """
    Simulate the rotation strategy over [start, end].
    Returns (equity_curve, trade_count).
    """
    period_closes = closes.loc[
        (closes.index >= pd.Timestamp(start)) & (closes.index <= pd.Timestamp(end))
    ].copy()

    if period_closes.empty or len(period_closes) < 5:
        return pd.Series(dtype=float), 0

    portfolio = pd.Series(1.0 / len(SPDR_ETFS), index=period_closes.columns)  # equal-weight start
    equity = [1.0]
    current_holdings = list(period_closes.columns) if equal_weight_all else []
    trades = 0
    prev_month = None

    for i, (dt, row) in enumerate(period_closes.iterrows()):
        month = (dt.year, dt.month)

        if month != prev_month:
            # Rebalance
            if i < int(MOM_MONTHS * 21) + int(MOM_SKIP_MONTHS * 21) + 5:
                prev_month = month
                continue  # need enough history for momentum

            if not equal_weight_all:
                # Compute momentum and select top-N
                hist_closes = closes.loc[closes.index <= dt]
                scores = _momentum_score(hist_closes, dt)
                scores = scores.reindex(period_closes.columns).dropna()
                if len(scores) < top_n:
                    prev_month = month
                    continue
                new_holdings = list(scores.nlargest(top_n).index)
            else:
                new_holdings = list(period_closes.columns)

            # Count rebalances as trades (round-trip cost when holdings change)
            n_changed = len(set(new_holdings) ^ set(current_holdings))
            if n_changed > 0 and equity:
                cost = equity[-1] * (n_changed / len(new_holdings)) * (COST_BPS / 10000)
                equity[-1] = equity[-1] - cost
                trades += n_changed

            current_holdings = new_holdings
            portfolio = pd.Series(
                1.0 / len(current_holdings), index=current_holdings
            )
            prev_month = month

        if not current_holdings or i == 0:
            equity.append(equity[-1])
            continue

        # Daily return of portfolio
        if i > 0:
            prev_row = period_closes.iloc[i - 1]
            rets = (row[current_holdings] / prev_row[current_holdings].replace(0, np.nan) - 1).fillna(0)
            port_ret = (portfolio * rets).sum()
            equity.append(equity[-1] * (1 + port_ret))

    return pd.Series(equity, dtype=float), trades


def _sharpe(equity: pd.Series, ann: float = 252.0) -> float:
    if len(equity) < 20:
        return float("nan")
    rets = equity.pct_change().dropna()
    if rets.std() == 0:
        return float("nan")
    return float(rets.mean() / rets.std() * ann ** 0.5)


def _total_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return float("nan")
    return float(equity.iloc[-1] / equity.iloc[0] - 1) * 100


def run_fold(closes: pd.DataFrame, fold_idx: int, te_start: date, te_end: date) -> FoldResult:
    eq_top3, trades = _simulate_strategy(closes, te_start, te_end, top_n=TOP_N)
    eq_ew, _ = _simulate_strategy(closes, te_start, te_end, top_n=len(SPDR_ETFS), equal_weight_all=True)

    sh_top3 = _sharpe(eq_top3)
    sh_ew = _sharpe(eq_ew)
    tr = _total_return(eq_top3)
    logger.info(
        "Fold %d  %s→%s  top3_sharpe=%.3f  ew_sharpe=%.3f  return=%.1f%%  trades=%d",
        fold_idx, te_start, te_end, sh_top3, sh_ew, tr, trades,
    )
    return FoldResult(fold_idx, te_start, te_end, sh_top3, sh_ew, tr, trades)


def main() -> int:
    end_all = date.today()
    start_all = end_all - timedelta(days=TOTAL_YEARS * 365 + MOM_MONTHS * 32 + 30)
    segment_days = int(TOTAL_YEARS * 365 / (N_FOLDS + 1))

    closes = _download_prices(SPDR_ETFS, start_all.isoformat(), end_all.isoformat())
    logger.info("Downloaded closes: %s → %s (%d rows, %d ETFs)",
                closes.index[0].date(), closes.index[-1].date(), len(closes), len(closes.columns))

    fold_results = []
    for fold_idx in range(N_FOLDS):
        train_end_dt = end_all - timedelta(days=segment_days * (N_FOLDS - fold_idx))
        te_start = train_end_dt + timedelta(days=PURGE_DAYS)
        te_end = end_all - timedelta(days=segment_days * (N_FOLDS - fold_idx - 1))
        te_end = min(te_end, end_all)
        fold_results.append(run_fold(closes, fold_idx, te_start, te_end))

    top3_sharpes = [r.sharpe_top3 for r in fold_results]
    ew_sharpes = [r.sharpe_ew for r in fold_results]
    avg_sh = float(np.nanmean(top3_sharpes))
    min_sh = float(np.nanmin(top3_sharpes))
    gate_ok = avg_sh >= GATE["min_avg_sharpe"] and min_sh >= GATE["min_fold_sharpe"]
    verdict = "GATE PASSED" if gate_ok else "GATE FAILED"

    print("\n=== SPDR SECTOR ROTATION WALK-FORWARD ===")
    print(f"  Config: top-{TOP_N}, {MOM_MONTHS}m momentum, 1m skip, {COST_BPS}bps costs")
    print(f"  Verdict: {verdict}")
    print(f"  Avg Sharpe (top-3): {avg_sh:.3f}  (gate: >={GATE['min_avg_sharpe']})")
    print(f"  Min fold  (top-3): {min_sh:.3f}  (gate: >={GATE['min_fold_sharpe']})")
    print(f"  EW all 11 avg: {np.nanmean(ew_sharpes):.3f}")
    print(f"\n  {'Fold':>4}  {'Test Start':>12}  {'Test End':>12}  {'Top3 Sharpe':>13}  {'EW Sharpe':>11}  {'Return':>8}")
    for r in fold_results:
        print(f"    {r.fold:>2}  {str(r.test_start):>12}  {str(r.test_end):>12}  "
              f"{r.sharpe_top3:>13.3f}  {r.sharpe_ew:>11.3f}  {r.total_return_top3:>7.1f}%")

    if gate_ok:
        print("\n  *** BENCHMARK PASSES GATE ***")
        print("  SPDR rotation beats the Sharpe threshold without any ML.")
        print("  Recommended: adopt sector rotation as primary strategy.")
        print("  Single-name picker adds value ONLY IF it beats this benchmark.")
    else:
        print("\n  Benchmark fails gate — confirms Fold 4 is a structural regime problem,")
        print("  not a picker problem. Regime Allocator (Phase H) is justified.")

    # Email notification
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from app.notifications.notifier import _smtp_send
        fold_rows = "\n".join(
            f"  Fold {r.fold}: Top3 Sharpe={r.sharpe_top3:.3f}  EW={r.sharpe_ew:.3f}  return={r.total_return_top3:.1f}%"
            for r in fold_results
        )
        _smtp_send(
            subject=f"MrTrader SPDR Rotation WF: {verdict} (avg={avg_sh:.3f})",
            html_body=f"""
<h2>Phase G — SPDR Sector Rotation Benchmark</h2>
<p><b>{verdict}</b></p>
<ul>
  <li>Avg Sharpe (top-3): {avg_sh:.3f} (gate: ≥0.80)</li>
  <li>Min fold: {min_sh:.3f} (gate: ≥-0.30)</li>
  <li>EW all-11 avg: {np.nanmean(ew_sharpes):.3f}</li>
</ul>
<pre>{fold_rows}</pre>
<p><small>Config: top-{TOP_N}, {MOM_MONTHS}m momentum 1m skip, {COST_BPS}bps costs, 5-fold 6yr WF</small></p>
""",
        )
    except Exception as e:
        logger.warning("Email failed: %s", e)

    return 0 if gate_ok else 2


if __name__ == "__main__":
    sys.exit(main())
