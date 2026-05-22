"""
G-Pre / Phase G.3 — Fold 4 SPY calibration.

Computes SPY buy-and-hold Sharpe for each of the 5 WF fold test windows
(same dates used in run_factor_portfolio_walkforward.py).

If SPY Sharpe in Fold 4 is deeply negative (e.g. < -0.50), then the WF gate
floor of -0.30 on Fold 4 is impossible without explicit short positions.
This check calibrates whether the gate is achievable or needs adjustment.

Usage:
    python scripts/audit_fold4_spy.py
"""
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_FOLDS = 5
TOTAL_YEARS = 6
PURGE_DAYS = 10  # standard WF purge between train and test


def _download_spy(start: str, end: str) -> "pd.Series":
    import yfinance as yf
    raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("SPY download returned empty DataFrame")
    closes = raw["Close"]
    if hasattr(closes, "squeeze"):
        closes = closes.squeeze()
    return closes


def _sharpe(prices: "pd.Series", ann_factor: float = 252.0) -> float:
    rets = prices.pct_change().dropna()
    if len(rets) < 20:
        return float("nan")
    return float(rets.mean() / rets.std() * ann_factor ** 0.5)


def _total_return(prices: "pd.Series") -> float:
    return float(prices.iloc[-1] / prices.iloc[0] - 1) * 100


def main() -> None:
    end_all = date.today()
    start_all = end_all - timedelta(days=TOTAL_YEARS * 365 + 30)
    segment_days = int(TOTAL_YEARS * 365 / (N_FOLDS + 1))

    logger.info("Downloading SPY %s → %s ...", start_all, end_all)
    spy = _download_spy(start_all.isoformat(), end_all.isoformat())

    results = []
    for fold_idx in range(N_FOLDS):
        train_end_dt = end_all - timedelta(days=segment_days * (N_FOLDS - fold_idx))
        te_start = train_end_dt + timedelta(days=PURGE_DAYS)
        te_end = end_all - timedelta(days=segment_days * (N_FOLDS - fold_idx - 1))
        te_end = min(te_end, end_all)

        import pandas as pd
        fold_spy = spy.loc[
            (spy.index >= pd.Timestamp(te_start)) & (spy.index <= pd.Timestamp(te_end))
        ]
        if len(fold_spy) < 20:
            logger.warning("Fold %d: insufficient SPY data (%d bars)", fold_idx, len(fold_spy))
            continue

        sh = _sharpe(fold_spy)
        tr = _total_return(fold_spy)
        results.append((fold_idx, te_start, te_end, sh, tr, len(fold_spy)))
        logger.info(
            "Fold %d  %s → %s  SPY Sharpe=%.3f  SPY return=%.1f%%  bars=%d",
            fold_idx, te_start, te_end, sh, tr, len(fold_spy),
        )

    print("\n=== SPY FOLD CALIBRATION ===")
    print(f"{'Fold':>5}  {'Test Start':>12}  {'Test End':>12}  {'Sharpe':>8}  {'Return':>8}  {'Gate -0.30':>12}")
    for fold_idx, te_start, te_end, sh, tr, bars in results:
        gate_ok = "OK" if sh >= -0.30 else "FAIL"
        print(f"  {fold_idx:>3}  {str(te_start):>12}  {str(te_end):>12}  {sh:>8.3f}  {tr:>7.1f}%  {gate_ok:>12}")

    sharpes = [r[3] for r in results]
    print(f"\nSPY avg Sharpe: {np.mean(sharpes):.3f}  min fold: {np.min(sharpes):.3f}")

    fold4_sh = results[4][3] if len(results) > 4 else float("nan")
    if fold4_sh < -0.30:
        print(
            f"\nFOLD 4 WARNING: SPY itself earns Sharpe {fold4_sh:.3f} in this window."
        )
        print("  The WF gate floor of -0.30 on Fold 4 may be unachievable for long-only strategies.")
        print("  Short exposure or regime-gating required to pass.")
    else:
        print(f"\nFold 4 SPY Sharpe {fold4_sh:.3f} >= -0.30: gate is achievable without shorts.")


if __name__ == "__main__":
    main()
