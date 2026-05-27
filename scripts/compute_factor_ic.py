"""
P0.3 — Information Coefficient (IC) analysis for the factor composite scorer.

Computes Spearman IC of composite factor scores vs. forward N-day realized returns
on a monthly rebalance grid over the full 6-year history (2019-01-01 to 2024-12-31).

Pass criteria: mean IC >= 0.02, IC t-stat >= 2.0

Usage:
    python scripts/compute_factor_ic.py [--forward-days 10] [--out docs/factor_ic.csv]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

START_DATE = "2019-01-01"
# WF-C1 (2026-05-27): END_DATE must end strictly before the earliest WF train_end
# so the resulting IC numbers / weights are not contaminated by any data inside
# a WF test fold. Fold 1's train ends 2021-04-27, so calibration ends one day
# earlier. Override via env or CLI for ad-hoc analysis; the default is the
# audit-safe pre-fold-1 window.
END_DATE   = "2021-04-26"
MIN_SYMBOLS = 50      # skip rebalance date if fewer symbols have scores
MIN_IC_PASS = 0.02
MIN_TSTAT_PASS = 2.0


def load_closes(min_date: str, max_date: str) -> pd.DataFrame:
    """Load daily closes from cache for all available symbols."""
    from app.data.cache import get_cache
    cache = get_cache()
    daily_dir = cache._dir / "daily"
    dfs = {}
    for path in sorted(daily_dir.glob("*.parquet")):
        sym = path.stem
        try:
            df = pd.read_parquet(path)
            if df.empty or "close" not in df.columns:
                continue
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.loc[min_date:max_date]
            if len(df) >= 300:
                dfs[sym] = df["close"]
        except Exception:
            pass
    closes = pd.DataFrame(dfs)
    closes.index = pd.to_datetime(closes.index)
    logger.info("Loaded closes: %d symbols, %d days", closes.shape[1], closes.shape[0])
    return closes


def load_bars_for_symbols(symbols: list[str], min_date: str, max_date: str) -> dict:
    """Load full OHLCV bars for volume_trend / range_expansion factors."""
    from app.data.cache import get_cache
    cache = get_cache()
    daily_dir = cache._dir / "daily"
    bars = {}
    for sym in symbols:
        path = daily_dir / f"{sym}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index().loc[min_date:max_date]
            if len(df) >= 300:
                bars[sym] = df
        except Exception:
            pass
    return bars


def monthly_rebalance_dates(closes: pd.DataFrame) -> list[pd.Timestamp]:
    """First trading day of each month within the closes index."""
    dates = closes.index
    months = dates.to_period("M")
    seen = set()
    result = []
    for dt, m in zip(dates, months):
        if m not in seen:
            seen.add(m)
            result.append(dt)
    # Need at least forward_days of future data — drop last entry
    return result[:-2]


def compute_ic_series(
    closes: pd.DataFrame,
    bars: dict,
    forward_days: int = 10,
) -> pd.DataFrame:
    from app.ml.factor_scorer import compute_composite_score

    rebal_dates = monthly_rebalance_dates(closes)
    records = []

    for as_of in rebal_dates:
        # Forward return: close[as_of + forward_days] / close[as_of] - 1
        future_idx = closes.index.searchsorted(as_of)
        fwd_idx = future_idx + forward_days
        if fwd_idx >= len(closes):
            continue

        fwd_price = closes.iloc[fwd_idx]
        cur_price = closes.loc[as_of]
        fwd_ret = (fwd_price / cur_price.replace(0, np.nan)) - 1.0
        fwd_ret = fwd_ret.dropna()

        if len(fwd_ret) < MIN_SYMBOLS:
            continue

        # Compute factor scores (PIT: closes up to as_of)
        try:
            scores = compute_composite_score(
                as_of=as_of,
                closes=closes,
                bars=bars,
                fundamentals=None,
                use_tier2=True,
            )
        except Exception as e:
            logger.warning("Score failed on %s: %s", as_of.date(), e)
            continue

        if scores.empty or len(scores) < MIN_SYMBOLS:
            continue

        # Align
        common = scores.index.intersection(fwd_ret.index)
        if len(common) < MIN_SYMBOLS:
            continue

        ic, pval = stats.spearmanr(scores[common], fwd_ret[common])
        records.append({
            "date": as_of.date(),
            "ic": ic,
            "pval": pval,
            "n_symbols": len(common),
        })
        logger.debug("  %s  IC=%.4f  n=%d", as_of.date(), ic, len(common))

    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-days", type=int, default=10)
    parser.add_argument("--out", default="docs/factor_ic.csv")
    args = parser.parse_args()

    logger.info("Loading price data %s → %s", START_DATE, END_DATE)
    closes = load_closes(START_DATE, END_DATE)
    if closes.empty:
        logger.error("No data loaded — check daily cache.")
        return 1

    syms = list(closes.columns)
    logger.info("Loading OHLCV bars for %d symbols (needed for tier-2 factors)", len(syms))
    bars = load_bars_for_symbols(syms, START_DATE, END_DATE)

    logger.info("Computing IC series (forward=%d days)...", args.forward_days)
    ic_df = compute_ic_series(closes, bars, forward_days=args.forward_days)

    if ic_df.empty:
        logger.error("No IC observations computed.")
        return 1

    mean_ic = ic_df["ic"].mean()
    std_ic  = ic_df["ic"].std()
    n_obs   = len(ic_df)
    tstat   = mean_ic / (std_ic / np.sqrt(n_obs)) if std_ic > 0 else 0.0
    pct_pos = (ic_df["ic"] > 0).mean() * 100

    # Output CSV
    out_path = ROOT / args.out
    ic_df.to_csv(out_path, index=False)
    logger.info("IC series saved → %s", out_path)

    # Summary
    pass_ic    = mean_ic >= MIN_IC_PASS
    pass_tstat = abs(tstat) >= MIN_TSTAT_PASS
    verdict    = "PASS" if (pass_ic and pass_tstat) else "FAIL"

    print("\n" + "=" * 60)
    print(f"FACTOR IC ANALYSIS — forward={args.forward_days}d")
    print("=" * 60)
    print(f"  Observations (monthly): {n_obs}")
    print(f"  Mean IC:    {mean_ic:+.4f}  (threshold >= {MIN_IC_PASS})")
    print(f"  Std IC:     {std_ic:.4f}")
    print(f"  t-statistic:{tstat:+.2f}  (threshold >= {MIN_TSTAT_PASS})")
    print(f"  % IC > 0:   {pct_pos:.1f}%")
    print(f"\n  Verdict: {verdict}")
    if not pass_ic:
        print("  !! Mean IC below threshold — factor has weak predictive signal")
    if not pass_tstat:
        print("  !! t-stat below threshold — signal not statistically significant")
    print("=" * 60 + "\n")

    return 0 if (pass_ic and pass_tstat) else 2


if __name__ == "__main__":
    sys.exit(main())
