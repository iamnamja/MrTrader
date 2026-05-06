"""
Label Distribution Analysis -- Intraday Cross-Sectional Labels

Diagnoses how the absolute hurdle (CS_ABSOLUTE_HURDLE = 0.30%) affects
positive class size per day, especially on down/flat market days.

Outputs:
  - Per-day positive rate: with vs without hurdle
  - Positive class collapse days (< 5% positives with hurdle)
  - SPY return buckets: how hurdle affects each regime
  - Summary stats

Run from project root:
    python scripts/label_distribution_analysis.py
"""
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from datetime import date, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CS_ABSOLUTE_HURDLE = 0.0030
TOP_PCT = 0.80  # top 20% = percentile 80


def load_raw_returns(days: int = 730) -> pd.DataFrame:
    """Load intraday cache and compute per-symbol per-day 2h returns."""
    from app.data import intraday_cache

    logger.info("Loading intraday cache...")
    syms = intraday_cache.available_symbols()
    logger.info("Available symbols: %d", len(syms))

    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    cache = intraday_cache.load_many(syms)

    records = []
    for sym, df in cache.items():
        if df is None or len(df) < 10:
            continue
        df = df[df.index >= cutoff].copy()
        if len(df) < 10:
            continue

        # Group by trading day
        df.index = pd.DatetimeIndex(df.index)
        days_in_df = df.index.normalize().unique()

        for day in days_in_df:
            day_bars = df[df.index.normalize() == day]
            if len(day_bars) < 12:  # need at least 12 bars (1h) of data
                continue

            open_price = day_bars["open"].iloc[0]
            # 2h return: close of bar at ~2h after open (bar 24 of 5min = 120min)
            bar_2h_idx = min(24, len(day_bars) - 1)
            close_2h = day_bars["close"].iloc[bar_2h_idx]

            if open_price <= 0:
                continue

            ret_2h = (close_2h - open_price) / open_price
            records.append({
                "symbol": sym,
                "date": day.date(),
                "day_ordinal": day.toordinal(),
                "ret_2h": ret_2h,
            })

    df_raw = pd.DataFrame(records)
    logger.info("Loaded %d symbol-day observations across %d unique days",
                len(df_raw), df_raw["date"].nunique())
    return df_raw


def load_spy_daily(days: int = 730) -> pd.Series:
    """Load SPY daily returns from Polygon cache."""
    try:
        import yfinance as yf
        raw = yf.download("SPY", period=f"{days}d", progress=False)
        # yfinance may return MultiIndex columns -- flatten
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        spy = raw["Close"]
        if isinstance(spy, pd.DataFrame):
            spy = spy.iloc[:, 0]
        spy_ret = spy.pct_change().dropna()
        spy_ret.index = pd.DatetimeIndex(spy_ret.index).normalize()
        return spy_ret
    except Exception as e:
        logger.warning("Could not load SPY: %s", e)
        return pd.Series(dtype=float)


def analyze_labels(df_raw: pd.DataFrame, spy_ret: pd.Series) -> None:
    """Run label distribution analysis."""

    rows = []
    for day_val, grp in df_raw.groupby("day_ordinal"):
        rets = grp["ret_2h"].values
        n = len(rets)
        if n < 2:
            continue

        threshold = np.percentile(rets, TOP_PCT * 100)

        # Without hurdle: top-20% only
        pos_no_hurdle = (rets >= threshold).sum()

        # With hurdle: top-20% AND >= 0.30%
        pos_with_hurdle = ((rets >= threshold) & (rets >= CS_ABSOLUTE_HURDLE)).sum()

        day_date = date.fromordinal(int(day_val))
        spy_day_ret = None
        spy_ts = pd.Timestamp(day_date)
        if spy_ts in spy_ret.index:
            val = spy_ret[spy_ts]
            spy_day_ret = float(val.iloc[0] if hasattr(val, 'iloc') else val)

        rows.append({
            "date": day_date,
            "n_symbols": n,
            "pos_no_hurdle": pos_no_hurdle,
            "pos_with_hurdle": pos_with_hurdle,
            "pct_pos_no_hurdle": pos_no_hurdle / n,
            "pct_pos_with_hurdle": pos_with_hurdle / n,
            "hurdle_killed": pos_no_hurdle - pos_with_hurdle,
            "hurdle_kill_rate": (pos_no_hurdle - pos_with_hurdle) / max(pos_no_hurdle, 1),
            "spy_ret": spy_day_ret,
            "median_ret": float(np.median(rets)),
            "pct_above_hurdle_raw": float((rets >= CS_ABSOLUTE_HURDLE).mean()),
        })

    df = pd.DataFrame(rows).sort_values("date")

    # -- Summary stats ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("LABEL DISTRIBUTION ANALYSIS -- INTRADAY CROSS-SECTIONAL")
    print("=" * 70)
    print(f"Period: {df['date'].min()} to {df['date'].max()}")
    print(f"Total trading days analyzed: {len(df)}")
    print(f"Avg symbols/day: {df['n_symbols'].mean():.0f}")
    print()

    print("-- Positive class size ----------------------------------------------")
    print(f"  Without hurdle (top-20% only):  avg {df['pct_pos_no_hurdle'].mean()*100:.1f}%  "
          f"(expected ~20%)")
    print(f"  With hurdle (top-20% + >=0.30%): avg {df['pct_pos_with_hurdle'].mean()*100:.1f}%")
    print(f"  Hurdle reduces positives by:    avg {df['hurdle_kill_rate'].mean()*100:.1f}% of top-20%")
    print()

    # Days where hurdle wipes out nearly all positives
    collapse_days = df[df["pct_pos_with_hurdle"] < 0.05]
    print(f"-- Days with <5% positives after hurdle: {len(collapse_days)} / {len(df)} days "
          f"({len(collapse_days)/len(df)*100:.1f}%)")
    if len(collapse_days) > 0 and spy_ret is not None:
        spy_on_collapse = collapse_days["spy_ret"].dropna()
        print(f"   Avg SPY return on collapse days: {spy_on_collapse.mean()*100:.2f}%")
        print(f"   Collapse days with SPY < -1%: "
              f"{(collapse_days['spy_ret'].dropna() < -0.01).sum()}")
        print(f"   Collapse days with SPY -1% to 0%: "
              f"{((collapse_days['spy_ret'].dropna() >= -0.01) & (collapse_days['spy_ret'].dropna() < 0)).sum()}")
        print(f"   Collapse days with SPY > 0%: "
              f"{(collapse_days['spy_ret'].dropna() >= 0).sum()}")
    print()

    # SPY regime buckets
    if df["spy_ret"].notna().sum() > 10:
        print("-- Positive rate by SPY daily return bucket -------------------------")
        bins = [(-1, -0.02), (-0.02, -0.01), (-0.01, 0), (0, 0.01), (0.01, 0.02), (0.02, 1)]
        labels = ["SPY<-2%", "-2%->-1%", "-1%->0%", "0%->+1%", "+1%->+2%", "SPY>+2%"]
        for (lo, hi), lbl in zip(bins, labels):
            bucket = df[(df["spy_ret"] >= lo) & (df["spy_ret"] < hi)]
            if len(bucket) == 0:
                continue
            print(f"  {lbl:12s}  days={len(bucket):3d}  "
                  f"pos_no_hurdle={bucket['pct_pos_no_hurdle'].mean()*100:5.1f}%  "
                  f"pos_with_hurdle={bucket['pct_pos_with_hurdle'].mean()*100:5.1f}%  "
                  f"collapse={( bucket['pct_pos_with_hurdle'] < 0.05).sum():3d}d")
        print()

    # Recent 90 days (tariff regime)
    cutoff_90 = df["date"].max() - timedelta(days=90)
    recent = df[df["date"] >= cutoff_90]
    print(f"-- Last 90 days (tariff regime: {cutoff_90} -> {df['date'].max()}) ---")
    print(f"  Days analyzed: {len(recent)}")
    print(f"  Avg pos without hurdle: {recent['pct_pos_no_hurdle'].mean()*100:.1f}%")
    print(f"  Avg pos with hurdle:    {recent['pct_pos_with_hurdle'].mean()*100:.1f}%")
    print(f"  Collapse days (<5%):    {(recent['pct_pos_with_hurdle'] < 0.05).sum()}")
    print(f"  Avg % symbols above hurdle raw: {recent['pct_above_hurdle_raw'].mean()*100:.1f}%")
    print()

    # Worst 10 days
    print("-- 10 worst days (fewest positives with hurdle) ---------------------")
    worst = df.nsmallest(10, "pct_pos_with_hurdle")[
        ["date", "n_symbols", "pct_pos_no_hurdle", "pct_pos_with_hurdle",
         "hurdle_kill_rate", "spy_ret", "median_ret"]
    ]
    worst = worst.copy()
    worst["pct_pos_no_hurdle"] = (worst["pct_pos_no_hurdle"] * 100).round(1).astype(str) + "%"
    worst["pct_pos_with_hurdle"] = (worst["pct_pos_with_hurdle"] * 100).round(1).astype(str) + "%"
    worst["hurdle_kill_rate"] = (worst["hurdle_kill_rate"] * 100).round(0).astype(int).astype(str) + "%"
    worst["spy_ret"] = worst["spy_ret"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "--")
    worst["median_ret"] = (worst["median_ret"] * 100).round(3).astype(str) + "%"
    print(worst.to_string(index=False))
    print()

    # Alternative hurdle suggestions
    print("-- Alternative hurdle thresholds ------------------------------------")
    for h in [0.0010, 0.0020, 0.0025, 0.0030, 0.0040, 0.0050]:
        pos_h = df.apply(
            lambda r: ((df["date"] == r["date"]).sum()),  # placeholder
            axis=1
        )
        # Recompute for this hurdle from raw data
        collapse_count = 0
        total_pos_rate = []
        for _, row in df.iterrows():
            day_data = df_raw[df_raw["date"] == row["date"]]
            rets = day_data["ret_2h"].values
            if len(rets) < 2:
                continue
            thr = np.percentile(rets, 80)
            pos = ((rets >= thr) & (rets >= h)).sum()
            rate = pos / len(rets)
            total_pos_rate.append(rate)
            if rate < 0.05:
                collapse_count += 1
        avg_pos = np.mean(total_pos_rate) * 100 if total_pos_rate else 0
        print(f"  hurdle={h*100:.2f}%  avg_pos={avg_pos:.1f}%  "
              f"collapse_days={collapse_count}/{len(df)} ({collapse_count/len(df)*100:.1f}%)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    df_raw = load_raw_returns(days=730)
    spy_ret = load_spy_daily(days=730)
    analyze_labels(df_raw, spy_ret)
