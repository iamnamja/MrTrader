"""
Phase 32 — MAE/MFE Trade Quality Diagnostic Report

Runs the Tier 3 backtest on v110 (or whatever model is active) and
breaks down stop-exit rate by 7 dimensions to identify the top factors
correlated with stop exits.

Usage:
    python scripts/analyze_trade_quality.py [--years N] [--sample N]

Output: Console table + CSV dump to results/trade_quality_YYYYMMDD.csv
"""
import argparse
import logging
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, window: int = 14) -> float:
    """14-day ATR as a pct of last close."""
    try:
        highs = df["high"].values[-window-1:].astype(float)
        lows = df["low"].values[-window-1:].astype(float)
        closes = df["close"].values[-window-1:].astype(float)
        if len(closes) < 2:
            return 0.02
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
        )
        atr = float(np.mean(tr[-window:])) if len(tr) >= window else float(np.mean(tr))
        return atr / max(float(closes[-1]), 1e-6)
    except Exception:
        return 0.02


def _ema(series: pd.Series, span: int) -> float:
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


def _decile(value: float, values: list) -> int:
    """Return 1-10 decile bucket."""
    if not values:
        return 5
    arr = np.array(values)
    pct = np.mean(arr <= value) * 100
    return max(1, min(10, int(pct / 10) + 1))


def _bucket_label(value: float, edges: list, labels: list) -> str:
    for i, e in enumerate(edges):
        if value <= e:
            return labels[i]
    return labels[-1]


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze(trades, symbols_data: dict, vix_data: pd.Series | None) -> pd.DataFrame:
    """
    Enrich each Trade with diagnostic features and return a DataFrame.

    Features computed per trade:
    - exit_type      : STOP / TARGET / TIME_EXIT / FORCE_CLOSE
    - is_stop        : 1 if STOP else 0
    - pm_score       : model confidence at entry (from trade.confidence if available)
    - entry_gap_atr  : (open - prior_close) / (atr * entry_price)
    - ema20_dist_atr : (entry_price - ema20) / (atr * entry_price)
    - atr_pct        : ATR as pct of entry price
    - hold_bars      : bars held
    - vix_at_entry   : VIX level on entry date (from spy proxy if not available)
    """
    rows = []
    for t in trades:
        sym = t.symbol
        df = symbols_data.get(sym)
        if df is None:
            continue

        entry_dt = t.entry_date if isinstance(t.entry_date, date) else t.entry_date.date()
        idx = df.index.date if hasattr(df.index, 'date') else pd.DatetimeIndex(df.index).date

        # Window up to entry date
        mask = idx <= entry_dt
        window = df.loc[mask].iloc[-60:]  # last 60 bars
        if len(window) < 5:
            continue

        closes = window["close"]
        entry_price = t.entry_price
        atr_pct = _atr(window)
        ema20 = _ema(closes, 20)

        # Entry gap: how far open is from prior close (in ATR units)
        prev_mask = idx < entry_dt
        prev_bars = df.loc[prev_mask]
        if len(prev_bars) >= 1:
            prior_close = float(prev_bars["close"].iloc[-1])
            gap_pct = (entry_price - prior_close) / max(prior_close, 1e-6)
            entry_gap_atr = gap_pct / max(atr_pct, 1e-6)
        else:
            entry_gap_atr = 0.0

        # Distance from EMA20 in ATR units
        ema20_dist_pct = (entry_price - ema20) / max(entry_price, 1e-6)
        ema20_dist_atr = ema20_dist_pct / max(atr_pct, 1e-6)

        # VIX on entry date
        vix_val = None
        if vix_data is not None:
            vix_dates = vix_data.index.date if hasattr(vix_data.index, 'date') else pd.DatetimeIndex(vix_data.index).date
            vix_on_day = vix_data.loc[vix_dates == entry_dt]
            if len(vix_on_day) > 0:
                vix_val = float(vix_on_day.iloc[-1])

        rows.append({
            "symbol": sym,
            "entry_date": entry_dt,
            "exit_date": t.exit_date,
            "exit_type": t.exit_reason,
            "is_stop": 1 if t.exit_reason == "STOP" else 0,
            "pnl_pct": t.pnl_pct,
            "hold_bars": t.hold_bars,
            "pm_score": getattr(t, "confidence", None),
            "entry_gap_atr": entry_gap_atr,
            "ema20_dist_atr": ema20_dist_atr,
            "atr_pct": atr_pct,
            "vix": vix_val,
        })

    return pd.DataFrame(rows)


def _print_breakdown(df: pd.DataFrame, col: str, label: str, n_buckets: int = 5) -> None:
    """Print stop rate breakdown by a numeric column, split into n_buckets."""
    if col not in df.columns or df[col].isna().all():
        print(f"\n  [{label}] — no data")
        return

    valid = df.dropna(subset=[col])
    if len(valid) == 0:
        return

    edges = np.percentile(valid[col], np.linspace(0, 100, n_buckets + 1))
    bucket_labels = [f"Q{i+1}" for i in range(n_buckets)]

    def assign_bucket(v):
        for i, e in enumerate(edges[1:]):
            if v <= e:
                return bucket_labels[i]
        return bucket_labels[-1]

    valid = valid.copy()
    valid["_bucket"] = valid[col].apply(assign_bucket)

    print(f"\n  {label}")
    print(f"  {'Bucket':<10} {'Trades':>7} {'StopRate':>9} {'WinRate':>9} {'AvgR':>8} {'AvgHold':>8}")
    print(f"  {'-'*55}")
    for bkt in bucket_labels:
        grp = valid[valid["_bucket"] == bkt]
        if len(grp) == 0:
            continue
        stop_rate = grp["is_stop"].mean()
        win_rate = (grp["pnl_pct"] > 0).mean()
        avg_r = grp["pnl_pct"].mean()
        avg_hold = grp["hold_bars"].mean()
        col_range = f"{grp[col].min():.3f}-{grp[col].max():.3f}"
        print(f"  {bkt:<10} {len(grp):>7} {stop_rate:>8.1%} {win_rate:>8.1%} {avg_r:>7.3f} {avg_hold:>7.1f}  [{col_range}]")


def _print_categorical(df: pd.DataFrame, col: str, label: str) -> None:
    """Print stop rate breakdown by a categorical column."""
    if col not in df.columns:
        return
    valid = df.dropna(subset=[col])
    print(f"\n  {label}")
    print(f"  {'Category':<20} {'Trades':>7} {'StopRate':>9} {'WinRate':>9} {'AvgR':>8}")
    print(f"  {'-'*55}")
    for cat, grp in valid.groupby(col):
        stop_rate = grp["is_stop"].mean()
        win_rate = (grp["pnl_pct"] > 0).mean()
        avg_r = grp["pnl_pct"].mean()
        print(f"  {str(cat):<20} {len(grp):>7} {stop_rate:>8.1%} {win_rate:>8.1%} {avg_r:>7.3f}")


def print_report(df: pd.DataFrame) -> None:
    n = len(df)
    stop_n = df["is_stop"].sum()
    print("\n" + "=" * 65)
    print(f"  TRADE QUALITY DIAGNOSTIC  —  {n} trades, {stop_n/n:.0%} stop exits")
    print("=" * 65)

    # 1. PM score decile (if available)
    if df["pm_score"].notna().any():
        _print_breakdown(df, "pm_score", "By PM Score (model confidence)")

    # 2. Entry gap (overnight gap in ATR units)
    _print_breakdown(df, "entry_gap_atr", "By Entry Gap (open vs prior close, ATR units)")

    # 3. Distance from EMA20 at entry
    _print_breakdown(df, "ema20_dist_atr", "By Distance from EMA20 at Entry (ATR units)")

    # 4. ATR percentile (stock volatility)
    _print_breakdown(df, "atr_pct", "By ATR% (stock volatility regime)")

    # 5. Time-to-stop / hold bars
    _print_breakdown(df, "hold_bars", "By Hold Duration (bars held)")

    # 6. VIX regime
    if df["vix"].notna().any():
        df = df.copy()
        df["vix_regime"] = pd.cut(
            df["vix"], bins=[0, 15, 20, 25, 30, 999],
            labels=["<15 (calm)", "15-20", "20-25", "25-30", ">30 (fear)"]
        )
        _print_categorical(df, "vix_regime", "By VIX Regime at Entry")

    # 7. Day-1 vs day-2 vs day-3+ stops
    df2 = df[df["is_stop"] == 1].copy()
    df2["stop_day"] = pd.cut(
        df2["hold_bars"], bins=[0, 1, 2, 3, 999],
        labels=["Day1", "Day2", "Day3", "Day4+"])
    print("\n  Stop Exit Timing (when do stops hit?)")
    print("  {:<10} {:>7} {:>11}".format("Day", "Count", "% of stops"))
    print("  " + "-" * 30)
    for lbl, grp in df2.groupby("stop_day", observed=True):
        print(f"  {str(lbl):<10} {len(grp):>7} {len(grp) / max(len(df2), 1):>10.1%}")

    print("\n" + "=" * 65)
    print("  TOP FINDINGS:")
    # Find biggest split
    for col, lbl in [("entry_gap_atr", "entry gap"), ("ema20_dist_atr", "EMA20 distance"),
                     ("atr_pct", "ATR volatility"), ("hold_bars", "hold duration")]:
        if col in df.columns and df[col].notna().any():
            valid = df.dropna(subset=[col])
            median = valid[col].median()
            low = valid[valid[col] <= median]["is_stop"].mean()
            high = valid[valid[col] > median]["is_stop"].mean()
            if abs(high - low) > 0.05:
                direction = "higher" if high > low else "lower"
                print(f"  • High {lbl}: stop rate {high:.0%} vs low {lbl}: {low:.0%} — "
                      f"higher {lbl} → {direction} stop rate")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Phase 32 — Trade quality diagnostic")
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--sample", type=int, default=None, help="Sample N symbols")
    parser.add_argument("--no-vix", action="store_true", help="Skip VIX download")
    args = parser.parse_args()

    from app.utils.constants import SP_500_TICKERS
    import random
    symbols = SP_500_TICKERS
    if args.sample:
        symbols = random.sample(symbols, min(args.sample, len(symbols)))

    print(f"Running Tier 3 backtest on {len(symbols)} symbols ({args.years}yr)...")
    print("(This will take a few minutes)")

    import time
    from datetime import datetime
    import yfinance as yf
    from app.backtesting.agent_simulator import AgentSimulator

    # Load model
    sys.path.insert(0, str(ROOT / "scripts"))
    from backtest_ml_models import _load_model, _load_price_cache, _save_price_cache

    model = _load_model("swing")
    if model is None:
        print("ERROR: No active swing model found. Train one first.")
        sys.exit(1)

    # Fetch bars (use cache if available)
    end = datetime.now()
    start = end - timedelta(days=365 * args.years + 35)

    cached = _load_price_cache(symbols, args.years)
    if cached is not None:
        symbols_data, spy_prices = cached
        print(f"Loaded {len(symbols_data)} symbols from cache")
    else:
        print("Downloading bars...")
        raw = yf.download(
            symbols, start=start.date().isoformat(), end=end.date().isoformat(),
            progress=False, auto_adjust=True, group_by="ticker",
        )
        symbols_data = {}
        for sym in symbols:
            try:
                df = raw[sym].dropna()
                df.columns = [c.lower() for c in df.columns]
                if not df.empty:
                    symbols_data[sym] = df
            except Exception:
                pass
        spy_raw = yf.download("SPY", start=start.date().isoformat(),
                              end=end.date().isoformat(), progress=False, auto_adjust=True)
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy_raw.columns = spy_raw.columns.get_level_values(0)
        spy_raw.columns = [c.lower() for c in spy_raw.columns]
        spy_prices = spy_raw["close"] if not spy_raw.empty else None
        _save_price_cache(symbols_data, spy_prices, symbols, args.years)
        print(f"Downloaded {len(symbols_data)} symbols")

    # Fetch VIX
    vix_data = None
    if not args.no_vix:
        try:
            vix_raw = yf.download(
                "^VIX", start=start.date().isoformat(),
                end=end.date().isoformat(), progress=False, auto_adjust=True)
            if isinstance(vix_raw.columns, pd.MultiIndex):
                vix_raw.columns = vix_raw.columns.get_level_values(0)
            vix_raw.columns = [c.lower() for c in vix_raw.columns]
            vix_data = vix_raw["close"]
            print(f"VIX data: {len(vix_data)} days")
        except Exception as e:
            print(f"VIX download failed ({e}) — skipping VIX breakdown")

    # Run Tier 3
    agent_start = start.date() + timedelta(days=420)
    print(f"Running agent sim {agent_start} -> {end.date()}...")
    t0 = time.time()
    agent_sim = AgentSimulator(model=model)
    result = agent_sim.run(
        symbols_data, spy_prices=spy_prices,
        start_date=agent_start, end_date=end.date(),
    )
    print(f"Simulation done in {time.time()-t0:.1f}s — {result.total_trades} trades")

    if result.total_trades == 0:
        print("No trades — cannot produce diagnostic report")
        sys.exit(0)

    # Analyze
    df = analyze(result.trades, symbols_data, vix_data)
    if df.empty:
        print("No enrichable trades")
        sys.exit(0)

    print_report(df)

    # Save CSV
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    fname = f"trade_quality_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(out_dir / fname, index=False)
    print(f"\nDetailed data saved -> results/{fname}")


if __name__ == "__main__":
    main()
