"""
Intraday model backtest — translates AUC/precision into actual P&L.

The intraday XGBoost+LightGBM model (v15) predicts which stocks will move
≥ ATR in a given trading day. This script:
  1. Loads saved intraday model (latest by default)
  2. Fetches 5-min bars for a set of symbols over the test period
  3. For each day × symbol, builds the same features used at training time
  4. Scores with the model and selects top-N by predicted probability
  5. Simulates trades: enter at open+1 bar, exit at EOD or target/stop
  6. Reports P&L tearsheet: total return, Sharpe, win rate, avg hold P&L

Usage:
    python scripts/backtest_intraday.py --days 120 --top-n 5 --threshold 0.45
    python scripts/backtest_intraday.py --model-version 15 --days 60
"""

import argparse
import sys
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

RESET = "\033[0m"; BOLD = "\033[1m"; GREEN = "\033[32m"
YELLOW = "\033[33m"; RED = "\033[31m"; CYAN = "\033[36m"; DIM = "\033[2m"

def ok(m):    print(f"  {GREEN}OK{RESET}  {m}")
def warn(m):  print(f"  {YELLOW}!!{RESET}  {m}")
def info(m):  print(f"     {m}")
def header(t): print(f"\n{BOLD}{CYAN}-- {t} --{RESET}"); print(DIM + "-"*60 + RESET)
def fail(m):  print(f"  {RED}FAIL{RESET}  {m}")


HOLD_BARS   = 24   # 2 hours of 5-min bars (matches training labeling)
MIN_BARS    = 12   # min bars needed to compute features
N_SYMBOLS   = 100  # symbols to score per day (top liquid from S&P 500)

# S&P 500 large-cap sample — same universe as training
from app.utils.constants import SP_100_TICKERS


def load_intraday_model(model_dir: str, version: Optional[int] = None):
    import pickle
    model_path = Path(model_dir)
    if version is not None:
        path = model_path / f"intraday_v{version}.pkl"
    else:
        # find latest
        paths = sorted(model_path.glob("intraday_v*.pkl"),
                       key=lambda p: int(p.stem.split("_v")[1]))
        if not paths:
            raise FileNotFoundError(f"No intraday model in {model_dir}")
        path = paths[-1]
        version = int(path.stem.split("_v")[1])
    with open(path, "rb") as f:
        model = pickle.load(f)
    ok(f"Loaded intraday model v{version} from {path.name}")
    return model, version


def fetch_5min_data(symbols: List[str], start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
    """Fetch 5-min bars via IntradayModelTrainer (uses Parquet cache from training)."""
    from app.ml.intraday_training import IntradayModelTrainer
    trainer = IntradayModelTrainer()
    data = trainer._fetch_all(symbols, start, end, force_refresh=False)
    # Filter to symbols with sufficient data
    return {sym: df for sym, df in data.items() if df is not None and len(df) >= MIN_BARS * 5}


def fetch_daily_data(symbols: List[str], start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
    """Fetch daily bars for prior-day context features."""
    try:
        from app.data import get_provider
        start_dt = (start - timedelta(days=365)).date()
        end_dt = end.date()
        return get_provider("polygon").get_daily_bars_bulk(symbols, start_dt, end_dt)
    except Exception:
        return {}


def simulate_day(
    day_bars: pd.DataFrame,
    prob: float,
    threshold: float,
) -> Optional[dict]:
    """
    Simulate one intraday trade matching the cross-sectional training label.

    The model was trained with label=1 for stocks in the top-20% by max HIGH
    over HOLD_BARS bars. So the correct simulation is:
      Entry: open of first feature bar (feat_bars[-1] open)
      Exit:  close of last HOLD_BARS bar (matches HOLD_BARS window used in training)
    No target/stop — the model predicts relative performance, not absolute moves.
    """
    if len(day_bars) < MIN_BARS + HOLD_BARS:
        return None

    feat_bars  = day_bars.iloc[:-HOLD_BARS]
    hold_bars  = day_bars.iloc[-HOLD_BARS:]

    entry_px = float(feat_bars["close"].iloc[-1])   # entry at last-feature-bar close
    exit_px  = float(hold_bars["close"].iloc[-1])   # exit at end of hold window
    best_high = float(hold_bars["high"].max())       # best achievable (for reference)

    if entry_px <= 0:
        return None

    ret      = (exit_px  - entry_px) / entry_px
    best_ret = (best_high - entry_px) / entry_px

    return {
        "prob": prob,
        "entry": entry_px,
        "exit": exit_px,
        "ret": ret,
        "best_ret": best_ret,
        "outcome": "hold",
    }


def run_intraday_backtest(
    symbols: List[str],
    days: int,
    model_version: Optional[int],
    top_n: int,
    threshold: float,
    model_dir: str,
) -> None:
    header("Loading model")
    model, version = load_intraday_model(model_dir, model_version)

    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days + 30)  # extra for warmup

    header("Fetching 5-min bars")
    info(f"Symbols: {len(symbols)}  Days: {days}  Model: v{version}")
    symbols_data = fetch_5min_data(symbols, start_dt, end_dt)
    ok(f"Fetched 5-min data: {len(symbols_data)} symbols")

    if not symbols_data:
        fail("No 5-min data available")
        return

    header("Fetching daily bars (for prior-day context)")
    daily_data = fetch_daily_data(list(symbols_data.keys()), start_dt, end_dt)
    ok(f"Daily data: {len(daily_data)} symbols")

    # Build SPY daily lookup (for relative-strength features)
    spy_data = symbols_data.get("SPY") if "SPY" in symbols_data else symbols_data.get("spy")
    spy_by_day: Dict[date, pd.DataFrame] = {}
    if spy_data is not None:
        spy_idx = pd.DatetimeIndex(spy_data.index)
        for d in spy_idx.normalize().unique():
            mask = spy_idx.normalize() == d
            spy_by_day[d.date()] = spy_data.loc[mask]

    # Collect all trading days in the test window
    all_days_set: set = set()
    for df in symbols_data.values():
        for d in pd.DatetimeIndex(df.index).normalize().unique():
            all_days_set.add(d.date())
    all_days = sorted(all_days_set)

    # Use last `days` trading days as the test window
    test_days = all_days[-days:] if len(all_days) > days else all_days

    header("Simulating trades")
    info(f"Test window: {test_days[0]} to {test_days[-1]}  ({len(test_days)} days)")

    from app.ml.intraday_features import compute_intraday_features

    daily_trades: List[dict] = []
    daily_pnl: List[float] = []
    days_with_trades = 0

    for day in test_days:
        day_scores: List[Tuple[str, float, pd.DataFrame]] = []

        for sym, df in symbols_data.items():
            if sym in ("SPY", "spy"):
                continue
            df_idx = pd.DatetimeIndex(df.index)
            date_arr = np.array(df_idx.normalize().date)
            day_bars = df.iloc[date_arr == day]
            if len(day_bars) < MIN_BARS + HOLD_BARS:
                continue

            # Feature bars = first part of day (same as training)
            feat_bars = day_bars.iloc[:-HOLD_BARS]
            if len(feat_bars) < MIN_BARS:
                continue

            # Prior-day context
            prior_close = prior_high = prior_low = None
            prev_days = [d for d in sorted(all_days) if d < day and d in spy_by_day]
            if prev_days:
                prev_day = prev_days[-1]
                prev_date_arr = np.array(pd.DatetimeIndex(df.index).normalize().date)
                prev_bars = df.iloc[prev_date_arr == prev_day]
                if len(prev_bars) > 0:
                    prior_close = float(prev_bars["close"].iloc[-1])
                    prior_high  = float(prev_bars["high"].max())
                    prior_low   = float(prev_bars["low"].min())

            # Daily bars up to this day
            daily_df = daily_data.get(sym)
            daily_as_of = None
            if daily_df is not None and len(daily_df) > 0:
                d_idx = pd.DatetimeIndex(daily_df.index)
                d_date_arr = np.array(d_idx.normalize().date)
                daily_as_of = daily_df.iloc[d_date_arr < day]

            try:
                feats = compute_intraday_features(
                    feat_bars,
                    spy_by_day.get(day),
                    prior_close,
                    prior_day_high=prior_high,
                    prior_day_low=prior_low,
                    daily_bars=daily_as_of,
                )
            except Exception:
                continue

            if feats is None:
                continue

            X = np.array(list(feats.values()), dtype=float).reshape(1, -1)
            X = np.nan_to_num(X)
            try:
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(X)[0, 1])
                else:
                    # PortfolioSelectorModel wrapper returns (labels, proba)
                    _, proba = model.predict(X)
                    prob = float(proba[0])
            except Exception:
                continue

            if prob >= threshold:
                day_scores.append((sym, prob, day_bars))

        if not day_scores:
            continue

        # Select top-N by probability
        day_scores.sort(key=lambda x: x[1], reverse=True)
        selected = day_scores[:top_n]

        day_rets = []
        for sym, prob, day_bars in selected:
            trade = simulate_day(day_bars, prob, threshold)
            if trade is None:
                continue
            trade["symbol"] = sym
            trade["day"] = str(day)
            daily_trades.append(trade)
            day_rets.append(trade["ret"])

        if day_rets:
            daily_pnl.append(float(np.mean(day_rets)))
            days_with_trades += 1

    header("Results")

    if not daily_trades:
        fail("No trades generated — check threshold or data availability")
        return

    rets = np.array([t["ret"] for t in daily_trades])
    best_rets = np.array([t["best_ret"] for t in daily_trades])
    total_trades = len(daily_trades)
    wins = int((rets > 0).sum())
    losses = int((rets <= 0).sum())
    win_rate = wins / total_trades * 100
    avg_ret  = float(rets.mean()) * 100
    avg_win  = float(rets[rets > 0].mean()) * 100 if wins > 0 else 0
    avg_loss = float(rets[rets <= 0].mean()) * 100 if losses > 0 else 0
    avg_best = float(best_rets.mean()) * 100   # avg best achievable (HIGH over hold)

    # Daily Sharpe (annualised, 252 trading days)
    daily_arr = np.array(daily_pnl)
    sharpe = float(daily_arr.mean() / (daily_arr.std() + 1e-9)) * np.sqrt(252) if len(daily_arr) > 1 else 0.0
    cum_ret = float(np.prod(1 + daily_arr) - 1) * 100

    print(f"\n  {'Metric':<40} {'Value':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Total trades':<40} {total_trades:>10,}")
    print(f"  {'Days with trades':<40} {days_with_trades:>10,}")
    print(f"  {'Win rate (close > entry)':<40} {win_rate:>9.1f}%")
    print(f"  {'Avg trade return (close)':<40} {avg_ret:>+9.2f}%")
    print(f"  {'Avg winner':<40} {avg_win:>+9.2f}%")
    print(f"  {'Avg loser':<40} {avg_loss:>+9.2f}%")
    print(f"  {'Avg best achievable (max high)':<40} {avg_best:>+9.2f}%")
    print(f"  {'Hold window':<40} {HOLD_BARS:>9} bars (~{HOLD_BARS*5//60}h)")
    print(f"  {'Daily Sharpe (ann.)':<40} {sharpe:>10.2f}")
    print(f"  {'Cumulative return':<40} {cum_ret:>+9.1f}%")
    print()

    gate_color = GREEN if sharpe > 0.5 else YELLOW if sharpe > 0 else RED
    status = "PASS" if sharpe > 0.5 else "REVIEW" if sharpe > 0 else "FAIL"
    print(f"  {gate_color}{BOLD}>> {status} — Intraday Sharpe {sharpe:.2f}{RESET}")
    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Intraday model P&L backtest")
    parser.add_argument("--days", type=int, default=60, help="Trading days to backtest")
    parser.add_argument("--top-n", type=int, default=5, help="Top-N stocks per day")
    parser.add_argument("--threshold", type=float, default=0.45, help="Min probability to trade")
    parser.add_argument("--model-version", type=int, default=None)
    parser.add_argument("--model-dir", default="app/ml/models")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Override symbol list (default: SP_100_TICKERS)")
    args = parser.parse_args()

    symbols = args.symbols or (SP_100_TICKERS + ["SPY"])

    print()
    print(BOLD + "=" * 60 + RESET)
    print(BOLD + "  MrTrader — Intraday Model Backtest" + RESET)
    print("=" * 60)
    info(f"top_n={args.top_n}  days={args.days}  threshold={args.threshold}")
    info(f"symbols={len(symbols)}  model_dir={args.model_dir}")

    run_intraday_backtest(
        symbols=symbols,
        days=args.days,
        model_version=args.model_version,
        top_n=args.top_n,
        threshold=args.threshold,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()
