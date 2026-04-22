"""
CLI: backtest the swing and/or intraday ML models.

Usage:
  python scripts/backtest_ml_models.py
  python scripts/backtest_ml_models.py --model swing --years 2
  python scripts/backtest_ml_models.py --model intraday --days 55 --symbols AAPL MSFT NVDA
  python scripts/backtest_ml_models.py --model both
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
DIM = "\033[2m"


def _c(colour, text):
    return f"{colour}{text}{RESET}"


def header(title):
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(_c(DIM, "-" * 60))


def ok(msg):
    print(f"  {GREEN}OK{RESET}  {msg}")


def warn(msg):
    print(f"  {YELLOW}!!{RESET}  {msg}")


def fail(msg):
    print(f"  {RED}FAIL{RESET}  {msg}")


def info(msg):
    print(f"     {msg}")


def _print_result(result):
    s = result.summary()
    print()
    ok(f"Model type     : {s['model_type']}")
    ok(f"Total trades   : {s['total_trades']}")
    ok(f"Win rate       : {s['win_rate']}")
    ok(f"Avg P&L/trade  : {s['avg_pnl_pct']}")
    ok(f"Avg hold       : {s['avg_hold_bars']} bars")
    ok(f"Sharpe ratio   : {s['sharpe_ratio']}")
    ok(f"Max drawdown   : {s['max_drawdown_pct']}")
    ok(f"Profit factor  : {s['profit_factor']}")
    ok(f"Total P&L      : {s['total_pnl']}")
    print()

    sharpe = result.sharpe_ratio
    if sharpe >= 1.0:
        print(f"  {GREEN}{BOLD}>> Strong signal -- Sharpe >= 1.0{RESET}")
    elif sharpe >= 0.5:
        print(f"  {YELLOW}{BOLD}>> Moderate -- Sharpe 0.5-1.0, consider more data{RESET}")
    else:
        print(f"  {RED}{BOLD}>> Weak signal -- Sharpe < 0.5, do not trade live{RESET}")

    win_rate = result.win_rate
    pf = result.profit_factor
    if win_rate >= 0.5 and pf >= 1.2:
        print(f"  {GREEN}{BOLD}>> Win rate {win_rate:.0%} + profit factor {pf:.2f}"
              f" -- proceed{RESET}")
    elif result.total_trades < 30:
        print(f"  {YELLOW}{BOLD}>> Only {result.total_trades} trades"
              f" -- expand data for significance{RESET}")

    if result.total_trades > 0:
        print()
        info("Exit breakdown:")
        by_reason = {}
        for t in result.trades:
            by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1
        for reason, cnt in sorted(by_reason.items(), key=lambda x: -x[1]):
            pct = cnt / result.total_trades
            bar_w = int(30 * pct)
            print(f"     {reason:<15} {'#' * bar_w} {cnt} ({pct:.0%})")


def run_swing_backtest(symbols, years):
    import yfinance as yf
    from datetime import datetime, timedelta
    from app.backtesting.swing_backtest import SwingBacktester

    header("Swing Model Backtest  (daily bars)")
    info(f"Symbols: {len(symbols)}  |  History: {years} year(s)")

    end = datetime.now()
    start = end - timedelta(days=365 * years + 100)
    info(f"Downloading {start.date()} -> {end.date()}...")

    symbols_data = {}
    try:
        raw = yf.download(
            symbols, start=start.date().isoformat(), end=end.date().isoformat(),
            interval="1d", progress=False, auto_adjust=True, group_by="ticker",
        )
        for sym in symbols:
            try:
                df = raw[sym].copy() if len(symbols) > 1 else raw.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                df = df.dropna(subset=["close"])
                if len(df) >= 50:
                    symbols_data[sym] = df
            except Exception:
                pass
    except Exception as exc:
        fail(f"Download failed: {exc}")
        return None

    # Fetch SPY for benchmark
    spy_prices = None
    try:
        spy_raw = yf.download("SPY", start=start.date().isoformat(),
                              end=end.date().isoformat(), interval="1d",
                              progress=False, auto_adjust=True)
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy_raw.columns = spy_raw.columns.get_level_values(0)
        spy_raw.columns = [c.lower() for c in spy_raw.columns]
        spy_prices = spy_raw["close"]
    except Exception:
        pass

    ok(f"Downloaded data for {len(symbols_data)} symbols")

    model = _load_model("swing")
    if model is None:
        warn("No swing model -- train first: python scripts/train_model.py")
        return None

    t0 = time.time()
    bt = SwingBacktester(model=model)
    raw_result = bt.run(symbols_data, fetch_fundamentals=False)
    elapsed = time.time() - t0
    ok(f"Raw backtest completed in {elapsed:.1f}s  ({raw_result.total_trades} trades)")

    # Tier 2: Portfolio-level simulation
    from app.backtesting.strategy_simulator import StrategySimulator
    sim = StrategySimulator()
    sim_result = sim.run(raw_result, spy_prices=spy_prices,
                         start_date=start.date(), end_date=end.date())
    sim_result.print_report()

    # Tier 3: Agent-driven simulation (PM + RM + Trader on historical bars)
    # Start 300 days into the data so EMA-200 and features have warm-up bars.
    from datetime import timedelta as _td
    agent_start = start.date() + _td(days=420)  # ~300 business days warm-up for EMA-200
    header(f"Tier 3 — Agent-Driven Simulation  (PM + RM + Trader)  [{agent_start} -> {end.date()}]")
    from app.backtesting.agent_simulator import AgentSimulator
    agent_sim = AgentSimulator(model=model)
    t0 = time.time()
    agent_result = agent_sim.run(
        symbols_data, spy_prices=spy_prices,
        start_date=agent_start, end_date=end.date(),
    )
    elapsed = time.time() - t0
    ok(f"Agent simulation completed in {elapsed:.1f}s  ({agent_result.total_trades} trades)")
    agent_result.print_report()

    return agent_result


def run_intraday_backtest(symbols, days):
    import yfinance as yf
    from datetime import datetime, timedelta
    from app.backtesting.intraday_backtest import IntradayBacktester
    from app.data.intraday_cache import load_many, available_symbols, cache_stats

    header("Intraday Model Backtest  (5-min bars)")
    info(f"Symbols: {len(symbols)}  |  History requested: {days} day(s)")

    end = datetime.now()
    start = end - timedelta(days=days + 5)

    # ── Prefer Polygon Parquet cache (2yr) over yfinance (55d max) ────────────
    polygon_syms = set(available_symbols())
    symbols_data = {}

    if polygon_syms:
        stats = cache_stats()
        info(f"Polygon cache: {stats['symbols']} symbols, {stats['total_bars']:,} bars, "
             f"oldest={stats['oldest_date']}")
        polygon_hit = load_many(
            [s for s in symbols if s in polygon_syms],
            start=start.date(), end=end.date(),
        )
        symbols_data.update(polygon_hit)
        missing = [s for s in symbols if s not in symbols_data]
        if missing:
            info(f"Falling back to yfinance for {len(missing)} symbols not in Polygon cache")
    else:
        missing = list(symbols)
        info("Polygon cache empty — using yfinance (≤55 days)")

    # yfinance fallback for symbols not in Polygon cache
    if missing:
        yf_days = min(days, 55)
        period_str = f"{yf_days}d"
        yf_start = end - timedelta(days=yf_days + 5)
        start = yf_start  # adjust effective start for reporting
        for sym in missing:
            try:
                df = yf.download(sym, period=period_str, interval="5m",
                                 progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                if len(df) >= 12:
                    symbols_data[sym] = df
            except Exception:
                pass

    ok(f"Data loaded: {len(symbols_data)} symbols  |  window {start.date()} → {end.date()}")

    # SPY 5-min for intraday features
    spy_data = None
    spy_sym_in_polygon = "SPY" in polygon_syms
    if spy_sym_in_polygon:
        spy_data = load_many(["SPY"], start=start.date(), end=end.date()).get("SPY")
    if spy_data is None:
        try:
            spy_raw = yf.download("SPY", period=f"{min(days,55)}d", interval="5m",
                                  progress=False, auto_adjust=True)
            if isinstance(spy_raw.columns, pd.MultiIndex):
                spy_raw.columns = spy_raw.columns.get_level_values(0)
            spy_raw.columns = [c.lower() for c in spy_raw.columns]
            spy_data = spy_raw if not spy_raw.empty else None
        except Exception:
            pass

    # Daily SPY for benchmark
    spy_daily = None
    try:
        spy_d = yf.download("SPY", period=f"{min(days,365)}d", interval="1d",
                            progress=False, auto_adjust=True)
        if isinstance(spy_d.columns, pd.MultiIndex):
            spy_d.columns = spy_d.columns.get_level_values(0)
        spy_d.columns = [c.lower() for c in spy_d.columns]
        spy_daily = spy_d["close"]
    except Exception:
        pass

    model = _load_model("intraday")
    if model is None:
        warn("No intraday model -- train first (IntradayModelTrainer)")
        return None

    t0 = time.time()
    bt = IntradayBacktester(model=model)
    raw_result = bt.run(symbols_data, spy_data)
    elapsed = time.time() - t0
    ok(f"Raw backtest completed in {elapsed:.1f}s  ({raw_result.total_trades} trades)")

    from app.backtesting.strategy_simulator import StrategySimulator
    from datetime import timedelta
    sim = StrategySimulator(position_budget_pct=0.03)
    sim_result = sim.run(raw_result, spy_prices=spy_daily,
                         start_date=start.date(), end_date=end.date())
    sim_result.print_report()

    # Tier 3: Agent-driven intraday simulation
    header("Tier 3 — Intraday Agent-Driven Simulation  (PM + RM + Trader)")
    from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
    agent_sim = IntradayAgentSimulator(model=model)
    t0 = time.time()
    agent_result = agent_sim.run(
        symbols_data, spy_data=spy_data, spy_prices=spy_daily,
        start_date=start.date(), end_date=end.date(),
    )
    elapsed = time.time() - t0
    ok(f"Intraday agent simulation completed in {elapsed:.1f}s  ({agent_result.total_trades} trades)")
    agent_result.print_report()

    return sim_result


def _load_model(model_name):
    import pickle
    from pathlib import Path
    try:
        from app.database.models import ModelVersion
        from app.database.session import get_session

        db = get_session()
        try:
            latest = (
                db.query(ModelVersion)
                .filter_by(model_name=model_name, status="ACTIVE")
                .order_by(ModelVersion.version.desc())
                .first()
            )
            if not latest or not latest.model_path:
                return None
            path = Path(latest.model_path)
            directory = str(path.parent)
            version = latest.version

            # Try loading as LambdaRankModel / ThreeStageModel (self-contained pickle)
            if path.exists():
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                if hasattr(obj, "is_trained"):
                    return obj

            # PortfolioSelectorModel: pkl contains raw XGBoost, needs wrapper load
            from app.ml.model import PortfolioSelectorModel
            m = PortfolioSelectorModel(model_type="xgboost")
            m.load(directory, version, model_name=model_name)
            return m
        finally:
            db.close()
    except Exception as exc:
        warn(f"Could not load {model_name} model: {exc}")
    return None


def main():
    from app.utils.constants import SP_100_TICKERS

    parser = argparse.ArgumentParser(
        description="MrTrader ML model backtest runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=["swing", "intraday", "both"], default="both",
    )
    parser.add_argument("--years", type=int, default=2,
                        help="Years of daily history for swing (default: 2)")
    parser.add_argument("--days", type=int, default=730,
                        help="Days of 5-min history for intraday (default: 730 via Polygon cache; "
                             "falls back to 55 days via yfinance if Polygon cache is empty)")
    parser.add_argument("--symbols", nargs="+", default=None, metavar="TICKER")
    args = parser.parse_args()

    symbols = args.symbols or SP_100_TICKERS[:30]

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  MrTrader -- ML Model Backtest{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"  Model   : {args.model}")
    print(f"  Symbols : {len(symbols)}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    t_start = time.time()

    if args.model in ("swing", "both"):
        run_swing_backtest(symbols, args.years)

    if args.model in ("intraday", "both"):
        run_intraday_backtest(symbols, args.days)

    elapsed = time.time() - t_start
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  Done in {elapsed:.0f}s{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")


if __name__ == "__main__":
    main()
