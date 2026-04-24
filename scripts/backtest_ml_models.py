"""
CLI: backtest the swing and/or intraday ML models.

Usage:
  python scripts/backtest_ml_models.py
  python scripts/backtest_ml_models.py --model swing --years 2
  python scripts/backtest_ml_models.py --model intraday --days 55 --symbols AAPL MSFT NVDA
  python scripts/backtest_ml_models.py --model both
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Parquet price cache for swing backtest (avoids re-downloading on every run)
_PRICE_CACHE_DIR = ROOT / "app" / "ml" / "models" / "price_cache" / "swing"
_PRICE_CACHE_TTL_HOURS = 23  # refresh after market close


def _price_cache_path(symbols: list, years: int) -> Path:
    """Stable cache key: sorted symbols + years → short hash."""
    h = hashlib.md5((",".join(sorted(symbols)) + str(years)).encode()).hexdigest()[:10]
    return _PRICE_CACHE_DIR / f"swing_{years}yr_{h}.parquet"


def _load_price_cache(symbols: list, years: int):
    """Return (symbols_data, spy_prices) from Parquet cache if fresh, else None."""
    path = _price_cache_path(symbols, years)
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > _PRICE_CACHE_TTL_HOURS:
        return None
    try:
        combined = pd.read_parquet(path)
        symbols_data = {}
        spy_prices = None
        for sym in combined["_symbol"].unique():
            sub = combined[combined["_symbol"] == sym].drop(columns=["_symbol"])
            sub.index = pd.to_datetime(sub.index)
            if sym == "__SPY__":
                spy_prices = sub["close"]
            else:
                symbols_data[sym] = sub
        return symbols_data, spy_prices
    except Exception:
        return None


def _save_price_cache(symbols_data: dict, spy_prices, symbols: list, years: int):
    """Save symbols_data + SPY to Parquet cache."""
    try:
        _PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frames = []
        for sym, df in symbols_data.items():
            tmp = df.copy()
            tmp["_symbol"] = sym
            frames.append(tmp)
        if spy_prices is not None:
            spy_df = pd.DataFrame({"close": spy_prices})
            spy_df["_symbol"] = "__SPY__"
            frames.append(spy_df)
        if frames:
            combined = pd.concat(frames)
            path = _price_cache_path(symbols, years)
            combined.to_parquet(path)
    except Exception:
        pass  # cache write failure is non-fatal


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


def run_swing_backtest(symbols, years, ablation_kwargs=None):
    import yfinance as yf
    from app.backtesting.swing_backtest import SwingBacktester

    header("Swing Model Backtest  (daily bars)")
    info(f"Symbols: {len(symbols)}  |  History: {years} year(s)")

    end = datetime.now()
    start = end - timedelta(days=365 * years + 100)

    # Try Parquet cache first (avoids re-downloading on repeat runs)
    cached = _load_price_cache(symbols, years)
    if cached is not None:
        symbols_data, spy_prices = cached
        ok(f"Loaded {len(symbols_data)} symbols from Parquet cache (skipped download)")
    else:
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
        _save_price_cache(symbols_data, spy_prices, symbols, years)

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
    agent_sim = AgentSimulator(model=model, **(ablation_kwargs or {}))
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

    ok(f"Data loaded: {len(symbols_data)} symbols  |  window {start.date()} -> {end.date()}")

    # SPY 5-min for intraday features
    spy_data = None
    spy_sym_in_polygon = "SPY" in polygon_syms
    if spy_sym_in_polygon:
        spy_data = load_many(["SPY"], start=start.date(), end=end.date()).get("SPY")
    if spy_data is None:
        try:
            spy_raw = yf.download("SPY", period=f"{min(days, 55)}d", interval="5m",
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
        spy_d = yf.download("SPY", period=f"{min(days, 365)}d", interval="1d",
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
        warn(f"DB unavailable ({exc.__class__.__name__}), falling back to file glob")
    # Fallback: load latest pkl from disk
    import pickle
    from pathlib import Path
    model_dir = Path("app/ml/models")
    files = sorted(model_dir.glob(f"{model_name}_v*.pkl"))
    if not files:
        warn(f"No {model_name} model pkl found — train first")
        return None
    with open(files[-1], "rb") as f:
        obj = pickle.load(f)
    if hasattr(obj, "is_trained"):
        return obj
    from app.ml.model import PortfolioSelectorModel
    m = PortfolioSelectorModel(model_type="xgboost")
    version = int(files[-1].stem.split("_v")[-1])
    m.load(str(files[-1].parent), version, model_name=model_name)
    return m


def _save_backtest_json(model_name: str, agent_result, t_elapsed: float) -> None:
    """Save Tier 3 backtest results to results/ as JSON for programmatic comparison."""
    try:
        from app.database.session import get_session
        from app.database.models import ModelVersion
        db = get_session()
        row = db.query(ModelVersion).filter_by(model_name=model_name, status="ACTIVE") \
                .order_by(ModelVersion.version.desc()).first()
        version = row.version if row else 0
        db.close()
    except Exception:
        version = 0

    r = agent_result
    result = {
        "model": model_name,
        "version": version,
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "tier3": {
            "sharpe": round(getattr(r, "sharpe", 0) or 0, 4),
            "win_rate": round(getattr(r, "win_rate", 0) or 0, 4),
            "profit_factor": round(getattr(r, "profit_factor", 0) or 0, 4),
            "trades": getattr(r, "total_trades", 0),
            "total_return": round(getattr(r, "total_return", 0) or 0, 4),
            "max_drawdown": round(getattr(r, "max_drawdown", 0) or 0, 4),
            "annualized_return": round(getattr(r, "annualized_return", 0) or 0, 4),
        },
        "duration_seconds": round(t_elapsed, 1),
    }
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    fname = f"backtest_{model_name}_v{version}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    path = out_dir / fname
    path.write_text(json.dumps(result, indent=2))
    ok(f"Results saved -> {path}")


def main():
    from app.utils.constants import SP_500_TICKERS

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
    parser.add_argument("--sample", type=int, default=None,
                        help="Randomly sample N symbols from universe (for quick runs)")
    # Ablation experiment flags (all default to baseline v110 values)
    parser.add_argument("--stop-mult", type=float, default=0.5,
                        help="ATR stop multiplier (default: 0.5 = baseline). Ablation 26b: try 0.75")
    parser.add_argument("--target-mult", type=float, default=1.5,
                        help="ATR target multiplier (default: 1.5 = baseline). Ablation 26b: try 2.25")
    parser.add_argument("--min-confidence", type=float, default=0.50,
                        help="Min model confidence to enter (default: 0.50). Ablation 26c: try 0.60")
    parser.add_argument("--max-vol-pct", type=float, default=None,
                        help="Block entries where vol_percentile_52w > this (0-100). Ablation 26d: try 75")
    parser.add_argument("--download-only", action="store_true",
                        help="Pre-warm the Parquet price cache then exit (no backtesting). "
                             "Use before overnight runs so the cache is ready.")
    args = parser.parse_args()

    import random
    universe = SP_500_TICKERS  # matches live PM universe
    if args.symbols:
        symbols = args.symbols
    elif args.sample:
        symbols = random.sample(universe, min(args.sample, len(universe)))
    else:
        symbols = universe

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  MrTrader -- ML Model Backtest{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"  Model   : {args.model}")
    print(f"  Symbols : {len(symbols)}  (universe: {'custom' if args.symbols else f'SP500={len(universe)}' + (f', sampled {args.sample}' if args.sample else '')})")
    print(f"  Ablation: stop_mult={args.stop_mult}  target_mult={args.target_mult}  "
          f"min_conf={args.min_confidence}  max_vol_pct={args.max_vol_pct}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    if args.download_only:
        header("Download-only mode — pre-warming Parquet price cache")
        if args.model in ("swing", "both"):
            import yfinance as yf
            from datetime import datetime, timedelta
            end = datetime.now()
            start = end - timedelta(days=365 * args.years + 35)
            cached = _load_price_cache(symbols, args.years)
            if cached is not None:
                ok(f"Swing cache already fresh ({len(symbols)} symbols, {args.years}yr)")
            else:
                info(f"Downloading {len(symbols)} symbols ({args.years}yr) ...")
                try:
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
                    spy_raw.columns = [c.lower() for c in spy_raw.columns]
                    spy_prices = spy_raw["close"] if not spy_raw.empty else None
                    _save_price_cache(symbols_data, spy_prices, symbols, args.years)
                    ok(f"Swing cache warmed: {len(symbols_data)} symbols")
                except Exception as exc:
                    fail(f"Download failed: {exc}")
        ok("Done — cache is ready for backtesting")
        return

    t_start = time.time()

    ablation_kwargs = dict(
        atr_stop_mult=args.stop_mult,
        atr_target_mult=args.target_mult,
        min_confidence=args.min_confidence,
        max_vol_pct=args.max_vol_pct,
    )

    if args.model in ("swing", "both"):
        swing_result = run_swing_backtest(symbols, args.years, ablation_kwargs=ablation_kwargs)
        if swing_result is not None:
            _save_backtest_json("swing", swing_result, time.time() - t_start)

    if args.model in ("intraday", "both"):
        intra_result = run_intraday_backtest(symbols, args.days)
        if intra_result is not None:
            _save_backtest_json("intraday", intra_result, time.time() - t_start)

    elapsed = time.time() - t_start
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  Done in {elapsed:.0f}s{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")


if __name__ == "__main__":
    main()
