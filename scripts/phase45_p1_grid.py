"""
Phase 45 — Phase 1: Stop/Target Structure Grid Search (v119, inference-only)

Tests 3 stop/target configs × 3 walk-forward folds using v110 model unchanged.
Outputs results to docs/phase45/v119_results.md and results/phase45_p1_grid.json.

Configs (stop_mult decoupled from max_hold per implementation note):
  Baseline : stop=0.5, target=1.5  (gambler's ruin P(stop)=75%)
  Config A  : stop=0.75, target=1.25 (P(stop)=62.5%)
  Config B  : stop=0.5,  target=1.0  (P(stop)=67%)

Usage:
    python scripts/phase45_p1_grid.py
    python scripts/phase45_p1_grid.py --folds 3 --years 5
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys as _sys

# Force UTF-8 stdout on Windows so unicode chars don't crash cp1252
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

CONFIGS = [
    {"name": "Baseline", "stop_mult": 0.5, "target_mult": 1.5,
     "ruin_prob": 75.0, "rr": "3.0:1"},
    {"name": "Config A", "stop_mult": 0.75, "target_mult": 1.25,
     "ruin_prob": 62.5, "rr": "1.67:1"},
    {"name": "Config B", "stop_mult": 0.5, "target_mult": 1.0,
     "ruin_prob": 67.0, "rr": "2.0:1"},
]

BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"


def _header(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")


def _subheader(msg: str) -> None:
    print(f"\n{BOLD}  {msg}{RESET}")
    print(f"{DIM}  {'-' * 56}{RESET}")


def _ok(msg: str) -> None:
    print(f"  {GREEN}OK{RESET}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}!!{RESET}  {msg}")


def _load_swing_model(version: int = 0):
    import pickle
    from pathlib import Path as P

    model_dir = P("app/ml/models")
    if version > 0:
        path = model_dir / f"swing_v{version}.pkl"
        if not path.exists():
            raise RuntimeError(f"swing_v{version}.pkl not found")
    else:
        # Numeric sort to get true latest (not alphabetical which puts v9x last)
        files = sorted(model_dir.glob("swing_v*.pkl"),
                       key=lambda p: int(p.stem.split("_v")[-1]))
        if not files:
            raise RuntimeError("No swing model pkl found in app/ml/models/")
        path = files[-1]
        version = int(path.stem.split("_v")[-1])
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if hasattr(obj, "is_trained"):
        return obj, version
    from app.ml.model import PortfolioSelectorModel
    m = PortfolioSelectorModel(model_type="xgboost")
    m.load(str(path.parent), version, model_name="swing")
    return m, version


def _download_data(symbols: List[str], years: int) -> Tuple[Dict, pd.Series]:
    end_all = datetime.now()
    start_all = end_all - timedelta(days=years * 365 + 30)
    print(f"  Downloading {len(symbols)} symbols ({start_all.date()} -> {end_all.date()})...",
          flush=True)
    symbols_data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = yf.download(sym, start=start_all.date().isoformat(),
                             end=end_all.date().isoformat(), progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            if len(df) >= 210:
                symbols_data[sym] = df
        except Exception:
            pass
    spy_raw = yf.download("SPY", start=start_all.date().isoformat(),
                          end=end_all.date().isoformat(), progress=False, auto_adjust=True)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)
    spy_raw.columns = [c.lower() for c in spy_raw.columns]
    spy_prices = spy_raw["close"]
    _ok(f"Loaded {len(symbols_data)} symbols")
    return symbols_data, spy_prices, start_all, end_all


def _build_folds(n_folds: int, start_all: datetime, end_all: datetime
                 ) -> List[Tuple[date, date, date, date]]:
    segment_days = int((end_all - start_all).days / (n_folds + 1))
    folds = []
    for fold_idx in range(n_folds):
        train_end_dt = end_all - timedelta(days=segment_days * (n_folds - fold_idx))
        test_start_dt = train_end_dt + timedelta(days=1)
        test_end_dt = train_end_dt + timedelta(days=segment_days)
        folds.append((
            start_all.date(),
            train_end_dt.date(),
            test_start_dt.date(),
            min(test_end_dt.date(), end_all.date()),
        ))
    return folds


def run_fold(model, symbols_data, spy_prices, fold, stop_mult, target_mult):
    from app.backtesting.agent_simulator import AgentSimulator
    tr_start, tr_end, te_start, te_end = fold
    sim = AgentSimulator(
        model=model,
        atr_stop_mult=stop_mult,
        atr_target_mult=target_mult,
    )
    result = sim.run(
        symbols_data,
        start_date=te_start,
        end_date=te_end,
        spy_prices=spy_prices,
    )
    stop_exits = result.exit_breakdown.get("STOP", 0)
    target_exits = result.exit_breakdown.get("TARGET", 0)
    time_exits = result.exit_breakdown.get("MAX_HOLD", 0) + result.exit_breakdown.get("TIME", 0)
    total = max(result.total_trades, 1)
    return {
        "test_start": str(te_start),
        "test_end": str(te_end),
        "trades": result.total_trades,
        "sharpe": round(result.sharpe_ratio, 3),
        "win_rate": round(result.win_rate, 3),
        "profit_factor": round(result.profit_factor, 3),
        "total_return": round(result.total_return_pct, 3),
        "max_drawdown": round(result.max_drawdown_pct, 3),
        "stop_exits_pct": round(stop_exits / total, 3),
        "target_exits_pct": round(target_exits / total, 3),
        "time_exits_pct": round(time_exits / total, 3),
        "avg_hold": round(getattr(result, "avg_hold_bars", 0), 2),
    }


def write_results_md(all_results: List[dict], winning_config: dict, output_path: Path):
    lines = [
        "# Phase 45 — Phase 1 (v119): Stop/Target Structure Grid Results",
        "",
        f"**Date:** {date.today()}  ",
        f"**Model:** v110 (unchanged — inference-only test)  ",
        f"**Folds:** 3 walk-forward OOS folds  ",
        "",
        "## Config Definitions",
        "",
        "| Config | stop_mult | target_mult | R:R | Gambler's Ruin P(stop) |",
        "|---|---|---|---|---|",
    ]
    for cfg in CONFIGS:
        lines.append(
            f"| {cfg['name']} | {cfg['stop_mult']} | {cfg['target_mult']} "
            f"| {cfg['rr']} | {cfg['ruin_prob']}% |"
        )
    lines += ["", "## Per-Config Walk-Forward Results", ""]

    for cfg_result in all_results:
        cfg = cfg_result["config"]
        folds = cfg_result["folds"]
        sharpes = [f["sharpe"] for f in folds]
        avg_sharpe = round(sum(sharpes) / len(sharpes), 3)
        avg_trades = round(sum(f["trades"] for f in folds) / len(folds))
        avg_stop = round(sum(f["stop_exits_pct"] for f in folds) / len(folds), 3)
        avg_win = round(sum(f["win_rate"] for f in folds) / len(folds), 3)
        avg_pf = round(sum(f["profit_factor"] for f in folds) / len(folds), 3)

        lines += [
            f"### {cfg['name']}  (stop={cfg['stop_mult']}×, target={cfg['target_mult']}×)",
            "",
            f"**Avg walk-forward Sharpe: {avg_sharpe}**  ",
            f"Avg trades/fold: {avg_trades}  |  Avg stop exits: {avg_stop:.1%}  "
            f"|  Avg win rate: {avg_win:.1%}  |  Avg profit factor: {avg_pf}",
            "",
            "| Fold | OOS Period | Trades | Win% | Sharpe | PF | Stop% | Target% |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for i, f in enumerate(folds, 1):
            sharpe_str = f"{f['sharpe']:+.2f}"
            lines.append(
                f"| {i} | {f['test_start']} -> {f['test_end']} "
                f"| {f['trades']} | {f['win_rate']:.1%} | {sharpe_str} "
                f"| {f['profit_factor']:.2f} | {f['stop_exits_pct']:.1%} "
                f"| {f['target_exits_pct']:.1%} |"
            )
        lines.append("")

    lines += [
        "## Summary Comparison",
        "",
        "| Config | Avg Sharpe | Min Fold | Avg Trades | Avg Stop% |",
        "|---|---|---|---|---|",
    ]
    for cfg_result in all_results:
        cfg = cfg_result["config"]
        folds = cfg_result["folds"]
        sharpes = [f["sharpe"] for f in folds]
        lines.append(
            f"| {cfg['name']} | {sum(sharpes)/len(sharpes):+.3f} "
            f"| {min(sharpes):+.3f} | {sum(f['trades'] for f in folds)//len(folds)} "
            f"| {sum(f['stop_exits_pct'] for f in folds)/len(folds):.1%} |"
        )

    lines += [
        "",
        "## Winning Config for Phase 2",
        "",
        f"**{winning_config['name']}**: stop_mult={winning_config['stop_mult']}, "
        f"target_mult={winning_config['target_mult']}",
        "",
        "Phase 2 will use these multipliers for the path_quality label construction.",
        "",
        "## Gate Assessment",
        "",
    ]

    # Check if any config beats baseline by ≥ 0.10 Sharpe
    baseline_sharpes = [f["sharpe"] for f in all_results[0]["folds"]]
    baseline_avg = sum(baseline_sharpes) / len(baseline_sharpes)
    winning_sharpes = [f["sharpe"] for f in
                       next(r for r in all_results if r["config"]["name"] == winning_config["name"])["folds"]]
    winning_avg = sum(winning_sharpes) / len(winning_sharpes)
    delta = winning_avg - baseline_avg

    if winning_config["name"] == "Baseline":
        lines.append(
            "Neither Config A nor B beat Baseline by ≥ +0.10 Sharpe. "
            "**Structure is not the bottleneck** — alpha quality in Phase 2 carries more weight."
        )
    else:
        lines.append(
            f"**{winning_config['name']}** beats Baseline by {delta:+.3f} Sharpe. "
            f"Structure improvement confirmed — locking in for Phase 2."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"\n  Results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 45 P1: stop/target grid search")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--model-version", type=int, default=110,
                        help="Swing model version to use (default: 110 = v110 best model)")
    args = parser.parse_args()

    _header("Phase 45 — P1: Stop/Target Structure Grid (v119, inference-only)")
    print(f"  Configs: {len(CONFIGS)}  |  Folds: {args.folds}  |  Years: {args.years}")
    print(f"  Total backtests: {len(CONFIGS) * args.folds}")

    # Load model
    model, version = _load_swing_model(version=args.model_version)
    _ok(f"Model: swing v{version}")

    # Load symbols
    from app.utils.constants import SP_100_TICKERS
    symbols = args.symbols or list(SP_100_TICKERS)

    # Download data once (shared across all configs)
    symbols_data, spy_prices, start_all, end_all = _download_data(symbols, args.years)
    folds = _build_folds(args.folds, start_all, end_all)
    _ok(f"Folds: {args.folds} | Test periods:")
    for i, (_, _, te_s, te_e) in enumerate(folds, 1):
        print(f"    Fold {i}: {te_s} -> {te_e}")

    all_results = []
    for cfg in CONFIGS:
        _subheader(f"{cfg['name']} -- stop={cfg['stop_mult']}x, target={cfg['target_mult']}x "
                   f"(P(stop)~{cfg['ruin_prob']:.0f}%)")
        fold_results = []
        for i, fold in enumerate(folds, 1):
            print(f"    Running fold {i}/{args.folds}...", end="", flush=True)
            t0 = time.time()
            r = run_fold(model, symbols_data, spy_prices, fold,
                         cfg["stop_mult"], cfg["target_mult"])
            elapsed = time.time() - t0
            sharpe_col = GREEN if r["sharpe"] > 0 else RED
            print(f"\r    Fold {i}: {r['test_start']}->{r['test_end']}  "
                  f"trades={r['trades']}  "
                  f"Sharpe={sharpe_col}{r['sharpe']:+.2f}{RESET}  "
                  f"stop={r['stop_exits_pct']:.0%}  ({elapsed:.0f}s)")
            fold_results.append(r)

        sharpes = [f["sharpe"] for f in fold_results]
        avg = sum(sharpes) / len(sharpes)
        col = GREEN if avg > 0 else RED
        print(f"  {BOLD}-> Avg Sharpe: {col}{avg:+.3f}{RESET}")
        all_results.append({"config": cfg, "folds": fold_results, "avg_sharpe": avg})

    # Determine winner: highest avg Sharpe, with trade count ≥ 80% of baseline
    baseline_trades = sum(f["trades"] for f in all_results[0]["folds"]) / args.folds
    winning_config = CONFIGS[0]  # default to baseline
    best_avg = all_results[0]["avg_sharpe"]
    for cfg_result in all_results[1:]:
        avg_trades = sum(f["trades"] for f in cfg_result["folds"]) / args.folds
        beats_by = cfg_result["avg_sharpe"] - all_results[0]["avg_sharpe"]
        if beats_by >= 0.10 and avg_trades >= baseline_trades * 0.80:
            if cfg_result["avg_sharpe"] > best_avg:
                best_avg = cfg_result["avg_sharpe"]
                winning_config = cfg_result["config"]

    _subheader("RESULTS SUMMARY")
    print(f"  {'Config':<12} {'Avg Sharpe':>12} {'Min Fold':>10} {'Avg Trades':>12} {'Avg Stop%':>10}")
    print(f"  {'-'*58}")
    for r in all_results:
        sharpes = [f["sharpe"] for f in r["folds"]]
        avg_t = sum(f["trades"] for f in r["folds"]) / args.folds
        avg_s = r["avg_sharpe"]
        col = GREEN if avg_s > all_results[0]["avg_sharpe"] + 0.09 else RESET
        marker = " << WINNER" if r["config"]["name"] == winning_config["name"] else ""
        print(f"  {col}{r['config']['name']:<12} {avg_s:>+12.3f} "
              f"{min(sharpes):>+10.3f} {avg_t:>12.0f} "
              f"{sum(f['stop_exits_pct'] for f in r['folds'])/args.folds:>10.1%}{RESET}{marker}")

    _ok(f"Winning config: {winning_config['name']} "
        f"(stop={winning_config['stop_mult']}×, target={winning_config['target_mult']}×)")

    # Save JSON
    output_dir = Path("results/phase45")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"p1_grid_{date.today().isoformat()}.json"
    json_path.write_text(json.dumps({
        "date": str(date.today()),
        "model_version": version,
        "folds": args.folds,
        "years": args.years,
        "winning_config": winning_config,
        "results": [
            {"config": r["config"], "folds": r["folds"], "avg_sharpe": r["avg_sharpe"]}
            for r in all_results
        ],
    }, indent=2))
    _ok(f"JSON saved to {json_path}")

    # Write markdown report
    md_path = Path("docs/phase45/v119_results.md")
    write_results_md(all_results, winning_config, md_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
