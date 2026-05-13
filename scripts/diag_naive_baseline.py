"""
scripts/diag_naive_baseline.py — Phase A3: Naive baseline strategies.

Runs two simple strategies on the same universe and time window as the WF,
applying identical cost assumptions, to determine whether ML is adding value.

Strategies:
  B1 — Top-20% 60d momentum, equal-weight, monthly rebalance
  B2 — Long SPY when SPY > 200d MA, else cash
  B3 — B1 gated by B2 (momentum portfolio only when SPY > MA200)

If any baseline achieves avg_sharpe >= 0.5 where the ML WF produced negative
Sharpes, the ML is destroying (not extracting) signal from the same data.

Kill criterion: if B1 or B3 Sharpe > best ML WF Sharpe (currently -0.085 from
v188 / +0.106 from v186) -> ML provides no alpha lift; go to Phase C.

Usage:
    python scripts/diag_naive_baseline.py
    python scripts/diag_naive_baseline.py --start 2019-01-01 --end 2026-05-09
    python scripts/diag_naive_baseline.py --cost-bps 5 --top-pct 0.20

Output:
    data/diagnostics/naive_baseline/<timestamp>/b1_equity.parquet
    data/diagnostics/naive_baseline/<timestamp>/b2_equity.parquet
    data/diagnostics/naive_baseline/<timestamp>/b3_equity.parquet
    data/diagnostics/naive_baseline/<timestamp>/baseline_metrics.csv
    data/diagnostics/naive_baseline/<timestamp>/baseline_summary.md
    data/diagnostics/naive_baseline/<timestamp>/manifest.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ml.retrain_config import (
    MAX_WORKERS,
    SACRED_HOLDOUT_START,
    _parse_sacred_holdout_start,
)

os.environ.setdefault("OMP_NUM_THREADS", str(MAX_WORKERS))

import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_HOLDOUT = _parse_sacred_holdout_start()
_DEFAULT_END = _HOLDOUT - timedelta(days=1)
_DEFAULT_START = date(2019, 1, 1)

# Reference ML WF results (from ML_EXPERIMENT_LOG.md, honest walk-forward)
_ML_REFERENCE = {
    "v186": 0.106,   # best recent, 3 folds
    "v188": -0.085,  # macro-dominated
    "v190": -0.620,
    "v195": -0.546,
}


# ── Metric helpers — reuse gate formulas exactly ──────────────────────────────

def _sharpe(daily_returns: pd.Series, annualise: bool = True) -> float:
    """Annualised Sharpe (same as WF simulation: sqrt(252) × mean/std)."""
    if daily_returns.empty or daily_returns.std() == 0:
        return 0.0
    sr = daily_returns.mean() / daily_returns.std()
    return float(sr * np.sqrt(252)) if annualise else float(sr)


def _max_drawdown(equity: pd.Series) -> float:
    """Max drawdown as positive fraction (0.2 = 20% drawdown)."""
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max.replace(0, np.nan)
    return float(-dd.min()) if not dd.empty else 0.0


def _calmar(equity: pd.Series, net_returns: pd.Series) -> float:
    from scripts.walkforward.cost_models import cost_from_turnover
    mdd = _max_drawdown(equity)
    if mdd <= 0:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)
    return float(cagr / mdd)


def _compute_metrics(equity: pd.Series, net_returns: pd.Series,
                     turnover_series: Optional[pd.Series] = None) -> Dict:
    if equity.empty or net_returns.empty:
        return {}
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    cagr = float((1 + total_ret) ** (1 / years) - 1)
    return {
        "sharpe": round(_sharpe(net_returns), 4),
        "max_drawdown": round(_max_drawdown(equity), 4),
        "calmar": round(_calmar(equity, net_returns), 4),
        "total_return": round(total_ret, 4),
        "cagr": round(cagr, 4),
        "avg_daily_turnover": round(float(turnover_series.mean()), 4) if turnover_series is not None else None,
        "n_days": len(net_returns),
    }


# ── Strategy B1: Top-20% momentum, monthly rebalance ─────────────────────────

def run_momentum_baseline(
    bars_map: Dict[str, pd.DataFrame],
    start: date,
    end: date,
    lookback: int = 60,
    top_pct: float = 0.20,
    cost_bps_per_side: float = 5.0,
) -> pd.DataFrame:
    """Simulate top-X% 60d momentum portfolio with monthly rebalancing.

    Returns daily DataFrame with columns:
        date, gross_ret, turnover, cost, net_ret, equity, n_holdings
    """
    # Build cross-sectional returns table
    close_dfs = {}
    for sym, df in bars_map.items():
        col = "close" if "close" in df.columns else "Close"
        if col not in df.columns:
            continue
        s = df[col].sort_index()
        s.index = pd.to_datetime(s.index)
        close_dfs[sym] = s

    if not close_dfs:
        return pd.DataFrame()

    prices = pd.DataFrame(close_dfs).sort_index()
    prices = prices.loc[str(start):str(end)]

    # Monthly rebalance dates = last trading day of each month
    rebalance_dates = set(
        prices.resample("BM").last().index.date
    )

    current_holdings: pd.Series = pd.Series(dtype=float)
    records = []
    prev_equity = 1.0

    for ts in prices.index:
        day = ts.date()
        if day < start or day > end:
            continue

        # Rebalance on month-end
        if day in rebalance_dates:
            # Compute 60d momentum for all symbols
            lookback_ts = ts - pd.Timedelta(days=lookback * 1.5)
            hist = prices.loc[lookback_ts:ts]
            if len(hist) >= lookback:
                mom = (hist.iloc[-1] / hist.iloc[-lookback] - 1).dropna()
                n_top = max(1, int(len(mom) * top_pct))
                new_holdings = mom.nlargest(n_top)
                weight = 1.0 / len(new_holdings)
                new_holdings = new_holdings * 0 + weight  # equal weight
            else:
                new_holdings = current_holdings.copy()

            # Compute turnover
            all_syms = new_holdings.index.union(current_holdings.index)
            old_w = current_holdings.reindex(all_syms).fillna(0.0)
            new_w = new_holdings.reindex(all_syms).fillna(0.0)
            turnover = float((new_w - old_w).abs().sum() / 2)
            from scripts.walkforward.cost_models import cost_from_turnover
            cost = cost_from_turnover(turnover, cost_bps_per_side)
            current_holdings = new_holdings
        else:
            turnover = 0.0
            cost = 0.0

        # Daily gross return
        if current_holdings.empty:
            gross_ret = 0.0
        else:
            prev_closes = prices.loc[prices.index < ts]
            if prev_closes.empty:
                gross_ret = 0.0
            else:
                today_close = prices.loc[ts]
                prev_close = prev_closes.iloc[-1]
                valid = current_holdings.index.intersection(today_close.index).intersection(prev_close.index)
                if valid.empty:
                    gross_ret = 0.0
                else:
                    rets = (today_close[valid] / prev_close[valid].replace(0, np.nan) - 1).dropna()
                    w = current_holdings.reindex(rets.index).fillna(0.0)
                    w = w / w.sum() if w.sum() > 0 else w
                    gross_ret = float((rets * w).sum())

        net_ret = gross_ret - cost
        prev_equity = prev_equity * (1 + net_ret)
        records.append({
            "date": day,
            "gross_ret": round(gross_ret, 8),
            "turnover": round(turnover, 6),
            "cost": round(cost, 8),
            "net_ret": round(net_ret, 8),
            "equity": round(prev_equity, 8),
            "n_holdings": len(current_holdings),
        })

    return pd.DataFrame(records).set_index("date")


# ── Strategy B2: SPY > 200d MA timing ────────────────────────────────────────

def run_spy_ma_timing(
    spy_df: pd.DataFrame,
    ma_window: int = 200,
    cost_bps_per_side: float = 5.0,
) -> pd.DataFrame:
    """Long SPY when close > 200d MA, else cash.

    Returns daily DataFrame with columns:
        date, position, gross_ret, cost, net_ret, equity
    """
    col = "close" if "close" in spy_df.columns else "Close"
    if col not in spy_df.columns:
        logger.warning("SPY DataFrame missing 'close' column")
        return pd.DataFrame()

    prices = spy_df[col].sort_index()
    prices.index = pd.to_datetime(prices.index)
    ma = prices.rolling(ma_window, min_periods=ma_window // 2).mean()

    prev_equity = 1.0
    prev_position = 0  # 0 = cash, 1 = long
    records = []

    for i in range(1, len(prices)):
        ts = prices.index[i]
        day = ts.date()
        new_position = 1 if prices.iloc[i - 1] > ma.iloc[i - 1] else 0

        # Cost on position change
        cost = 0.0
        if new_position != prev_position:
            from scripts.walkforward.cost_models import cost_from_turnover
            cost = cost_from_turnover(1.0, cost_bps_per_side)

        gross_ret = float(prices.iloc[i] / prices.iloc[i - 1] - 1.0) if new_position else 0.0
        net_ret = gross_ret - cost
        prev_equity = prev_equity * (1 + net_ret)
        records.append({
            "date": day,
            "position": new_position,
            "gross_ret": round(gross_ret, 8),
            "cost": round(cost, 8),
            "net_ret": round(net_ret, 8),
            "equity": round(prev_equity, 8),
        })
        prev_position = new_position

    return pd.DataFrame(records).set_index("date")


def _write_manifest(out_dir: Path, args: argparse.Namespace, runtime_s: float) -> None:
    import subprocess
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_sha = "unknown"
    manifest = {
        "script": "diag_naive_baseline.py",
        "git_sha": git_sha,
        "start": str(args.start),
        "end": str(args.end),
        "cost_bps": args.cost_bps,
        "top_pct": args.top_pct,
        "lookback": args.lookback,
        "ma_window": args.ma_window,
        "sacred_holdout_start": SACRED_HOLDOUT_START,
        "runtime_seconds": round(runtime_s, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _write_summary(out_dir: Path, metrics: Dict[str, Dict]) -> None:
    lines = [
        "# Phase A3 — Naive Baseline Comparison Report",
        "",
        f"**Generated:** {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Kill Criterion",
        "",
        "If B1 or B3 Sharpe > best ML WF Sharpe (v186 = +0.106), the ML model is",
        "**destroying** alpha. Go to Phase C (re-architect label, model, or strategy).",
        "",
        "## Reference ML WF Results",
        "",
        "| Model | Avg Sharpe |",
        "|---|---|",
    ]
    for model, sharpe in _ML_REFERENCE.items():
        lines.append(f"| {model} | {sharpe:+.3f} |")
    lines += [
        "",
        "## Baseline Results",
        "",
        "| Strategy | Sharpe | Max DD | Calmar | Total Ret | CAGR |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in metrics.items():
        if m:
            lines.append(
                f"| {name} "
                f"| {m.get('sharpe', 0):+.3f} "
                f"| {m.get('max_drawdown', 0):.1%} "
                f"| {m.get('calmar', 0):+.3f} "
                f"| {m.get('total_return', 0):+.1%} "
                f"| {m.get('cagr', 0):+.1%} |"
            )
    best_ml = max(_ML_REFERENCE.values())
    best_baseline = max((m.get("sharpe", -99) for m in metrics.values() if m), default=-99)
    lines += [
        "",
        "## Verdict",
        "",
        f"- Best ML WF Sharpe: **{best_ml:+.3f}**",
        f"- Best baseline Sharpe: **{best_baseline:+.3f}**",
    ]
    if best_baseline > best_ml:
        lines.append(
            f"\n**KILL CRITERION HIT**: naive baseline ({best_baseline:+.3f}) beats ML ({best_ml:+.3f}). "
            "ML is not adding value. Recommend Phase C."
        )
    else:
        lines.append(
            f"\nBaseline ({best_baseline:+.3f}) is below ML best ({best_ml:+.3f}). "
            "ML appears to add alpha over naive momentum."
        )
    (out_dir / "baseline_summary.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase A3: Naive baseline strategies vs ML walk-forward"
    )
    parser.add_argument("--start", type=date.fromisoformat, default=_DEFAULT_START)
    parser.add_argument("--end", type=date.fromisoformat, default=_DEFAULT_END)
    parser.add_argument("--cost-bps", type=float, default=5.0,
                        help="Transaction cost per side in basis points (default 5)")
    parser.add_argument("--top-pct", type=float, default=0.20,
                        help="Top percentile for momentum portfolio (default 0.20)")
    parser.add_argument("--lookback", type=int, default=60,
                        help="Momentum lookback in trading days (default 60)")
    parser.add_argument("--ma-window", type=int, default=200,
                        help="MA window for SPY timing (default 200)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/diagnostics/naive_baseline"))
    args = parser.parse_args()

    from app.ml.retrain_config import assert_no_sacred_holdout
    assert_no_sacred_holdout(args.end, context="diag_naive_baseline")

    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    logger.info("Phase A3 naive baselines: %s -> %s | cost=%.0fbps | top_pct=%.0f%%",
                args.start, args.end, args.cost_bps, args.top_pct * 100)

    # ── Load SPY bars ──
    from app.data.polygon_s3 import fetch_bulk_daily_bars
    spy_map = fetch_bulk_daily_bars(["SPY"], start_date=args.start, end_date=args.end, workers=1)
    spy_df = spy_map.get("SPY", pd.DataFrame())

    # ── Load universe bars for B1/B3 ──
    from app.data.universe_history import get_russell1000_symbols
    symbols = get_russell1000_symbols()
    if args.max_symbols:
        symbols = symbols[:args.max_symbols]
    logger.info("Loading bars for %d symbols", len(symbols))
    bars_map = fetch_bulk_daily_bars(symbols, start_date=args.start, end_date=args.end, workers=args.workers)
    bars_map["SPY"] = spy_df

    all_metrics = {}

    # ── B1: Momentum ──
    logger.info("Running B1: top-%.0f%% %dd momentum, monthly rebalance", args.top_pct * 100, args.lookback)
    b1 = run_momentum_baseline(
        bars_map, args.start, args.end,
        lookback=args.lookback, top_pct=args.top_pct, cost_bps_per_side=args.cost_bps,
    )
    if not b1.empty:
        b1.to_parquet(out_dir / "b1_equity.parquet")
        all_metrics["B1_momentum"] = _compute_metrics(b1["equity"], b1["net_ret"], b1["turnover"])
        logger.info("B1 Sharpe: %.3f", all_metrics["B1_momentum"].get("sharpe", 0))

    # ── B2: SPY MA timing ──
    logger.info("Running B2: SPY > %dd MA timing", args.ma_window)
    b2 = run_spy_ma_timing(spy_df, ma_window=args.ma_window, cost_bps_per_side=args.cost_bps)
    if not b2.empty:
        b2.to_parquet(out_dir / "b2_equity.parquet")
        all_metrics["B2_spy_ma"] = _compute_metrics(b2["equity"], b2["net_ret"])
        logger.info("B2 Sharpe: %.3f", all_metrics["B2_spy_ma"].get("sharpe", 0))

    # ── B3: Momentum gated by SPY regime ──
    if not b1.empty and not b2.empty:
        logger.info("Running B3: B1 gated by B2 (momentum only in RISK_ON regime)")
        b3 = b1.copy()
        b2_aligned = b2.reindex(b1.index, method="ffill")
        in_regime = b2_aligned["position"] == 1
        b3.loc[~in_regime, "net_ret"] = 0.0
        b3.loc[~in_regime, "gross_ret"] = 0.0
        b3["equity"] = (1 + b3["net_ret"]).cumprod()
        b3.to_parquet(out_dir / "b3_equity.parquet")
        all_metrics["B3_momentum_gated"] = _compute_metrics(b3["equity"], b3["net_ret"])
        logger.info("B3 Sharpe: %.3f", all_metrics["B3_momentum_gated"].get("sharpe", 0))

    # ── Write outputs ──
    if all_metrics:
        pd.DataFrame(all_metrics).T.to_csv(out_dir / "baseline_metrics.csv")
    _write_summary(out_dir, all_metrics)
    runtime_s = time.time() - t_start
    _write_manifest(out_dir, args, runtime_s)

    logger.info("Artifacts written to: %s", out_dir)
    logger.info("Done in %.1fs", runtime_s)

    # Print summary to stdout
    print("\n" + (out_dir / "baseline_summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
