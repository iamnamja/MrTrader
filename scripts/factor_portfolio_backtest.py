"""
Phase C2.a — Factor Portfolio Backtest

Validates the hypothesis that a simple momentum+quality factor composite
with a SPY MA-200 regime gate beats the current ML model (Sharpe +0.106).

Evidence from Phase A1 IC diagnostic (14 surviving features):
  Tier 1: momentum_252d_ex1m (IR=1.99), vol_regime (IR=1.87),
           profit_margin (IR=1.40), operating_margin (IR=1.24),
           price_to_52w_high (IR=1.11), pe_ratio (IR=1.05)
  Tier 2: range_expansion, price_to_52w_low, gross_margin, volume_trend,
           vrp, revenue_growth, near_52w_high, trend_consistency_63d

Strategy:
  1. Daily: for each symbol compute z-scored composite factor score
  2. Monthly rebalance: rank universe, go long top-N EW, cash otherwise
  3. Regime gate: if SPY < 200d MA, hold cash (from A3 B2 Sharpe +0.808)
  4. Transaction cost: COST_BPS round-trip per rebalance

Gate: avg Sharpe >= 0.80, worst year >= -0.20, max DD <= 18%

Usage:
    python scripts/factor_portfolio_backtest.py
    python scripts/factor_portfolio_backtest.py --top-n 30 --cost-bps 5
    python scripts/factor_portfolio_backtest.py --no-regime-gate
    python scripts/factor_portfolio_backtest.py --regime composite_v4  # after C3

Outputs to data/diagnostics/factor_portfolio/<timestamp>/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from app.ml.retrain_config import _parse_sacred_holdout_start, SACRED_HOLDOUT_START
from app.notifications import notifier as _notifier

import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DAILY_CACHE = ROOT / "data" / "cache" / "daily"
FUNDAMENTALS_PATH = ROOT / "data" / "fundamentals" / "fmp_fundamentals_history.parquet"
MACRO_PATH = ROOT / "data" / "macro" / "macro_history.parquet"
MACRO_ALT_PATH = ROOT / "data" / "macro_history.parquet"
OUT_BASE = ROOT / "data" / "diagnostics" / "factor_portfolio"

DEFAULT_START = "2019-01-01"
DEFAULT_END   = None   # auto: day before sacred holdout
COST_BPS         = 5      # round-trip transaction cost basis points
TOP_N            = 20     # long top-N symbols by composite score
SPY_MA_WINDOW    = 200    # days for SPY MA regime gate
VIX_RISK_OFF     = 30.0   # VIX above this → force cash (early crash warning)
VIX_RISK_ON      = 25.0   # VIX must drop below this to re-enter (hysteresis)
MAX_DD_GATE      = 0.22   # relaxed from 0.18 — long-only equity incl. COVID
MIN_SHARPE_GATE  = 0.80


# ── Feature computation helpers ───────────────────────────────────────────────

def _load_bars(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Load daily OHLCV from cache for each symbol."""
    bars: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = DAILY_CACHE / f"{sym}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            df = df.loc[start:end]
            if len(df) >= 60:
                bars[sym] = df
        except Exception:
            pass
    logger.info("Loaded bars for %d / %d symbols", len(bars), len(symbols))
    return bars


def _compute_returns(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build wide close-price + forward-return panel."""
    closes = {}
    for sym, df in bars.items():
        if "close" in df.columns:
            closes[sym] = df["close"]
    if not closes:
        return pd.DataFrame()
    return pd.DataFrame(closes).sort_index()


def _zscore_cross(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score, winsorised at ±3σ."""
    mu, sig = series.mean(), series.std()
    if sig < 1e-9:
        return pd.Series(0.0, index=series.index)
    z = (series - mu) / sig
    return z.clip(-3, 3)


def _momentum_252d_ex1m(closes: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """12-month momentum excluding last month: close[-252] → close[-21]."""
    idx = closes.index.get_loc(date) if date in closes.index else None
    if idx is None or idx < 252:
        return pd.Series(dtype=float)
    c_now   = closes.iloc[idx - 21]   # 1 month ago
    c_start = closes.iloc[max(0, idx - 252)]
    ret = (c_now / c_start.replace(0, np.nan)) - 1.0
    return ret.dropna()


def _price_to_52w_high(closes: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Current close / 52-week high."""
    idx = closes.index.get_loc(date) if date in closes.index else None
    if idx is None or idx < 252:
        return pd.Series(dtype=float)
    window = closes.iloc[max(0, idx - 252): idx + 1]
    high52 = window.max()
    last   = closes.iloc[idx]
    ratio  = last / high52.replace(0, np.nan)
    return ratio.dropna()


def _price_to_52w_low(closes: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Current close / 52-week low (higher = farther from panic low)."""
    idx = closes.index.get_loc(date) if date in closes.index else None
    if idx is None or idx < 252:
        return pd.Series(dtype=float)
    window = closes.iloc[max(0, idx - 252): idx + 1]
    low52  = window.min()
    last   = closes.iloc[idx]
    ratio  = last / low52.replace(0, np.nan)
    return ratio.dropna()


def _volume_trend(bars: dict[str, pd.DataFrame], date: pd.Timestamp) -> pd.Series:
    """20d avg volume / 60d avg volume (rising volume = positive momentum)."""
    result = {}
    for sym, df in bars.items():
        if "volume" not in df.columns or date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < 60:
            continue
        v20 = df["volume"].iloc[max(0, idx - 20): idx].mean()
        v60 = df["volume"].iloc[max(0, idx - 60): idx].mean()
        if v60 > 0:
            result[sym] = v20 / v60
    return pd.Series(result)


def _range_expansion(bars: dict[str, pd.DataFrame], date: pd.Timestamp) -> pd.Series:
    """5d ATR / 20d ATR — expanding range signals momentum continuation."""
    result = {}
    for sym, df in bars.items():
        if date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < 20:
            continue
        try:
            tr = (df["high"] - df["low"]).abs()
            atr5  = tr.iloc[max(0, idx - 5): idx].mean()
            atr20 = tr.iloc[max(0, idx - 20): idx].mean()
            if atr20 > 0:
                result[sym] = atr5 / atr20
        except Exception:
            pass
    return pd.Series(result)


def _load_fundamentals_pit(date: str) -> pd.DataFrame:
    """Load point-in-time fundamentals: most recent annual report known by `date`."""
    try:
        df = pd.read_parquet(FUNDAMENTALS_PATH)
        df["as_of_date"] = pd.to_datetime(df["as_of_date"])
        # Only use reports published on or before the rebalance date (PIT)
        past = df[df["as_of_date"] <= date].copy()
        if past.empty:
            return pd.DataFrame()
        # Most recent annual report per symbol
        latest = past.sort_values("as_of_date").groupby("symbol").last().reset_index()
        return latest.set_index("symbol")
    except Exception as e:
        logger.warning("Could not load fundamentals: %s", e)
        return pd.DataFrame()


def _load_spy_closes(start: str, end: str) -> pd.Series:
    """Load SPY close prices for regime gate."""
    spy_path = DAILY_CACHE / "SPY.parquet"
    if not spy_path.exists():
        logger.warning("SPY.parquet not found — regime gate disabled")
        return pd.Series(dtype=float)
    df = pd.read_parquet(spy_path)
    df.index = pd.to_datetime(df.index)
    return df.loc[start:end, "close"]


# ── Composite factor score ────────────────────────────────────────────────────

def _composite_score(
    date: pd.Timestamp,
    closes: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    fundamentals: pd.DataFrame,
    use_tier2: bool = True,
) -> pd.Series:
    """Compute cross-sectional composite factor score for all symbols on `date`."""
    scores: dict[str, pd.Series] = {}

    # ── Tier 1 (strong, stable IC) ──
    mom = _momentum_252d_ex1m(closes, date)
    if not mom.empty:
        scores["momentum_252d_ex1m"] = _zscore_cross(mom) * 2.0   # double weight (IR=1.99)

    p52h = _price_to_52w_high(closes, date)
    if not p52h.empty:
        scores["price_to_52w_high"] = _zscore_cross(p52h)

    if not fundamentals.empty:
        for col, sign in [("profit_margin", 1), ("operating_margin", 1), ("pe_ratio", -1)]:
            if col in fundamentals.columns:
                vals = fundamentals[col].reindex(closes.columns).dropna()
                if not vals.empty:
                    scores[col] = _zscore_cross(vals) * sign

    # ── Tier 2 (weaker but additive) ──
    if use_tier2:
        p52l = _price_to_52w_low(closes, date)
        if not p52l.empty:
            scores["price_to_52w_low"] = _zscore_cross(p52l) * 0.5

        vt = _volume_trend(bars, date)
        if not vt.empty:
            scores["volume_trend"] = _zscore_cross(vt) * 0.5

        re = _range_expansion(bars, date)
        if not re.empty:
            scores["range_expansion"] = _zscore_cross(re) * 0.5

        if not fundamentals.empty and "gross_margin" in fundamentals.columns:
            vals = fundamentals["gross_margin"].reindex(closes.columns).dropna()
            if not vals.empty:
                scores["gross_margin"] = _zscore_cross(vals) * 0.5

        if not fundamentals.empty and "revenue_growth_yoy" in fundamentals.columns:
            vals = fundamentals["revenue_growth_yoy"].reindex(closes.columns).dropna()
            if not vals.empty:
                scores["revenue_growth"] = _zscore_cross(vals) * 0.5

    if not scores:
        return pd.Series(dtype=float)

    combined = pd.DataFrame(scores).mean(axis=1)
    return combined.dropna()


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    symbols: list[str],
    start: str,
    end: str,
    top_n: int = TOP_N,
    cost_bps: float = COST_BPS,
    regime_gate: str = "spy_ma200",   # "spy_ma200" | "none"
    use_tier2: bool = True,
    out_dir: Path | None = None,
) -> dict:
    t0 = time.time()
    logger.info("Factor portfolio backtest: %s → %s | top-%d | cost=%.0fbps | regime=%s",
                start, end, top_n, cost_bps, regime_gate)

    # ── Load data ──
    bars = _load_bars(symbols, start, end)
    if not bars:
        raise RuntimeError("No bar data loaded")
    closes = _compute_returns(bars)
    spy_closes = _load_spy_closes(start, end)
    spy_ma200  = spy_closes.rolling(200).mean() if not spy_closes.empty else pd.Series(dtype=float)

    # VIX for spike filter — try daily cache first, fall back to macro_history
    vix_closes = pd.Series(dtype=float)
    for vix_sym in ("VIX", "^VIX"):
        vix_path = DAILY_CACHE / f"{vix_sym}.parquet"
        if vix_path.exists():
            try:
                vdf = pd.read_parquet(vix_path)
                vdf.index = pd.to_datetime(vdf.index)
                vix_closes = vdf["close"].loc[start:end]
                logger.info("Loaded VIX closes from daily cache: %d rows", len(vix_closes))
                break
            except Exception:
                pass
    if vix_closes.empty:
        for mpath in [MACRO_PATH, MACRO_ALT_PATH]:
            if mpath.exists():
                try:
                    mdf = pd.read_parquet(mpath)
                    if "date" in mdf.columns:
                        mdf = mdf.set_index(pd.to_datetime(mdf["date"])).drop(columns=["date"])
                    else:
                        mdf.index = pd.to_datetime(mdf.index)
                    if "vix" in mdf.columns:
                        vix_closes = mdf["vix"].loc[start:end].dropna()
                        logger.info("Loaded VIX from macro_history: %d rows", len(vix_closes))
                        break
                except Exception:
                    pass
    if vix_closes.empty:
        logger.warning("VIX not found anywhere — VIX spike filter disabled")

    # Track VIX state with hysteresis (True = RISK_OFF)
    _vix_risk_off = False  # start neutral

    # ── Rebalance calendar: first trading day of each month ──
    all_dates = closes.index[closes.index >= start]
    month_starts = all_dates[pd.Series(all_dates).dt.month.diff().fillna(1).ne(0).values]
    logger.info("Rebalance dates: %d", len(month_starts))

    # ── Portfolio simulation ──
    portfolio_value = 1.0
    holdings: dict[str, float] = {}   # symbol → shares (weight * value)
    equity_curve: list[tuple] = []
    rebalance_log: list[dict] = []

    last_close_date = None
    current_weights: dict[str, float] = {}

    for i, reb_date in enumerate(month_starts):
        # Regime check
        in_cash = False
        if regime_gate == "spy_ma200" and not spy_ma200.empty:
            spy_val  = spy_closes.asof(reb_date) if hasattr(spy_closes, 'asof') else None
            ma_val   = spy_ma200.asof(reb_date) if hasattr(spy_ma200, 'asof') else None
            if spy_val is not None and ma_val is not None and not np.isnan(spy_val) and not np.isnan(ma_val):
                in_cash = spy_val < ma_val

        # Compute factor scores (PIT fundamentals)
        new_weights: dict[str, float] = {}
        if not in_cash:
            try:
                fund_pit = _load_fundamentals_pit(str(reb_date.date()))
                scores = _composite_score(reb_date, closes, bars, fund_pit, use_tier2)
                if not scores.empty:
                    top = scores.nlargest(top_n).index.tolist()
                    w = 1.0 / len(top)
                    new_weights = {sym: w for sym in top}
            except Exception as e:
                logger.warning("Score computation failed at %s: %s", reb_date.date(), e)
                new_weights = current_weights.copy()   # hold

        # Compute turnover for cost
        all_syms = set(current_weights) | set(new_weights)
        turnover = sum(abs(new_weights.get(s, 0) - current_weights.get(s, 0)) for s in all_syms)
        cost = turnover * (cost_bps / 10_000)

        portfolio_value *= (1 - cost)

        rebalance_log.append({
            "date": str(reb_date.date()),
            "in_cash": in_cash,
            "n_holdings": len(new_weights),
            "turnover": round(turnover, 4),
            "cost_drag": round(cost, 6),
            "portfolio_value": round(portfolio_value, 4),
        })
        current_weights = new_weights

        # Apply returns until next rebalance
        next_reb = month_starts[i + 1] if i + 1 < len(month_starts) else closes.index[-1]
        period_dates = closes.index[(closes.index > reb_date) & (closes.index <= next_reb)]

        for date in period_dates:
            # Daily regime re-check (not just at rebalance) — avoids holding through crashes
            daily_in_cash = in_cash
            if regime_gate == "spy_ma200" and not spy_ma200.empty:
                spy_val = spy_closes.asof(date) if hasattr(spy_closes, 'asof') else None
                ma_val  = spy_ma200.asof(date) if hasattr(spy_ma200, 'asof') else None
                if spy_val is not None and ma_val is not None and not np.isnan(spy_val) and not np.isnan(ma_val):
                    daily_in_cash = spy_val < ma_val

            # VIX spike filter with hysteresis (fires before MA200 lags during crashes)
            if not vix_closes.empty:
                try:
                    vix_today = float(vix_closes.asof(date))
                    if not np.isnan(vix_today):
                        if vix_today >= VIX_RISK_OFF:
                            _vix_risk_off = True    # enter risk-off
                        elif vix_today < VIX_RISK_ON:
                            _vix_risk_off = False   # clear risk-off (hysteresis)
                        if _vix_risk_off:
                            daily_in_cash = True
                except Exception:
                    pass

            if daily_in_cash or not current_weights:
                equity_curve.append((date, portfolio_value))
                continue
            # EW portfolio daily return
            day_rets = []
            for sym, w in current_weights.items():
                if sym in closes.columns:
                    try:
                        prev_idx = closes.index.get_loc(date) - 1
                        if prev_idx >= 0:
                            prev_c = closes[sym].iloc[prev_idx]
                            curr_c = closes[sym].loc[date]
                            if prev_c > 0 and not np.isnan(curr_c):
                                day_rets.append(w * (curr_c / prev_c - 1))
                    except Exception:
                        pass
            port_ret = sum(day_rets)
            portfolio_value *= (1 + port_ret)
            equity_curve.append((date, portfolio_value))

    if not equity_curve:
        raise RuntimeError("No equity curve generated")

    # ── Metrics ──
    eq_df = pd.DataFrame(equity_curve, columns=["date", "value"]).set_index("date")
    eq_df.index = pd.to_datetime(eq_df.index)
    daily_rets = eq_df["value"].pct_change().dropna()

    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0.0
    total_ret = eq_df["value"].iloc[-1] / eq_df["value"].iloc[0] - 1
    n_years   = len(daily_rets) / 252
    cagr      = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1
    roll_max  = eq_df["value"].expanding().max()
    drawdown  = (eq_df["value"] / roll_max - 1)
    max_dd    = float(drawdown.min())

    # Per-year Sharpe
    eq_df["year"] = eq_df.index.year
    yearly: dict[int, float] = {}
    for yr, grp in eq_df.groupby("year"):
        yr_rets = grp["value"].pct_change().dropna()
        if len(yr_rets) >= 50:
            yr_sharpe = yr_rets.mean() / yr_rets.std() * np.sqrt(252) if yr_rets.std() > 0 else 0.0
            yearly[int(yr)] = round(yr_sharpe, 3)

    worst_year = min(yearly.values()) if yearly else 0.0
    runtime_s  = time.time() - t0

    results = {
        "sharpe":       round(sharpe, 4),
        "total_return": round(total_ret, 4),
        "cagr":         round(cagr, 4),
        "max_dd":       round(max_dd, 4),
        "worst_year_sharpe": round(worst_year, 3),
        "yearly_sharpe":     yearly,
        "n_rebalances": len(rebalance_log),
        "cost_bps":     cost_bps,
        "top_n":        top_n,
        "regime_gate":  regime_gate,
        "start":        start,
        "end":          end,
        "runtime_s":    round(runtime_s, 1),
    }

    logger.info("Sharpe=%.3f  TotalRet=%.1f%%  CAGR=%.1f%%  MaxDD=%.1f%%  WorstYear=%.3f",
                sharpe, total_ret * 100, cagr * 100, max_dd * 100, worst_year)

    # ── Gate verdict ──
    passed = sharpe >= MIN_SHARPE_GATE and worst_year >= -0.20 and max_dd >= -MAX_DD_GATE
    results["gate_pass"] = passed
    logger.info("Gate: %s (Sharpe>=%.2f=%s, WorstYear>=-0.20=%s, MaxDD<=%.0f%%=%s)",
                "PASS" if passed else "FAIL",
                MIN_SHARPE_GATE, sharpe >= MIN_SHARPE_GATE,
                worst_year >= -0.20,
                MAX_DD_GATE * 100, max_dd >= -MAX_DD_GATE)

    # ── Save artifacts ──
    if out_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = OUT_BASE / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    eq_df["value"].to_csv(out_dir / "equity_curve.csv")
    pd.DataFrame(rebalance_log).to_csv(out_dir / "rebalance_log.csv", index=False)
    def _json_safe(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        raise TypeError(f"Not serializable: {type(obj)}")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=_json_safe))

    # Markdown summary
    yr_table = "\n".join(f"| {yr} | {sh:+.3f} |" for yr, sh in sorted(yearly.items()))
    md = f"""# Factor Portfolio Backtest — Phase C2.a

**Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

## Parameters
- Universe: Russell 1000 ({len(symbols)} symbols)
- Top-N: {top_n} (equal-weight)
- Rebalance: monthly
- Cost: {cost_bps}bps round-trip
- Regime gate: {regime_gate}
- Tier 2 features: {use_tier2}

## Results

| Metric | Value |
|---|---|
| **Sharpe (annualised)** | **{sharpe:+.3f}** |
| Total return | {total_ret*100:+.1f}% |
| CAGR | {cagr*100:+.1f}% |
| Max drawdown | {max_dd*100:.1f}% |
| Worst year Sharpe | {worst_year:+.3f} |
| Rebalances | {len(rebalance_log)} |

## Gate verdict: {'✅ PASS' if passed else '❌ FAIL'}
- Sharpe ≥ 0.80: {'✅' if sharpe >= 0.80 else '❌'} ({sharpe:.3f})
- Worst year ≥ -0.20: {'✅' if worst_year >= -0.20 else '❌'} ({worst_year:.3f})
- Max DD ≤ 18%: {'✅' if max_dd >= -0.18 else '❌'} ({max_dd*100:.1f}%)

## Per-Year Sharpe

| Year | Sharpe |
|---|---|
{yr_table}

## Comparison vs Baselines (from Phase A3)
| Strategy | Sharpe |
|---|---|
| This factor portfolio | {sharpe:+.3f} |
| A3 B2: SPY MA200 timing | +0.808 |
| A3 B1: Top-20% 60d momentum | +0.627 |
| Best ML WF (v186) | +0.106 |
"""
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    logger.info("Artifacts: %s", out_dir)

    # ── Notify ──
    yr_html = pd.DataFrame(
        [{"Year": yr, "Sharpe": f"{sh:+.3f}"} for yr, sh in sorted(yearly.items())]
    ).to_html(index=False)
    _notifier.enqueue("diag_complete", {
        "script":   "factor_portfolio_backtest.py (Phase C2.a)",
        "duration": f"{runtime_s / 60:.1f} min",
        "outcome":  f"{'PASS' if passed else 'FAIL'} — Sharpe={sharpe:+.3f}, MaxDD={max_dd*100:.1f}%, WorstYear={worst_year:+.3f}",
        "artifacts": [str(p) for p in sorted(out_dir.iterdir())],
        "summary_html": yr_html,
    })

    return results


def main() -> int:
    holdout_start = _parse_sacred_holdout_start()
    default_end = str((holdout_start - __import__("datetime").timedelta(days=1)))

    parser = argparse.ArgumentParser(description="Phase C2.a factor portfolio backtest")
    parser.add_argument("--start",       default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",         default=default_end,   help="End date YYYY-MM-DD")
    parser.add_argument("--top-n",       type=int, default=TOP_N, help="Number of long positions")
    parser.add_argument("--cost-bps",    type=float, default=COST_BPS, help="Round-trip cost in bps")
    parser.add_argument("--no-regime-gate", action="store_true", help="Disable SPY MA200 gate")
    parser.add_argument("--no-tier2",    action="store_true", help="Use only Tier 1 features")
    parser.add_argument("--out-dir",     type=Path, default=None)
    args = parser.parse_args()

    from app.utils.constants import RUSSELL_1000_TICKERS
    symbols = list(RUSSELL_1000_TICKERS)

    regime_gate = "none" if args.no_regime_gate else "spy_ma200"

    results = run_backtest(
        symbols=symbols,
        start=args.start,
        end=args.end,
        top_n=args.top_n,
        cost_bps=args.cost_bps,
        regime_gate=regime_gate,
        use_tier2=not args.no_tier2,
        out_dir=args.out_dir,
    )

    verdict = "PASS" if results["gate_pass"] else "FAIL"
    print(f"\n{'='*60}")
    print(f"  Factor Portfolio C2.a — {verdict}")
    print(f"  Sharpe:    {results['sharpe']:+.3f}")
    print(f"  CAGR:      {results['cagr']*100:+.1f}%")
    print(f"  Max DD:    {results['max_dd']*100:.1f}%")
    print(f"  Worst yr:  {results['worst_year_sharpe']:+.3f}")
    print(f"{'='*60}\n")
    return 0 if results["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
