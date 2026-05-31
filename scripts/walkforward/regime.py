"""
regime.py — Daily market regime tagger for walk-forward fold stratification.

Regime label = VIX_quartile (1-4) x SPY_trend (U/D) x SPY_momentum (P/N)
Results in up to 8 buckets per day (e.g. "1UP", "2DN", "3UP" ...).

Usage:
    from scripts.walkforward.regime import load_regime_map, label_days
    regime_map = load_regime_map(start_date, end_date)
    labels = label_days(date_list, regime_map)
    n_distinct = len(set(labels))
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, List

logger = logging.getLogger(__name__)

# ── public API ────────────────────────────────────────────────────────────────


def load_regime_map(
    start: date,
    end: date,
    vix_ticker: str = "^VIX",
    spy_ticker: str = "SPY",
) -> Dict[date, str]:
    """Download SPY + VIX history and return {date: regime_label}.

    Label format: "<vix_quartile><trend><momentum>"
      vix_quartile: 1 (lowest) .. 4 (highest), based on rolling quartile of VIX over the window
      trend:  U = SPY above 50d MA, D = SPY below 50d MA
      momentum: P = SPY 20d return > 0, N = SPY 20d return <= 0

    Examples: "1UP", "4DN", "2DP"
    """
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError as e:
        logger.warning("yfinance/pandas/numpy not available: %s — returning empty regime map", e)
        return {}

    fetch_start = start - timedelta(days=80)  # need 50d MA lookback
    tickers = yf.download(
        [spy_ticker, vix_ticker],
        start=fetch_start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
    )

    # Handle both single and multi-ticker response shapes
    if isinstance(tickers.columns, pd.MultiIndex):
        spy_close = tickers["Close"][spy_ticker].dropna()
        vix_close = tickers["Close"][vix_ticker].dropna()
    else:
        # Single ticker fallback (shouldn't happen with list input)
        spy_close = tickers["Close"].dropna()
        vix_close = spy_close.copy() * 0 + 20  # dummy VIX

    # Align on common index
    common = spy_close.index.intersection(vix_close.index)
    spy = spy_close.reindex(common)
    vix = vix_close.reindex(common)

    # Indicators
    spy_ma50 = spy.rolling(50, min_periods=20).mean()
    spy_ret20 = spy.pct_change(20)

    # VIX quartile labels (1=lowest, 4=highest) over the full window
    vix_q = pd.qcut(vix, q=4, labels=["1", "2", "3", "4"], duplicates="drop")

    regime_map: Dict[date, str] = {}
    for dt in common:
        d = dt.date() if hasattr(dt, "date") else dt
        if d < start or d > end:
            continue
        try:
            vq = str(vix_q.loc[dt])
            trend = "U" if spy.loc[dt] > spy_ma50.loc[dt] else "D"
            mom = "P" if spy_ret20.loc[dt] > 0 else "N"
            regime_map[d] = vq + trend + mom
        except (KeyError, TypeError, ValueError):
            pass

    logger.info("Regime map: %d dates tagged (%s → %s)", len(regime_map), start, end)
    return regime_map


def label_days(days: List[date], regime_map: Dict[date, str]) -> List[str]:
    """Return regime label for each day in `days`. Unknown dates get 'UNK'."""
    return [regime_map.get(d, "UNK") for d in days]


def regime_diversity(days: List[date], regime_map: Dict[date, str]) -> int:
    """Return the number of distinct regime labels in `days`."""
    labels = set(label_days(days, regime_map)) - {"UNK"}
    return len(labels)


def compute_regime_sharpes(
    equity_curve: list,
    te_start,
    te_end,
) -> Dict[str, float]:
    """Compute annualised Sharpe ratio per regime label from a daily equity curve.

    equity_curve is a list of (date, equity) tuples (one per trading day).
    Returns {} on any error (network failure, insufficient data, etc.) so callers
    can safely ignore regime_sharpes when regime data is unavailable.
    """
    try:
        import numpy as np

        if len(equity_curve) < 2:
            return {}

        start = te_start.date() if hasattr(te_start, "date") else te_start
        end = te_end.date() if hasattr(te_end, "date") else te_end
        regime_map = load_regime_map(start, end)
        if not regime_map:
            return {}

        regime_daily_rets: Dict[str, list] = {}
        for i in range(1, len(equity_curve)):
            d, eq = equity_curve[i]
            _, eq_prev = equity_curve[i - 1]
            if eq_prev <= 0:
                continue
            ret = (eq - eq_prev) / eq_prev
            key = d.date() if hasattr(d, "date") else d
            label = regime_map.get(key)
            if label is None:
                continue
            regime_daily_rets.setdefault(label, []).append(ret)

        result: Dict[str, float] = {}
        for label, rets in regime_daily_rets.items():
            arr = np.array(rets)
            if len(arr) < 2:
                result[label] = float(np.mean(arr)) * np.sqrt(252)
            else:
                std = float(np.std(arr, ddof=1))
                result[label] = float(np.mean(arr)) / std * np.sqrt(252) if std > 0 else 0.0
        return result
    except Exception:
        return {}
