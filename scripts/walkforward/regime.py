"""
regime.py — Daily market regime tagger for walk-forward fold stratification.

Two schemes (selected via app.ml.retrain_config.REGIME_SCHEME):
  coarse3 (default): BULL / BEAR / NEUTRAL — 3 buckets with expanding-quantile
    VIX thresholds (PIT-correct, no look-ahead). Enough obs per bucket for the
    worst-regime-sharpe gate to be meaningful.
  legacy16: VIX_quartile (1-4) x SPY_trend (U/D) x SPY_momentum (P/N), up to
    16 buckets (e.g. "1UP", "2DN"). Original scheme; too-sparse per bucket.

Usage:
    from scripts.walkforward.regime import load_regime_map, label_days
    regime_map = load_regime_map(start_date, end_date)
    labels = label_days(date_list, regime_map)
    n_distinct = len(set(labels))
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── public API ────────────────────────────────────────────────────────────────


def load_regime_map(
    start: date,
    end: date,
    vix_ticker: str = "^VIX",
    spy_ticker: str = "SPY",
) -> Dict[date, str]:
    """Download SPY + VIX history and return {date: regime_label}.

    When REGIME_SCHEME='coarse3' (default):
      Label = 'BULL', 'BEAR', or 'NEUTRAL'
      - BEAR: VIX >= expanding-pctile(REGIME_BEAR_VIX_PCTILE) OR SPY < REGIME_SPY_MA_BEAR-d MA
      - BULL: VIX <= expanding-pctile(REGIME_BULL_VIX_PCTILE) AND SPY > REGIME_SPY_MA_BULL-d MA
      - NEUTRAL: everything else
      VIX thresholds use expanding quantiles (PIT-correct, no look-ahead).
      Days with < REGIME_VIX_WARMUP_DAYS of prior VIX history → NEUTRAL.

    When REGIME_SCHEME='legacy16':
      Label = '<vix_quartile><trend><momentum>' (original scheme, up to 16 buckets).
    """
    from app.ml.retrain_config import REGIME_SCHEME
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np  # noqa: F401  (imported for availability check)
    except ImportError as e:
        logger.warning("yfinance/pandas/numpy not available: %s — returning empty regime map", e)
        return {}

    # Fetch with warmup buffer: need MA lookback + warmup days before start
    from app.ml.retrain_config import REGIME_SPY_MA_BEAR, REGIME_VIX_WARMUP_DAYS
    _buffer_days = max(REGIME_SPY_MA_BEAR, REGIME_VIX_WARMUP_DAYS) + 30
    fetch_start = start - timedelta(days=_buffer_days)
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

    if REGIME_SCHEME == "coarse3":
        return _load_coarse3(spy, vix, common, start, end)
    else:
        return _load_legacy16(spy, vix, common, start, end)


def _load_coarse3(spy, vix, common, start, end) -> Dict[date, str]:
    """Build BULL/BEAR/NEUTRAL regime map using expanding VIX quantiles (PIT-correct)."""
    from app.ml.retrain_config import (
        REGIME_BULL_VIX_PCTILE, REGIME_BEAR_VIX_PCTILE,
        REGIME_SPY_MA_BULL, REGIME_SPY_MA_BEAR, REGIME_VIX_WARMUP_DAYS,
    )
    import numpy as np
    import pandas as pd

    spy_ma_bull = spy.rolling(REGIME_SPY_MA_BULL, min_periods=REGIME_SPY_MA_BULL // 2).mean()
    spy_ma_bear = spy.rolling(REGIME_SPY_MA_BEAR, min_periods=REGIME_SPY_MA_BEAR // 2).mean()
    vix_arr = vix.values

    regime_map: Dict[date, str] = {}
    for i, dt in enumerate(common):
        d = dt.date() if hasattr(dt, "date") else dt
        if d < start or d > end:
            continue
        # Insufficient warmup: NEUTRAL
        if i < REGIME_VIX_WARMUP_DAYS:
            regime_map[d] = "NEUTRAL"
            continue
        # NaN MA: NEUTRAL
        ma_bull_val = spy_ma_bull.iloc[i]
        ma_bear_val = spy_ma_bear.iloc[i]
        if pd.isna(ma_bull_val) or pd.isna(ma_bear_val):
            regime_map[d] = "NEUTRAL"
            continue
        # Expanding-window VIX quantile thresholds (PIT-correct — uses only [:i+1])
        vix_lo = float(np.percentile(vix_arr[:i + 1], REGIME_BULL_VIX_PCTILE))
        vix_hi = float(np.percentile(vix_arr[:i + 1], REGIME_BEAR_VIX_PCTILE))
        spy_val = float(spy.iloc[i])
        vix_val = float(vix_arr[i])
        if vix_val >= vix_hi or spy_val < float(ma_bear_val):
            regime_map[d] = "BEAR"
        elif vix_val <= vix_lo and spy_val > float(ma_bull_val):
            regime_map[d] = "BULL"
        else:
            regime_map[d] = "NEUTRAL"

    logger.info("Regime map (coarse3): %d dates tagged (%s -> %s)", len(regime_map), start, end)
    return regime_map


def _load_legacy16(spy, vix, common, start, end) -> Dict[date, str]:
    """Original VIX-quartile x trend x momentum scheme (up to 16 labels)."""
    import pandas as pd
    spy_ma50 = spy.rolling(50, min_periods=20).mean()
    spy_ret20 = spy.pct_change(20)
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
    logger.info("Regime map (legacy16): %d dates tagged (%s -> %s)", len(regime_map), start, end)
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
    regime_map: Optional[Dict[date, str]] = None,
    obs_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """Compute annualised Sharpe ratio per regime label from a daily equity curve.

    equity_curve is a list of (date, equity) tuples (one per trading day).
    Pass regime_map to use a pre-computed global map so VIX quartile thresholds
    are consistent across folds. If None, loads a per-test-window map (less stable).
    Returns {} on any error (network failure, insufficient data, etc.) so callers
    can safely ignore regime_sharpes when regime data is unavailable.

    Phase-4 FIX-2 (event-sparsity vs data-bug disambiguation): when an `obs_counts`
    dict is supplied, it is populated with the RAW per-regime observation count
    (BEFORE the REGIME_MIN_OBS filter). This lets the caller distinguish the two
    causes of an empty result:
      - EVENT-SPARSITY (structural, "not a bug"): obs_counts is non-empty (returns
        WERE mapped to regime labels) but every bucket fell below REGIME_MIN_OBS,
        so all were dropped → result {} while obs_counts has entries. This is
        PEAD's case (event-driven; flat most days → < REGIME_MIN_OBS same-regime
        trading days per bucket).
      - DATA-BUG (broken): obs_counts is empty — no regime map, no labelled days,
        a degenerate curve, or an exception. Genuinely no regime data to gate on.
    obs_counts is mutated in place (cleared first) so a single shared dict can be
    reused across calls if desired.
    """
    if obs_counts is not None:
        obs_counts.clear()
    try:
        import numpy as np
        from app.ml.retrain_config import REGIME_MIN_OBS as _REGIME_MIN_OBS

        if len(equity_curve) < 2:
            return {}

        if regime_map is None:
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

        # FIX-2: record raw per-regime obs counts BEFORE the REGIME_MIN_OBS filter
        # so the caller can tell "insufficient obs" (sparsity) from "no data" (bug).
        if obs_counts is not None:
            for label, rets in regime_daily_rets.items():
                obs_counts[label] = len(rets)

        result: Dict[str, float] = {}
        for label, rets in regime_daily_rets.items():
            arr = np.array(rets)
            if len(arr) < _REGIME_MIN_OBS:
                # Phase 2 (C10-3): drop regimes with < REGIME_MIN_OBS obs —
                # mean*sqrt(252) on a handful of returns is not a Sharpe and would
                # corrupt worst_regime_sharpe with phantom extremes.
                continue
            std = float(np.std(arr, ddof=1))
            result[label] = float(np.mean(arr)) / std * np.sqrt(252) if std > 0 else 0.0
        return result
    except Exception:
        # DATA-BUG path: leave obs_counts empty (already cleared above).
        return {}
