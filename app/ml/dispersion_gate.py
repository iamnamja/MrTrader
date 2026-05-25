"""
Phase 89 v2: Cross-sectional return dispersion regime gate.

Measures the ratio of current cross-sectional MAD of 5-day returns to its
6-month trailing median baseline. High DRR = factor decoherence / rotation.

Key advantage over rolling realized IC gate (Phase 89 v1, rejected):
- Concurrent signal (~5d lag) vs IC gate's ~80d structural lag
- Correctly identifies correlated sell-offs (low dispersion = low DRR = gate OPEN)
  vs factor rotations (high dispersion = high DRR = gate throttles)
- The Oct 2022 bear bottom has LOW dispersion (everything sold off together)
  so the gate stays open during the recovery — exactly what the IC gate failed to do.

Parameters (2 tunable):
    k (int): Return window in trading days. Default 5.
    L (int): Baseline lookback in trading days. Default 126 (~6 months).

Throttle:
    DRR <= 1.5 → 1.0 (full exposure)
    DRR >= 2.5 → 0.0 (flat)
    Linear in between.

Combined with SPY+VIX gate multiplicatively, floored at 0.10 to keep system alive
during sustained high-dispersion regimes.

Design from Opus 4.7 analysis (2026-05-25).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def compute_dispersion_ratio(
    prices: pd.DataFrame,
    *,
    k: int = 5,
    L: int = 126,
    min_symbols: int = 50,
) -> pd.Series:
    """Compute the cross-sectional Dispersion Ratio (DRR) series.

    Args:
        prices: DataFrame indexed by trading date, columns = symbols (adj close).
            Must be PIT — pass only data available as of each date.
        k: Return window in trading days.
        L: Baseline lookback for median normalization.
        min_symbols: Minimum symbols with valid returns to compute dispersion.

    Returns:
        Series of DRR(t), indexed by date. Values of 1.0 = normal, >1.5 = elevated,
        >2.5 = severe rotation. NaN during burn-in.
    """
    # k-day log returns per symbol
    log_rets = np.log(prices / prices.shift(k))

    def _mad(row: pd.Series) -> float:
        x = row.dropna().values
        if len(x) < min_symbols:
            return np.nan
        med = np.median(x)
        return float(np.median(np.abs(x - med)))

    disp = log_rets.apply(_mad, axis=1)

    # Trailing 6-month median of dispersion (shift 1 to exclude today)
    baseline = (
        disp.shift(1)
        .rolling(L, min_periods=L // 2)
        .median()
    )

    # Floor baseline at its own 10th percentile to prevent false positives
    # in low-volatility regimes where baseline collapses
    baseline_p10 = baseline.quantile(0.10) if baseline.notna().sum() > 20 else None
    if baseline_p10 is not None and baseline_p10 > 0:
        baseline = baseline.clip(lower=baseline_p10)

    drr = disp / baseline
    drr.name = "DRR"
    return drr


def dispersion_multiplier(drr: float, lo: float = 1.5, hi: float = 2.5) -> float:
    """Convert DRR to a gross exposure multiplier [0, 1].

    Continuous linear throttle between lo and hi.
    """
    if not np.isfinite(drr):
        return 1.0
    if drr <= lo:
        return 1.0
    if drr >= hi:
        return 0.0
    return 1.0 - (drr - lo) / (hi - lo)


class DispersionGate:
    """Callable gate for use in the rebalance loop.

    Usage:
        gate = DispersionGate(prices_df)
        mult = gate(rebalance_date)  # float [0.0, 1.0]

    The multiplier is combined multiplicatively with the SPY+VIX gate, floored
    at 0.10 so the system stays alive during sustained high-dispersion regimes.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        *,
        k: int = 5,
        L: int = 126,
        min_symbols: int = 50,
        lo: float = 1.5,
        hi: float = 2.5,
        floor: float = 0.10,
    ) -> None:
        self._drr = compute_dispersion_ratio(prices, k=k, L=L, min_symbols=min_symbols)
        self._lo = lo
        self._hi = hi
        self._floor = floor

    def __call__(self, day: date) -> float:
        """Return dispersion multiplier for the given date (PIT-safe)."""
        ts = pd.Timestamp(day)
        history = self._drr[self._drr.index < ts]
        if history.empty or history.isna().all():
            return 1.0
        drr_val = float(history.iloc[-1])
        return dispersion_multiplier(drr_val, lo=self._lo, hi=self._hi)

    @property
    def drr_series(self) -> pd.Series:
        return self._drr


def make_combined_dispersion_regime_fn(
    fold_symbols_data: dict,
    *,
    spy_vix_fn=None,
    k: int = 5,
    L: int = 126,
    floor: float = 0.10,
):
    """
    Return a combined regime multiplier = max(floor, SPY/VIX gate × dispersion gate).

    Args:
        fold_symbols_data: dict of {symbol: DataFrame} from fold data.
        spy_vix_fn: Existing SPY+VIX callable(day) -> float, or None.
        k: Dispersion return window (trading days).
        L: Baseline lookback (trading days).
        floor: Minimum combined multiplier (keeps system alive during sustained rotation).
    """
    # Build price DataFrame from fold symbols
    price_dict = {}
    for sym, df in fold_symbols_data.items():
        if sym in ("SPY", "^VIX", "VIX") or df is None or "close" not in df.columns:
            continue
        price_dict[sym] = df["close"]

    if len(price_dict) < 50:
        # Not enough symbols — fall back to SPY+VIX only
        return spy_vix_fn

    prices = pd.DataFrame(price_dict)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    try:
        disp_gate = DispersionGate(prices, k=k, L=L)
    except Exception:
        return spy_vix_fn

    def _combined_fn(day: date) -> float:
        spy_mult = spy_vix_fn(day) if spy_vix_fn is not None else 1.0
        disp_mult = disp_gate(day)
        return max(floor, spy_mult * disp_mult)

    return _combined_fn
