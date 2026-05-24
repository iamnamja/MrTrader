"""Tests for Phase RA: app/strategy/portfolio_construction.py

13 unit tests covering liquidity filter, sector cap, hysteresis target
portfolio, and equal-weight sizing.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from app.strategy.portfolio_construction import (
    RebalanceDelta,
    apply_sector_cap,
    compute_equal_weights,
    compute_target_portfolio,
    liquidity_filter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(closes, volumes, start_date=date(2024, 1, 2)):
    """Build a minimal bars DataFrame from lists of closes/volumes."""
    dates = pd.date_range(start=start_date, periods=len(closes), freq="B")
    return pd.DataFrame({"close": closes, "volume": volumes}, index=dates)


def _daily_bars(n_days, close=50.0, volume=1_000_000, start_date=date(2024, 1, 2)):
    return _make_bars([close] * n_days, [volume] * n_days, start_date)


# ---------------------------------------------------------------------------
# liquidity_filter
# ---------------------------------------------------------------------------

class TestLiquidityFilter:
    def test_basic_two_pass_one_fail(self):
        # AAPL: $50 * 500k = $25M/day ✅  JUNK: $1 * 1k = $1k/day ❌
        as_of = date(2024, 4, 1)
        bars = {
            "AAPL": _daily_bars(60, close=50.0, volume=500_000),
            "JUNK": _daily_bars(60, close=1.0, volume=1_000),
        }
        result = liquidity_filter(bars, as_of, min_avg_daily_dollar_vol=20_000_000)
        assert "AAPL" in result
        assert "JUNK" not in result

    def test_strict_pit_excludes_as_of_bar(self):
        # The bar on as_of itself must NOT be included (look-ahead).
        as_of = date(2024, 4, 1)
        # 30 bars before as_of all pass, then one huge bar ON as_of (should be ignored)
        dates_before = pd.date_range(end=pd.Timestamp(as_of) - pd.Timedelta(days=1),
                                     periods=30, freq="B")
        dates_on = pd.DatetimeIndex([pd.Timestamp(as_of)])
        closes = [1.0] * 30 + [1e9]   # giant close on as_of
        volumes = [1_000] * 30 + [1_000]
        df = pd.DataFrame({"close": closes, "volume": volumes},
                          index=dates_before.append(dates_on))
        result = liquidity_filter({"SYM": df}, as_of, min_avg_daily_dollar_vol=20_000_000)
        assert "SYM" not in result  # $1*$1k avg on the 30 real bars → fail

    def test_insufficient_history_excluded(self):
        # Only 10 bars, lookback=60 → min_bars = 30 → excluded
        as_of = date(2024, 4, 1)
        bars = {"FEW": _daily_bars(10, close=1000.0, volume=1_000_000)}
        result = liquidity_filter(bars, as_of, lookback_days=60,
                                  min_avg_daily_dollar_vol=1_000_000)
        assert "FEW" not in result

    def test_empty_bars_excluded(self):
        as_of = date(2024, 4, 1)
        result = liquidity_filter({"X": pd.DataFrame()}, as_of)
        assert "X" not in result


# ---------------------------------------------------------------------------
# apply_sector_cap
# ---------------------------------------------------------------------------

class TestApplySectorCap:
    def test_sector_cap_enforced(self):
        # 5 tech stocks ranked 1-5, cap=0.30, n=10 → max 3 tech accepted
        ranked = ["T1", "T2", "T3", "T4", "T5", "F1", "F2", "F3", "F4", "F5"]
        sector_map = {f"T{i}": "Tech" for i in range(1, 6)}
        sector_map.update({f"F{i}": "Finance" for i in range(1, 6)})
        result = apply_sector_cap(ranked, sector_map, cap=0.30, n_target=10)
        tech_count = sum(1 for s in result if s.startswith("T"))
        assert tech_count <= 3
        assert len(result) <= 10

    def test_unknown_sector_capped(self):
        # Symbols not in sector_map → bucketed as UNKNOWN, still capped
        ranked = [f"U{i}" for i in range(20)]
        result = apply_sector_cap(ranked, {}, cap=0.30, n_target=10)
        # UNKNOWN capped at floor(0.30 * 10) = 3
        assert len(result) <= 3

    def test_preserves_rank_order(self):
        ranked = ["A", "B", "C", "D", "E"]
        sector_map = {s: "X" for s in ranked}
        result = apply_sector_cap(ranked, sector_map, cap=1.0, n_target=5)
        assert result == ["A", "B", "C", "D", "E"]

    def test_n_target_cap(self):
        ranked = [f"S{i}" for i in range(100)]
        sector_map = {s: f"SEC{i % 10}" for i, s in enumerate(ranked)}
        result = apply_sector_cap(ranked, sector_map, cap=1.0, n_target=20)
        assert len(result) == 20


# ---------------------------------------------------------------------------
# compute_target_portfolio
# ---------------------------------------------------------------------------

class TestComputeTargetPortfolio:
    def test_no_existing_holdings(self):
        ranked = [f"S{i}" for i in range(20)]
        delta = compute_target_portfolio(ranked, [], n_target=10,
                                         add_rank_threshold=10, drop_rank_threshold=15)
        assert len(delta.target) == 10
        assert set(delta.to_add) == set(delta.target)
        assert delta.to_drop == []
        assert delta.held == []

    def test_hysteresis_hold_within_drop_band(self):
        # Held symbol at rank 25, drop_threshold=30 → kept
        ranked = [f"S{i}" for i in range(40)]
        holdings = ["S24"]  # 0-indexed → rank 25
        delta = compute_target_portfolio(ranked, holdings, n_target=10,
                                         add_rank_threshold=5, drop_rank_threshold=30)
        assert "S24" in delta.held
        assert "S24" not in delta.to_drop

    def test_hysteresis_drop_beyond_threshold(self):
        # Held symbol at rank 35, drop_threshold=30 → dropped
        ranked = [f"S{i}" for i in range(40)]
        holdings = ["S34"]  # rank 35
        delta = compute_target_portfolio(ranked, holdings, n_target=10,
                                         add_rank_threshold=5, drop_rank_threshold=30)
        assert "S34" in delta.to_drop
        assert "S34" not in delta.held

    def test_symbol_missing_from_ranking_dropped(self):
        ranked = ["A", "B", "C"]
        holdings = ["GONE"]  # not in ranked_symbols at all
        delta = compute_target_portfolio(ranked, holdings, n_target=3,
                                         add_rank_threshold=3, drop_rank_threshold=3)
        assert "GONE" in delta.to_drop

    def test_target_never_exceeds_n_target(self):
        ranked = [f"S{i}" for i in range(100)]
        holdings = [f"S{i}" for i in range(10)]
        delta = compute_target_portfolio(ranked, holdings, n_target=30,
                                         add_rank_threshold=15, drop_rank_threshold=30)
        assert len(delta.target) <= 30


# ---------------------------------------------------------------------------
# compute_equal_weights
# ---------------------------------------------------------------------------

class TestComputeEqualWeights:
    def test_weights_sum_to_gross(self):
        syms = ["A", "B", "C", "D"]
        weights = compute_equal_weights(syms, total_equity=10_000.0,
                                        gross_exposure_multiplier=1.0)
        assert abs(sum(weights.values()) - 10_000.0) < 1e-6

    def test_regime_multiplier_scales_gross(self):
        syms = ["A", "B", "C"]
        weights = compute_equal_weights(syms, total_equity=10_000.0,
                                        gross_exposure_multiplier=0.3)
        assert abs(sum(weights.values()) - 3_000.0) < 1e-6

    def test_empty_symbols_returns_empty(self):
        assert compute_equal_weights([], total_equity=10_000.0) == {}

    def test_each_weight_equal(self):
        syms = ["A", "B", "C", "D", "E"]
        weights = compute_equal_weights(syms, total_equity=5_000.0)
        vals = list(weights.values())
        assert all(abs(v - vals[0]) < 1e-9 for v in vals)


# ---------------------------------------------------------------------------
# Phase RB.2 — inverse-vol sizing tests
# ---------------------------------------------------------------------------

from app.strategy.portfolio_construction import compute_inverse_vol_weights


def _bars_with_vol(n=60, vol=0.01, seed=0):
    """Create synthetic close series with given daily vol."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, vol, n)
    prices = 100.0 * np.cumprod(1 + rets)
    idx = pd.date_range("2023-01-03", periods=n, freq="B")
    return pd.DataFrame({"close": prices, "open": prices, "high": prices,
                         "low": prices, "volume": [1_000_000] * n}, index=idx)


class TestInverseVolWeights:
    def test_low_vol_gets_higher_weight(self):
        syms = ["HIGH", "LOW"]
        bars = {
            "HIGH": _bars_with_vol(60, vol=0.03, seed=1),
            "LOW": _bars_with_vol(60, vol=0.005, seed=2),
        }
        as_of = date(2023, 4, 1)
        weights = compute_inverse_vol_weights(syms, bars, as_of, total_equity=10000.0)
        assert weights["LOW"] > weights["HIGH"]

    def test_weights_sum_to_gross_exposure(self):
        syms = [f"S{i}" for i in range(5)]
        bars = {s: _bars_with_vol(60, vol=0.01 * (i + 1), seed=i) for i, s in enumerate(syms)}
        as_of = date(2023, 4, 1)
        weights = compute_inverse_vol_weights(syms, bars, as_of, total_equity=50000.0,
                                              gross_exposure_multiplier=0.7)
        assert abs(sum(weights.values()) - 50000.0 * 0.7) < 1.0

    def test_cap_limits_extreme_weights(self):
        # One symbol has very low vol (would get huge weight without cap)
        syms = ["LOWVOL", "NORM"]
        bars = {
            "LOWVOL": _bars_with_vol(60, vol=0.0001, seed=10),
            "NORM": _bars_with_vol(60, vol=0.02, seed=11),
        }
        as_of = date(2023, 4, 1)
        eq_weight = 5000.0  # 10000 / 2
        weights = compute_inverse_vol_weights(syms, bars, as_of, total_equity=10000.0,
                                              min_weight_mult=0.5, max_weight_mult=2.0)
        assert weights["LOWVOL"] <= eq_weight * 2.0 + 1.0  # capped at 2x

    def test_fallback_to_equal_weight_insufficient_history(self):
        # Only 5 bars — insufficient for 20d lookback
        syms = ["A", "B"]
        idx = pd.date_range("2023-01-03", periods=5, freq="B")
        short_bars = pd.DataFrame({"close": [100.0] * 5, "open": [100.0] * 5,
                                   "high": [100.0] * 5, "low": [100.0] * 5,
                                   "volume": [1_000_000] * 5}, index=idx)
        bars = {"A": short_bars, "B": short_bars}
        weights = compute_inverse_vol_weights(syms, bars, date(2023, 1, 15),
                                              total_equity=10000.0)
        # Should fall back to equal weight
        assert abs(weights.get("A", 0) - weights.get("B", 0)) < 1.0

    def test_empty_symbols_returns_empty(self):
        weights = compute_inverse_vol_weights([], {}, date(2023, 1, 1), total_equity=10000.0)
        assert weights == {}
