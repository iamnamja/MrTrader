"""Tests for scripts/diag_naive_baseline.py — Phase A3 baseline strategies."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from scripts.diag_naive_baseline import (
    _compute_metrics,
    _max_drawdown,
    _sharpe,
    run_momentum_baseline,
    run_spy_ma_timing,
)
from scripts.walkforward.cost_models import cost_from_turnover


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_price_series(
    n_days: int = 252,
    drift: float = 0.0003,
    vol: float = 0.01,
    seed: int = 42,
    start: str = "2022-01-03",
) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(drift, vol, n_days)
    prices = np.exp(np.cumsum(log_rets))
    idx = pd.bdate_range(start, periods=n_days)
    return pd.Series(prices, index=idx)


def _make_bars_map(
    n_symbols: int = 20,
    n_days: int = 300,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start_date = date(2022, 1, 3)
    idx = pd.bdate_range(str(start_date), periods=n_days)
    bars_map = {}
    for i in range(n_symbols):
        log_rets = rng.normal(0.0003, 0.012, n_days)
        prices = np.exp(np.cumsum(log_rets))
        bars_map[f"SYM{i:03d}"] = pd.DataFrame({"close": prices}, index=idx)
    return bars_map


# ── Sharpe / drawdown helpers ─────────────────────────────────────────────────

class TestSharpe:
    def test_positive_drift_positive_sharpe(self):
        s = pd.Series([0.001] * 252)
        assert _sharpe(s) > 0

    def test_zero_std_returns_zero(self):
        assert _sharpe(pd.Series([0.0] * 10)) == 0.0

    def test_empty_returns_zero(self):
        assert _sharpe(pd.Series(dtype=float)) == 0.0

    def test_annualised_scaling(self):
        daily = pd.Series([0.001] * 252)
        sr_ann = _sharpe(daily, annualise=True)
        sr_raw = _sharpe(daily, annualise=False)
        assert abs(sr_ann - sr_raw * np.sqrt(252)) < 1e-6


class TestMaxDrawdown:
    def test_flat_equity_zero_drawdown(self):
        eq = pd.Series([1.0, 1.0, 1.0, 1.0])
        assert _max_drawdown(eq) == 0.0

    def test_monotone_rising_zero_drawdown(self):
        eq = pd.Series([1.0, 1.1, 1.2, 1.3])
        assert _max_drawdown(eq) == pytest.approx(0.0, abs=1e-9)

    def test_known_drawdown(self):
        eq = pd.Series([1.0, 1.2, 0.9, 1.1])
        mdd = _max_drawdown(eq)
        assert abs(mdd - 0.25) < 1e-6  # 1.2 -> 0.9 = 25%

    def test_empty_returns_zero(self):
        assert _max_drawdown(pd.Series(dtype=float)) == 0.0


# ── run_momentum_baseline ─────────────────────────────────────────────────────

class TestMomentumBaseline:
    def test_output_columns(self):
        bars = _make_bars_map(n_symbols=15, n_days=300)
        result = run_momentum_baseline(
            bars, date(2022, 2, 1), date(2022, 12, 30),
            lookback=60, top_pct=0.20, cost_bps_per_side=5.0,
        )
        assert not result.empty
        assert set(result.columns) == {"gross_ret", "turnover", "cost", "net_ret", "equity", "n_holdings"}

    def test_equity_starts_near_one(self):
        bars = _make_bars_map(n_symbols=15, n_days=300)
        result = run_momentum_baseline(
            bars, date(2022, 2, 1), date(2022, 12, 30),
        )
        assert not result.empty
        assert abs(result["equity"].iloc[0] - 1.0) < 0.05

    def test_top_pct_controls_n_holdings(self):
        bars = _make_bars_map(n_symbols=20, n_days=300)
        result = run_momentum_baseline(
            bars, date(2022, 3, 1), date(2022, 12, 30),
            top_pct=0.10,
        )
        if not result.empty:
            # Should hold ~10% of 20 = 2 symbols once rebalanced
            max_holdings = result["n_holdings"].max()
            assert max_holdings <= 5  # some slack for edge months

    def test_cost_applied_on_rebalance_days_only(self):
        bars = _make_bars_map(n_symbols=15, n_days=300)
        result = run_momentum_baseline(
            bars, date(2022, 2, 1), date(2022, 12, 30),
            cost_bps_per_side=5.0,
        )
        if not result.empty:
            # On non-rebalance days turnover is 0
            non_rebalance = result[result["turnover"] == 0.0]
            assert (non_rebalance["cost"] == 0.0).all()

    def test_cost_matches_cost_from_turnover(self):
        bars = _make_bars_map(n_symbols=15, n_days=300)
        result = run_momentum_baseline(
            bars, date(2022, 3, 1), date(2022, 12, 30),
            cost_bps_per_side=5.0,
        )
        if not result.empty:
            for _, row in result[result["turnover"] > 0].iterrows():
                expected_cost = cost_from_turnover(row["turnover"], 5.0)
                assert abs(row["cost"] - expected_cost) < 1e-7  # rounded to 8 decimal places

    def test_net_ret_equals_gross_minus_cost(self):
        bars = _make_bars_map(n_symbols=15, n_days=300)
        result = run_momentum_baseline(
            bars, date(2022, 3, 1), date(2022, 12, 30),
        )
        if not result.empty:
            diff = (result["net_ret"] - (result["gross_ret"] - result["cost"])).abs()
            assert diff.max() < 1e-7  # rounded to 8 decimal places

    def test_empty_bars_returns_empty(self):
        result = run_momentum_baseline({}, date(2022, 1, 1), date(2022, 12, 31))
        assert result.empty

    def test_deterministic_top_selection(self):
        """Given controlled returns, momentum picks the right symbols."""
        idx = pd.bdate_range("2022-01-03", periods=200)
        bars = {}
        # SYM_HIGH has strong uptrend over lookback
        high_prices = np.linspace(100, 200, 200)
        bars["SYM_HIGH"] = pd.DataFrame({"close": high_prices}, index=idx)
        # SYM_LOW has flat price (zero momentum)
        bars["SYM_LOW"] = pd.DataFrame({"close": np.ones(200) * 100}, index=idx)
        for i in range(8):
            bars[f"SYM_MID{i}"] = pd.DataFrame({"close": np.linspace(100, 110, 200)}, index=idx)

        result = run_momentum_baseline(
            bars, date(2022, 4, 1), date(2022, 9, 30),
            lookback=60, top_pct=0.20,
        )
        # We can't easily inspect the holdings set from the result DataFrame,
        # but we can confirm n_holdings reflects top_pct of ~10 symbols = ~2
        if not result.empty:
            rebalance_days = result[result["turnover"] > 0]
            if not rebalance_days.empty:
                assert rebalance_days["n_holdings"].max() <= 4


# ── run_spy_ma_timing ─────────────────────────────────────────────────────────

class TestSpyMaTiming:
    def _make_spy_df(self, n_days: int = 300, drift: float = 0.0003) -> pd.DataFrame:
        prices = _make_price_series(n_days=n_days, drift=drift)
        return pd.DataFrame({"close": prices.values}, index=prices.index)

    def test_output_columns(self):
        spy = self._make_spy_df()
        result = run_spy_ma_timing(spy, ma_window=20)
        assert not result.empty
        assert set(result.columns) == {"position", "gross_ret", "cost", "net_ret", "equity"}

    def test_position_binary(self):
        spy = self._make_spy_df()
        result = run_spy_ma_timing(spy, ma_window=50)
        assert set(result["position"].unique()).issubset({0, 1})

    def test_flat_when_below_ma(self):
        """When SPY is in strong downtrend, should stay in cash (position=0)."""
        n = 300
        # Monotone declining prices → always below MA
        prices = np.linspace(200, 100, n)
        idx = pd.bdate_range("2022-01-03", periods=n)
        spy = pd.DataFrame({"close": prices}, index=idx)
        result = run_spy_ma_timing(spy, ma_window=50)
        if not result.empty:
            # After the MA stabilises, position should be 0 (below MA)
            tail = result.iloc[100:]
            assert (tail["position"] == 0).all(), "Expected cash when SPY declining"
            assert (tail["gross_ret"] == 0.0).all()

    def test_invested_when_above_ma(self):
        """When SPY is in strong uptrend, should be long (position=1)."""
        n = 300
        prices = np.linspace(100, 250, n)
        idx = pd.bdate_range("2022-01-03", periods=n)
        spy = pd.DataFrame({"close": prices}, index=idx)
        result = run_spy_ma_timing(spy, ma_window=50)
        if not result.empty:
            tail = result.iloc[100:]
            assert (tail["position"] == 1).all(), "Expected long when SPY rising"

    def test_cost_on_position_change_only(self):
        spy = self._make_spy_df(n_days=300, drift=0.0001)
        result = run_spy_ma_timing(spy, ma_window=50, cost_bps_per_side=5.0)
        if not result.empty:
            # Days where position changes should have cost; unchanged days should not
            position_changes = result["position"].diff().abs().fillna(0) != 0
            assert (result.loc[~position_changes, "cost"] == 0.0).all()
            assert (result.loc[position_changes, "cost"] > 0.0).all()

    def test_cost_equals_cost_from_turnover_on_switch(self):
        spy = self._make_spy_df()
        result = run_spy_ma_timing(spy, ma_window=50, cost_bps_per_side=5.0)
        if not result.empty:
            expected = cost_from_turnover(1.0, 5.0)
            switching_days = result[result["cost"] > 0]
            for _, row in switching_days.iterrows():
                assert abs(row["cost"] - expected) < 1e-9

    def test_missing_close_column_returns_empty(self):
        spy = pd.DataFrame({"open": [100.0, 101.0]})
        result = run_spy_ma_timing(spy)
        assert result.empty

    def test_equity_compounding(self):
        """equity must equal cumprod of (1 + net_ret)."""
        spy = self._make_spy_df()
        result = run_spy_ma_timing(spy, ma_window=50)
        if not result.empty:
            expected_equity = (1 + result["net_ret"]).cumprod()
            np.testing.assert_allclose(
                result["equity"].values, expected_equity.values, rtol=1e-5
            )


# ── _compute_metrics ──────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_expected_keys(self):
        eq = pd.Series([1.0, 1.01, 1.02, 1.015], index=pd.bdate_range("2022-01-03", periods=4))
        rets = eq.pct_change().dropna()
        m = _compute_metrics(eq, rets)
        assert {"sharpe", "max_drawdown", "calmar", "total_return", "cagr", "n_days"}.issubset(m.keys())

    def test_empty_returns_empty_dict(self):
        m = _compute_metrics(pd.Series(dtype=float), pd.Series(dtype=float))
        assert m == {}

    def test_positive_cagr_for_rising_equity(self):
        idx = pd.bdate_range("2020-01-02", periods=252)
        eq = pd.Series(np.linspace(1.0, 1.5, 252), index=idx)
        rets = eq.pct_change().dropna()
        m = _compute_metrics(eq, rets)
        assert m["cagr"] > 0
        assert m["total_return"] > 0

    def test_sharpe_matches_standalone(self):
        idx = pd.bdate_range("2022-01-03", periods=120)
        eq = pd.Series(np.exp(np.cumsum(np.random.default_rng(1).normal(0.0005, 0.01, 120))), index=idx)
        rets = eq.pct_change().dropna()
        m = _compute_metrics(eq.iloc[1:], rets)
        assert abs(m["sharpe"] - _sharpe(rets)) < 0.01
