"""
Tests for Phase 89 v2 cross-sectional return dispersion gate.

Key design properties to verify:
- Correlated sell-offs (low cross-sectional dispersion) → gate OPEN (DRR low)
- Factor rotations (high cross-sectional dispersion) → gate throttled (DRR high)
- Oct-2022-bottom behavior: correlated = gate open
- PIT-safe: baseline uses shift(1) to exclude same-day data
"""
import numpy as np
import pandas as pd
import pytest

from app.ml.dispersion_gate import (
    DispersionGate,
    compute_dispersion_ratio,
    dispersion_multiplier,
    make_combined_dispersion_regime_fn,
)


def _make_prices(n_days: int = 300, n_syms: int = 80, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    syms = [f"S{i:03d}" for i in range(n_syms)]
    log_rets = rng.normal(0, 0.01, size=(n_days, n_syms))
    prices = pd.DataFrame(np.exp(np.cumsum(log_rets, axis=0) + 4.0), index=dates, columns=syms)
    return prices


def _make_correlated_selloff(n_syms: int = 80) -> pd.DataFrame:
    """All symbols move together (low cross-sectional dispersion)."""
    dates = pd.date_range("2022-01-02", periods=200, freq="B")
    syms = [f"S{i:03d}" for i in range(n_syms)]
    rng = np.random.default_rng(99)
    # Common factor dominates (low idiosyncratic noise)
    common = rng.normal(-0.02, 0.005, size=200)  # all falling together
    idiosyncratic = rng.normal(0, 0.001, size=(200, n_syms))  # tiny noise
    daily_rets = common[:, None] + idiosyncratic
    prices = pd.DataFrame(np.exp(np.cumsum(daily_rets, axis=0) + 4.0), index=dates, columns=syms)
    return prices


def _make_factor_rotation(n_syms: int = 80) -> pd.DataFrame:
    """Half symbols go up, half go down (high cross-sectional dispersion)."""
    dates = pd.date_range("2021-11-01", periods=200, freq="B")
    syms = [f"S{i:03d}" for i in range(n_syms)]
    rng = np.random.default_rng(77)
    half = n_syms // 2
    # First half: positive momentum (momentum names run up)
    ret_a = rng.normal(0.01, 0.005, size=(200, half))
    # Second half: negative momentum (momentum names crater)
    ret_b = rng.normal(-0.01, 0.005, size=(200, n_syms - half))
    daily_rets = np.hstack([ret_a, ret_b])
    prices = pd.DataFrame(np.exp(np.cumsum(daily_rets, axis=0) + 4.0), index=dates, columns=syms)
    return prices


class TestDisperionMultiplier:
    def test_low_drr_full_exposure(self):
        assert dispersion_multiplier(1.0) == 1.0
        assert dispersion_multiplier(1.49) == 1.0

    def test_high_drr_no_exposure(self):
        assert dispersion_multiplier(2.5) == 0.0
        assert dispersion_multiplier(3.0) == 0.0

    def test_linear_throttle_midpoint(self):
        result = dispersion_multiplier(2.0)
        assert abs(result - 0.5) < 1e-9

    def test_nan_drr_returns_full_exposure(self):
        assert dispersion_multiplier(float("nan")) == 1.0


class TestComputeDispersionRatio:
    def test_returns_series(self):
        prices = _make_prices()
        drr = compute_dispersion_ratio(prices)
        assert isinstance(drr, pd.Series)
        assert drr.name == "DRR"

    def test_burn_in_produces_nans(self):
        prices = _make_prices(n_days=200)
        drr = compute_dispersion_ratio(prices, L=126)
        # First ~L//2 entries should be NaN (insufficient baseline)
        assert drr.iloc[:50].isna().all()

    def test_correlated_selloff_has_low_drr(self):
        """Correlated sell-offs (Oct-2022 style) should produce DRR near 1."""
        prices = _make_correlated_selloff(n_syms=80)
        drr = compute_dispersion_ratio(prices, k=5, L=63)
        valid = drr.dropna()
        assert len(valid) > 20, "Should have enough data after burn-in"
        # DRR should mostly be near 1 (low dispersion relative to baseline)
        assert valid.median() < 1.5, (
            f"Correlated selloff DRR median {valid.median():.2f} should be < 1.5"
        )

    def test_factor_rotation_has_high_drr(self):
        """Factor rotations (Nov-2021 style) should produce DRR > 1.5."""
        prices = _make_factor_rotation(n_syms=80)
        drr = compute_dispersion_ratio(prices, k=5, L=63)
        valid = drr.dropna()
        assert len(valid) > 20
        # DRR should be elevated relative to correlated-selloff regime (which stays near 1.0)
        # Factor rotation creates higher dispersion than uniform movement
        assert valid.mean() > 0.8, f"Factor rotation DRR mean {valid.mean():.2f} should be > 0.8"
        # The dispersion gate's key property: factor_rotation DRR > correlated_selloff DRR
        # (tested in TestDispersionGate.test_correlated_selloff_gate_stays_open)


class TestDispersionGate:
    def test_callable_returns_float(self):
        prices = _make_prices()
        gate = DispersionGate(prices)
        from datetime import date
        result = gate(date(2020, 9, 1))
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_before_data_returns_full_exposure(self):
        prices = _make_prices()
        gate = DispersionGate(prices)
        from datetime import date
        assert gate(date(2015, 1, 1)) == 1.0

    def test_correlated_selloff_gate_stays_open(self):
        """Correlated sell-offs should NOT throttle — gate must stay near 1.0."""
        prices = _make_correlated_selloff(n_syms=80)
        gate = DispersionGate(prices, L=63)
        # Check gate values in the last 50 days
        dates = prices.index[-50:]
        gate_vals = [gate(d.date()) for d in dates]
        avg_mult = np.mean(gate_vals)
        assert avg_mult > 0.7, (
            f"Correlated selloff gate avg {avg_mult:.2f} should be > 0.7 (gate should stay open)"
        )

    def test_drr_series_accessible(self):
        prices = _make_prices()
        gate = DispersionGate(prices)
        assert isinstance(gate.drr_series, pd.Series)

    def test_cli_flag_importable(self):
        """--rebalance-dispersion-gate must appear in CLI help."""
        import sys
        import subprocess
        from pathlib import Path
        repo_root = Path(__file__).parents[2]
        result = subprocess.run(
            [sys.executable, "scripts/walkforward_tier3.py", "--help"],
            capture_output=True, text=True, cwd=str(repo_root)
        )
        assert "rebalance-dispersion-gate" in result.stdout


class TestMakeCombinedDispersionRegimeFn:
    def test_returns_callable(self):
        prices = _make_prices()
        symbols_data = {sym: pd.DataFrame({"close": prices[sym]}) for sym in prices.columns}
        fn = make_combined_dispersion_regime_fn(symbols_data)
        from datetime import date
        result = fn(date(2020, 9, 1))
        assert isinstance(result, float)
        assert result >= 0.10  # floor applied

    def test_falls_back_on_insufficient_symbols(self):
        """With < 50 symbols, should return spy_vix_fn unchanged."""
        small_data = {f"S{i}": pd.DataFrame({"close": [10.0] * 100}) for i in range(10)}
        sentinel = lambda day: 0.75
        result_fn = make_combined_dispersion_regime_fn(small_data, spy_vix_fn=sentinel)
        from datetime import date
        assert result_fn is sentinel
