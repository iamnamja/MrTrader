"""
Tests for Phase 89 factor stability gate.

The gate measures rolling realized rank-IC between frozen IC composite scores
and forward 20d returns. When rolling IC drops below threshold, factor leadership
is rotating and gross exposure should be reduced.
"""
import numpy as np
import pandas as pd
import pytest

from app.ml.factor_stability_gate import FactorStabilityGate, compute_factor_stability_gate


def _make_synthetic_data(n_days: int = 300, n_syms: int = 80, seed: int = 42):
    """Generate synthetic prices and scores for testing.

    In the first half, scores are strongly predictive of 20d forward returns (IC ~0.4).
    In the second half, scores are pure noise (IC ~0).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    symbols = [f"SYM{i:03d}" for i in range(n_syms)]
    fwd_window = 20

    # True latent 20d forward return for each symbol and day
    latent_fwd = rng.normal(0, 0.05, size=(n_days, n_syms))

    # Prices: built from log-returns derived from latent_fwd (approximate)
    log_returns = rng.normal(0, 0.01, size=(n_days, n_syms))
    # Inject latent forward signal into returns (20d cumulative ≈ latent_fwd)
    for t in range(n_days - fwd_window):
        log_returns[t + fwd_window] += latent_fwd[t] * 0.05
    log_prices = np.cumsum(log_returns, axis=0) + 4.0
    prices = pd.DataFrame(np.exp(log_prices), index=dates, columns=symbols)

    # Scores: in first half, strongly correlated with latent_fwd (IC ~0.4)
    # In second half, pure noise
    scores_vals = np.zeros((n_days, n_syms))
    half = n_days // 2
    for i in range(n_days):
        if i < half:
            scores_vals[i] = latent_fwd[i] + rng.normal(0, 0.02, n_syms)  # high IC
        else:
            scores_vals[i] = rng.normal(0, 1.0, n_syms)  # pure noise

    scores = pd.DataFrame(scores_vals, index=dates, columns=symbols)
    return scores, prices


class TestComputeFactorStabilityGate:
    def test_returns_dataframe_with_required_columns(self):
        scores, prices = _make_synthetic_data()
        result = compute_factor_stability_gate(scores, prices)
        assert set(result.columns) == {"daily_ic", "realized_ic", "raw_gate", "gate"}

    def test_gate_values_are_0_05_or_1(self):
        scores, prices = _make_synthetic_data()
        result = compute_factor_stability_gate(scores, prices)
        assert set(result["gate"].dropna().unique()).issubset({0.0, 0.5, 1.0})

    def test_informative_period_has_higher_gate_than_noise_period(self):
        # Use longer dataset so gate has time to detect the regime shift
        scores, prices = _make_synthetic_data(n_days=600, seed=7)
        # fwd_window=20 + lookback=40 = need ~60 days of burn-in; regime shifts at day 300
        result = compute_factor_stability_gate(scores, prices, lookback_days=40, fwd_window=20)
        gate = result["gate"].dropna()
        # Informative window: days 150-250 (well into first half, past burn-in)
        # Noise window: days 450-550 (well into second half)
        informative_mean = gate.iloc[150:250].mean()
        noise_mean = gate.iloc[450:550].mean()
        assert informative_mean > noise_mean, (
            f"Informative period gate ({informative_mean:.2f}) should be > "
            f"noise period gate ({noise_mean:.2f})"
        )

    def test_raises_on_insufficient_symbols(self):
        dates = pd.date_range("2020-01-02", periods=100, freq="B")
        small_scores = pd.DataFrame(
            np.random.randn(100, 10),
            index=dates,
            columns=[f"S{i}" for i in range(10)],
        )
        small_prices = pd.DataFrame(
            np.abs(np.random.randn(100, 10)) + 10,
            index=dates,
            columns=[f"S{i}" for i in range(10)],
        )
        with pytest.raises(ValueError, match="common symbols"):
            compute_factor_stability_gate(small_scores, small_prices)

    def test_burn_in_period_defaults_to_full_exposure(self):
        scores, prices = _make_synthetic_data(n_days=300)
        result = compute_factor_stability_gate(scores, prices, lookback_days=63, fwd_window=20)
        # First fwd_window + lookback_days/2 rows have no realized IC → should be 1.0
        early = result["gate"].iloc[:20]
        assert (early == 1.0).all(), "Burn-in gate values should default to 1.0"


class TestFactorStabilityGateCallable:
    def test_callable_returns_float(self):
        scores, prices = _make_synthetic_data()
        gate = FactorStabilityGate(scores, prices)
        from datetime import date
        day = date(2020, 9, 1)
        result = gate(day)
        assert isinstance(result, float)
        assert result in (0.0, 0.5, 1.0)

    def test_before_data_returns_full_exposure(self):
        scores, prices = _make_synthetic_data()
        gate = FactorStabilityGate(scores, prices)
        from datetime import date
        result = gate(date(2015, 1, 1))  # before all data
        assert result == 1.0

    def test_gate_series_accessible(self):
        scores, prices = _make_synthetic_data()
        gate = FactorStabilityGate(scores, prices)
        assert isinstance(gate.gate_series, pd.Series)
        assert len(gate.gate_series) > 0

    def test_cli_flag_importable(self):
        """--rebalance-factor-stability-gate must appear in CLI help."""
        import sys
        import subprocess
        from pathlib import Path
        repo_root = Path(__file__).parents[2]
        result = subprocess.run(
            [sys.executable, "scripts/walkforward_tier3.py", "--help"],
            capture_output=True, text=True, cwd=str(repo_root)
        )
        assert "rebalance-factor-stability-gate" in result.stdout, (
            "--rebalance-factor-stability-gate flag not found in CLI help"
        )
