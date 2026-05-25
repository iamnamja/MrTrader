"""Tests for IcCompositeScorer and V219 IC weights (Phase 88)."""
import numpy as np
import pandas as pd
import pytest


class TestV219Weights:
    def test_weights_sum_to_one(self):
        from app.ml.factor_scorer import V219_IC_WEIGHTS
        total = sum(V219_IC_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6, f"V219_IC_WEIGHTS sum = {total}, expected 1.0"

    def test_no_negative_weights(self):
        from app.ml.factor_scorer import V219_IC_WEIGHTS
        negatives = {k: v for k, v in V219_IC_WEIGHTS.items() if v < 0}
        assert not negatives, f"Negative weights found: {negatives}"

    def test_top_features_dominate(self):
        """ix_momentum_vol, momentum_252d_ex1m, price_to_52w_low should be top-3."""
        from app.ml.factor_scorer import V219_IC_WEIGHTS
        sorted_feats = sorted(V219_IC_WEIGHTS.items(), key=lambda x: x[1], reverse=True)
        top3 = {k for k, _ in sorted_feats[:3]}
        expected_top3 = {"ix_momentum_vol", "momentum_252d_ex1m", "price_to_52w_low"}
        assert top3 == expected_top3, f"Top-3 features: {top3}, expected {expected_top3}"

    def test_vol_percentile_52w_excluded(self):
        """vol_percentile_52w has negative IC (-0.13) and must be excluded."""
        from app.ml.factor_scorer import V219_IC_WEIGHTS
        assert "vol_percentile_52w" not in V219_IC_WEIGHTS, \
            "vol_percentile_52w has negative h20 IC IR and must not appear in V219_IC_WEIGHTS"


class TestIcCompositeScorer:
    def _make_sym_data(self, n=300, seed=42):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        prices = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
        vol = rng.integers(100_000, 5_000_000, n).astype(float)
        return pd.DataFrame({
            "open": prices * 0.998,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": vol,
        }, index=dates)

    def _make_symbols_data(self, syms=("AAPL", "MSFT", "GOOG"), n=300):
        data = {}
        for i, sym in enumerate(syms):
            data[sym] = self._make_sym_data(n=n, seed=i * 10 + 1)
        spy = self._make_sym_data(n=n, seed=999)
        data["SPY"] = spy
        return data

    def test_scorer_returns_list(self):
        from app.ml.factor_scorer import IcCompositeScorer
        scorer = IcCompositeScorer()
        symbols_data = self._make_symbols_data()
        day = pd.Timestamp("2023-01-02")
        result = scorer(day, symbols_data)
        assert isinstance(result, list)

    def test_scorer_excludes_spy(self):
        from app.ml.factor_scorer import IcCompositeScorer
        scorer = IcCompositeScorer()
        symbols_data = self._make_symbols_data()
        day = pd.Timestamp("2023-01-02")
        result = scorer(day, symbols_data)
        syms = [s for s, _ in result]
        assert "SPY" not in syms, "SPY should not appear in scored symbols"

    def test_scorer_sorted_descending(self):
        from app.ml.factor_scorer import IcCompositeScorer
        scorer = IcCompositeScorer()
        symbols_data = self._make_symbols_data(syms=("A", "B", "C", "D", "E"), n=300)
        day = pd.Timestamp("2023-01-02")
        result = scorer(day, symbols_data)
        if len(result) >= 2:
            scores = [s for _, s in result]
            assert scores == sorted(scores, reverse=True), "Results must be sorted descending"

    def test_scorer_empty_on_insufficient_data(self):
        from app.ml.factor_scorer import IcCompositeScorer
        scorer = IcCompositeScorer()
        # Only 30 bars — not enough for any feature
        symbols_data = self._make_symbols_data(n=30)
        day = pd.Timestamp("2022-02-10")
        result = scorer(day, symbols_data)
        assert result == [], "Should return empty list with insufficient history"

    def test_cli_flag_importable(self):
        """--rebalance-ic-composite flag must be parseable from the CLI."""
        import sys
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/walkforward_tier3.py", "--help"],
            capture_output=True, text=True, cwd="c:/Projects/MrTrader"
        )
        assert "rebalance-ic-composite" in result.stdout, \
            "--rebalance-ic-composite flag not found in CLI help"
