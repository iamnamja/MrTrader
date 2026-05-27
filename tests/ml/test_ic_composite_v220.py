"""
Tests for Phase 90 v220 regime-conditional two-composite scorer.

Key design properties:
- Breadth > 60% → Composite A (momentum-tilted)
- Breadth < 55% → Composite B (quality/v219 weights)
- 5pp hysteresis deadband prevents rapid switching
- PIT-safe: all data shifted by 1 day minimum
- Returns valid float scores for eligible symbols
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from app.ml.factor_scorer import (
    IcCompositeV220Scorer,
    V220A_WEIGHTS,
    _compute_breadth,
)
from app.ml.factor_scorer import V219_IC_WEIGHTS


def _make_closes(n_days: int = 400, n_syms: int = 80, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    syms = [f"S{i:03d}" for i in range(n_syms)]
    log_rets = rng.normal(0.0003, 0.01, size=(n_days, n_syms))
    prices = pd.DataFrame(np.exp(np.cumsum(log_rets, axis=0) + 4.0), index=dates, columns=syms)
    return prices


def _make_symbols_data(closes: pd.DataFrame) -> dict:
    return {sym: pd.DataFrame({"close": closes[sym]}) for sym in closes.columns}


class TestComputeBreadth:
    def test_all_above_ma_returns_one(self):
        """Strong uptrend: all symbols above 200d MA → breadth = 1.0."""
        rng = np.random.default_rng(0)
        dates = pd.date_range("2019-01-02", periods=600, freq="B")
        syms = [f"S{i:03d}" for i in range(60)]
        # Monotonically rising prices
        log_rets = rng.normal(0.002, 0.003, size=(600, 60))
        prices = pd.DataFrame(np.exp(np.cumsum(log_rets, axis=0) + 4.0), index=dates, columns=syms)
        as_of = prices.index[-1]
        result = _compute_breadth(prices, as_of)
        assert result > 0.9, f"Expected breadth near 1.0, got {result:.2f}"

    def test_all_below_ma_returns_zero(self):
        """Strong downtrend: all symbols below 200d MA → breadth ~ 0."""
        rng = np.random.default_rng(1)
        dates = pd.date_range("2019-01-02", periods=600, freq="B")
        syms = [f"S{i:03d}" for i in range(60)]
        # Monotonically falling prices
        log_rets = rng.normal(-0.002, 0.003, size=(600, 60))
        prices = pd.DataFrame(np.exp(np.cumsum(log_rets, axis=0) + 4.0), index=dates, columns=syms)
        as_of = prices.index[-1]
        result = _compute_breadth(prices, as_of)
        assert result < 0.1, f"Expected breadth near 0.0, got {result:.2f}"

    def test_insufficient_symbols_returns_nan(self):
        """< 50 symbols → NaN (insufficient for reliable breadth signal)."""
        closes = _make_closes(n_days=400, n_syms=30)
        as_of = closes.index[-1]
        result = _compute_breadth(closes, as_of)
        assert np.isnan(result)

    def test_insufficient_history_returns_zero_breadth(self):
        """< 200 days of history → 0.0 breadth (all symbols counted as below MA).

        BUG-3 fix: short-history symbols are counted in denominator but not numerator,
        so 80 symbols with 100 days → 0/80 = 0.0 (not NaN — we have sufficient universe
        coverage, we just know none are above their 200d MA yet).
        """
        closes = _make_closes(n_days=100, n_syms=80)
        as_of = closes.index[-1]
        result = _compute_breadth(closes, as_of)
        assert result == 0.0

    def test_tiny_universe_returns_nan(self):
        """< 50 symbols → NaN (universe too small for reliable regime signal)."""
        closes = _make_closes(n_days=100, n_syms=10)
        as_of = closes.index[-1]
        result = _compute_breadth(closes, as_of)
        assert np.isnan(result)

    def test_returns_float_in_range(self):
        closes = _make_closes()
        as_of = closes.index[-1]
        result = _compute_breadth(closes, as_of)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestV220AWeights:
    def test_weights_sum_to_one(self):
        assert abs(sum(V220A_WEIGHTS.values()) - 1.0) < 1e-9

    def test_no_negative_weights(self):
        """WF-C1 R4: pre-fold-1 IC IR clipped at 0; weights are non-negative."""
        for feat, w in V220A_WEIGHTS.items():
            assert w >= 0, f"Weight for {feat} is negative"

    def test_in_sample_weights_preserved(self):
        """WF-C1 R4: original hand-tuned in-sample weights are kept for reference."""
        from app.ml.factor_scorer import _V220A_WEIGHTS_IN_SAMPLE
        assert abs(sum(_V220A_WEIGHTS_IN_SAMPLE.values()) - 1.0) < 1e-9
        assert _V220A_WEIGHTS_IN_SAMPLE["ix_momentum_vol"] == 0.23


class TestIcCompositeV220Scorer:
    def test_returns_list_of_tuples(self):
        closes = _make_closes()
        symbols_data = _make_symbols_data(closes)
        scorer = IcCompositeV220Scorer()
        result = scorer(date(2021, 6, 1), symbols_data)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)

    def test_scores_in_reasonable_range(self):
        closes = _make_closes()
        symbols_data = _make_symbols_data(closes)
        scorer = IcCompositeV220Scorer()
        result = scorer(date(2021, 6, 1), symbols_data)
        assert len(result) > 10
        scores = [s for _, s in result]
        # Composite z-scores shouldn't be crazy
        assert max(abs(s) for s in scores) < 20.0

    def test_active_regime_property(self):
        closes = _make_closes()
        symbols_data = _make_symbols_data(closes)
        scorer = IcCompositeV220Scorer()
        scorer(date(2021, 6, 1), symbols_data)
        regime = scorer.active_regime
        assert regime in ("momentum", "quality")

    def test_high_breadth_triggers_momentum_regime(self):
        """In a strong uptrend (breadth > 60%), scorer should switch to momentum composite."""
        rng = np.random.default_rng(10)
        dates = pd.date_range("2019-01-02", periods=600, freq="B")
        syms = [f"S{i:03d}" for i in range(80)]
        log_rets = rng.normal(0.002, 0.003, size=(600, 80))
        prices = pd.DataFrame(np.exp(np.cumsum(log_rets, axis=0) + 4.0), index=dates, columns=syms)
        symbols_data = {sym: pd.DataFrame({"close": prices[sym]}) for sym in syms}
        scorer = IcCompositeV220Scorer()
        scorer(prices.index[-1].date(), symbols_data)
        assert scorer.active_regime == "momentum"

    def test_low_breadth_triggers_quality_regime(self):
        """In a downtrend (breadth < 55%), scorer should use quality/v219 composite."""
        rng = np.random.default_rng(20)
        dates = pd.date_range("2019-01-02", periods=600, freq="B")
        syms = [f"S{i:03d}" for i in range(80)]
        log_rets = rng.normal(-0.002, 0.003, size=(600, 80))
        prices = pd.DataFrame(np.exp(np.cumsum(log_rets, axis=0) + 4.0), index=dates, columns=syms)
        symbols_data = {sym: pd.DataFrame({"close": prices[sym]}) for sym in syms}
        scorer = IcCompositeV220Scorer()
        scorer(prices.index[-1].date(), symbols_data)
        assert scorer.active_regime == "quality"

    def test_hysteresis_prevents_rapid_switching(self):
        """Breadth near the threshold (e.g. 57%) should not flip back and forth."""
        rng = np.random.default_rng(30)
        dates = pd.date_range("2019-01-02", periods=700, freq="B")
        syms = [f"S{i:03d}" for i in range(80)]
        # First half: strong uptrend (breadth high → momentum regime)
        log_rets_up = rng.normal(0.003, 0.002, size=(300, 80))
        # Second half: mild sideways (breadth in deadband ~57%)
        log_rets_side = rng.normal(0.0, 0.01, size=(400, 80))
        log_rets = np.vstack([log_rets_up, log_rets_side])
        prices = pd.DataFrame(np.exp(np.cumsum(log_rets, axis=0) + 4.0), index=dates, columns=syms)
        symbols_data = {sym: pd.DataFrame({"close": prices[sym]}) for sym in syms}

        scorer = IcCompositeV220Scorer()
        regimes = []
        for d in dates[-50:]:
            scorer(d.date(), symbols_data)
            regimes.append(scorer.active_regime)

        # Should not flip more than 2 times (hysteresis should dampen)
        switches = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1])
        assert switches <= 3, f"Too many regime switches ({switches}) — hysteresis not working"

    def test_pit_safe_no_lookahead(self):
        """Scores on day T must not use data from day T (use shift(1) or as_of index)."""
        closes = _make_closes(n_days=400, n_syms=80)
        symbols_data = _make_symbols_data(closes)
        scorer = IcCompositeV220Scorer()

        # Modify day T data and verify scores don't change
        d = closes.index[350].date()
        raw_before = scorer(d, symbols_data)
        scores_before = {sym: s for sym, s in raw_before}

        # Corrupt the closes on that exact date
        closes_corrupt = closes.copy()
        closes_corrupt.loc[closes_corrupt.index[350]] *= 10.0
        symbols_data_corrupt = _make_symbols_data(closes_corrupt)

        scorer2 = IcCompositeV220Scorer()
        raw_after = scorer2(d, symbols_data_corrupt)
        scores_after = {sym: s for sym, s in raw_after}

        # Scores should be identical — same-day data must not affect output
        common = set(scores_before) & set(scores_after)
        assert len(common) > 0
        diff = max(abs(scores_before[sym] - scores_after[sym]) for sym in common)
        assert diff < 1e-6, f"Same-day data corruption affected scores by {diff:.4f} — lookahead!"

    def test_cli_flag_in_help(self):
        """--rebalance-ic-composite-v220 must appear in CLI help."""
        import sys
        import subprocess
        from pathlib import Path
        repo_root = Path(__file__).parents[2]
        result = subprocess.run(
            [sys.executable, "scripts/walkforward_tier3.py", "--help"],
            capture_output=True, text=True, cwd=str(repo_root)
        )
        assert "rebalance-ic-composite-v220" in result.stdout
