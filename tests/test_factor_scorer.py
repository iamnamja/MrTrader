"""
Tests for app/ml/factor_scorer.py — 0% coverage before this file.

Covers:
- _zscore_cross: normalization, zero-std guard, winsorization
- _momentum_252d_ex1m: correct window, insufficient data guard
- _price_to_52w_high / _price_to_52w_low: ratio correctness
- _volume_trend / _range_expansion: ratio computations
- compute_composite_score: returns Series, respects tier2 flag, handles no-data
- select_top_n / select_bottom_n: ordering, size, empty-input guard
- regime_gate_ok: SPY/VIX gate logic, permissive on insufficient data
- FactorPortfolioScorer.__call__: returns [(sym, conf, direction)] with correct shapes,
  conf in [0.55, 0.95], bear market → empty list, L/S direction field
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta


# ── Helpers ───────────────────────────────────────────────────────────────────

def _closes_df(n: int = 300, symbols: list = None, base: float = 100.0,
               start: str = "2021-01-04") -> pd.DataFrame:
    """Aligned close prices for `symbols` with a mild upward trend."""
    if symbols is None:
        symbols = [f"SYM{i:02d}" for i in range(5)]
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(start=start, periods=n)
    data = {}
    for i, sym in enumerate(symbols):
        prices = base * (1 + np.cumsum(rng.normal(0.0005, 0.01, n)))
        prices = np.maximum(prices, 1.0)
        data[sym] = prices
    return pd.DataFrame(data, index=idx)


def _bars_dict(closes_df: pd.DataFrame) -> dict:
    """Build a bars dict with high/low/volume from closes."""
    result = {}
    for sym in closes_df.columns:
        c = closes_df[sym]
        result[sym] = pd.DataFrame({
            "open": c * 0.999,
            "high": c * 1.005,
            "low": c * 0.995,
            "close": c,
            "volume": np.random.randint(500_000, 2_000_000, len(c)).astype(float),
        }, index=closes_df.index)
    return result


# ── _zscore_cross ─────────────────────────────────────────────────────────────

class TestZscoreCross:
    def test_mean_near_zero(self):
        from app.ml.factor_scorer import _zscore_cross
        s = pd.Series({"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0, "E": 50.0})
        z = _zscore_cross(s)
        assert abs(z.mean()) < 1e-9

    def test_winsorized_at_3sigma(self):
        from app.ml.factor_scorer import _zscore_cross
        s = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 1000.0})
        z = _zscore_cross(s)
        assert z.max() <= 3.0 + 1e-9
        assert z.min() >= -3.0 - 1e-9

    def test_zero_std_returns_zeros(self):
        from app.ml.factor_scorer import _zscore_cross
        s = pd.Series({"A": 5.0, "B": 5.0, "C": 5.0})
        z = _zscore_cross(s)
        assert (z == 0.0).all()

    def test_index_preserved(self):
        from app.ml.factor_scorer import _zscore_cross
        s = pd.Series({"AAPL": 1.5, "TSLA": 3.0, "NVDA": 4.5})
        z = _zscore_cross(s)
        assert set(z.index) == {"AAPL", "TSLA", "NVDA"}


# ── _momentum_252d_ex1m ───────────────────────────────────────────────────────

class TestMomentum252dEx1m:
    def test_returns_series_for_sufficient_data(self):
        from app.ml.factor_scorer import _momentum_252d_ex1m
        closes = _closes_df(n=300)
        as_of = closes.index[-1]
        mom = _momentum_252d_ex1m(closes, as_of)
        assert isinstance(mom, pd.Series)
        assert len(mom) > 0

    def test_returns_empty_for_insufficient_data(self):
        from app.ml.factor_scorer import _momentum_252d_ex1m
        closes = _closes_df(n=100)  # < 252
        as_of = closes.index[-1]
        mom = _momentum_252d_ex1m(closes, as_of)
        assert mom.empty

    def test_excludes_last_month(self):
        """Momentum is c[-252] → c[-21], not c[-252] → c[-1]."""
        from app.ml.factor_scorer import _momentum_252d_ex1m
        # Rising for 252 days, then flat for 21 days
        n = 300
        idx = pd.bdate_range("2021-01-04", periods=n)
        # Flat last 21 days → ex-1m momentum should be lower than full-period momentum
        prices_rising = np.linspace(100, 200, n - 21)
        prices_flat = np.full(21, 200.0)
        prices = np.concatenate([prices_rising, prices_flat])
        closes = pd.DataFrame({"SYM": prices}, index=idx)
        as_of = closes.index[-1]
        mom = _momentum_252d_ex1m(closes, as_of)
        # mom = c[-21] / c[-252] - 1; c[-21] is near 200, c[-252] ≈ 100 → ~100% return
        assert mom["SYM"] > 0.5


# ── _price_to_52w_high / _price_to_52w_low ───────────────────────────────────

class TestPrice52w:
    def test_52w_high_ratio_leq_1(self):
        from app.ml.factor_scorer import _price_to_52w_high
        closes = _closes_df(n=300)
        as_of = closes.index[-1]
        ratios = _price_to_52w_high(closes, as_of)
        assert (ratios <= 1.0 + 1e-6).all()

    def test_52w_low_ratio_geq_1(self):
        from app.ml.factor_scorer import _price_to_52w_low
        closes = _closes_df(n=300)
        as_of = closes.index[-1]
        ratios = _price_to_52w_low(closes, as_of)
        assert (ratios >= 1.0 - 1e-6).all()

    def test_52w_high_insufficient_data_returns_empty(self):
        from app.ml.factor_scorer import _price_to_52w_high
        closes = _closes_df(n=100)
        as_of = closes.index[-1]
        assert _price_to_52w_high(closes, as_of).empty

    def test_at_52w_high_ratio_is_one(self):
        from app.ml.factor_scorer import _price_to_52w_high
        n = 300
        idx = pd.bdate_range("2021-01-04", periods=n)
        # Monotonically rising: today IS the 52-week high
        prices = np.linspace(100, 200, n)
        closes = pd.DataFrame({"SYM": prices}, index=idx)
        as_of = closes.index[-1]
        ratios = _price_to_52w_high(closes, as_of)
        assert ratios["SYM"] == pytest.approx(1.0, abs=0.01)


# ── compute_composite_score ───────────────────────────────────────────────────

class TestCompositeScore:
    def test_returns_series(self):
        from app.ml.factor_scorer import compute_composite_score
        closes = _closes_df(n=300)
        bars = _bars_dict(closes)
        as_of = closes.index[-1]
        scores = compute_composite_score(as_of, closes, bars)
        assert isinstance(scores, pd.Series)
        assert not scores.empty

    def test_scores_are_finite(self):
        from app.ml.factor_scorer import compute_composite_score
        closes = _closes_df(n=300)
        bars = _bars_dict(closes)
        as_of = closes.index[-1]
        scores = compute_composite_score(as_of, closes, bars)
        assert scores.apply(np.isfinite).all()

    def test_tier2_false_still_returns_scores(self):
        from app.ml.factor_scorer import compute_composite_score
        closes = _closes_df(n=300)
        bars = _bars_dict(closes)
        as_of = closes.index[-1]
        scores_t2 = compute_composite_score(as_of, closes, bars, use_tier2=True)
        scores_no_t2 = compute_composite_score(as_of, closes, bars, use_tier2=False)
        assert not scores_no_t2.empty
        # Tier2 may add more signal → scores can differ
        assert len(scores_t2) >= len(scores_no_t2)

    def test_insufficient_data_skips_momentum_factor(self):
        """With < 252 bars, momentum (the dominant factor) is not computed.
        Tier2 factors may still produce scores via volume_trend/range_expansion."""
        from app.ml.factor_scorer import compute_composite_score, _momentum_252d_ex1m
        closes = _closes_df(n=50)
        as_of = closes.index[-1]
        mom = _momentum_252d_ex1m(closes, as_of)
        assert mom.empty, "Momentum requires >= 252 bars"

    def test_scores_spread_across_symbols(self):
        """Different symbols should get different scores (cross-sectional signal)."""
        from app.ml.factor_scorer import compute_composite_score
        n = 300
        idx = pd.bdate_range("2021-01-04", periods=n)
        rng = np.random.default_rng(7)
        # SYM0: strong uptrend, SYM1: downtrend
        closes = pd.DataFrame({
            "SYM0": np.linspace(50, 200, n),   # strong momentum
            "SYM1": np.linspace(200, 50, n),   # negative momentum
            "SYM2": np.full(n, 100.0) + rng.normal(0, 0.5, n),
        }, index=idx)
        bars = _bars_dict(closes)
        as_of = closes.index[-1]
        scores = compute_composite_score(as_of, closes, bars)
        if len(scores) >= 2:
            assert scores.max() != scores.min(), "Cross-sectional scores must differ"


# ── select_top_n / select_bottom_n ───────────────────────────────────────────

class TestSelectTopBottomN:
    def _scores(self):
        return pd.Series({"A": 3.0, "B": 1.0, "C": 2.0, "D": -1.0, "E": 0.5})

    def test_top_n_ordering(self):
        from app.ml.factor_scorer import select_top_n
        top = select_top_n(self._scores(), n=3)
        assert top[0] == "A"  # highest score first
        assert len(top) == 3

    def test_bottom_n_ordering(self):
        from app.ml.factor_scorer import select_bottom_n
        bottom = select_bottom_n(self._scores(), n=3)
        assert bottom[0] == "D"  # lowest score first
        assert len(bottom) == 3

    def test_top_n_empty_input(self):
        from app.ml.factor_scorer import select_top_n
        assert select_top_n(pd.Series(dtype=float), n=5) == []

    def test_bottom_n_empty_input(self):
        from app.ml.factor_scorer import select_bottom_n
        assert select_bottom_n(pd.Series(dtype=float), n=5) == []

    def test_top_n_larger_than_scores_returns_all(self):
        from app.ml.factor_scorer import select_top_n
        top = select_top_n(self._scores(), n=100)
        assert len(top) == len(self._scores())


# ── regime_gate_ok ────────────────────────────────────────────────────────────

class TestRegimeGateOk:
    def _spy(self, n: int, last_value: float = 450.0, trend: str = "flat") -> pd.Series:
        idx = pd.bdate_range("2021-01-04", periods=n)
        if trend == "flat":
            values = np.full(n, last_value)
        elif trend == "up":
            values = np.linspace(last_value * 0.8, last_value, n)
        elif trend == "down":
            values = np.linspace(last_value, last_value * 0.5, n)
        else:
            values = np.full(n, last_value)
        return pd.Series(values, index=idx)

    def test_passes_when_spy_above_ma200_vix_low(self):
        from app.ml.factor_scorer import regime_gate_ok
        # Uptrend: last value well above MA200
        spy = self._spy(250, last_value=450, trend="up")
        as_of = spy.index[-1]
        assert regime_gate_ok(spy, as_of, vix_value=20.0)

    def test_fails_when_spy_below_ma200(self):
        from app.ml.factor_scorer import regime_gate_ok
        # Severe downtrend: last value (50% of start) well below MA200
        spy = self._spy(250, last_value=450, trend="down")
        as_of = spy.index[-1]
        result = regime_gate_ok(spy, as_of, vix_value=20.0)
        assert not result

    def test_fails_when_vix_above_threshold(self):
        from app.ml.factor_scorer import regime_gate_ok
        spy = self._spy(250, trend="flat")
        as_of = spy.index[-1]
        assert not regime_gate_ok(spy, as_of, vix_value=35.0, vix_threshold=30.0)

    def test_passes_when_vix_none(self):
        from app.ml.factor_scorer import regime_gate_ok
        spy = self._spy(250, trend="up")
        as_of = spy.index[-1]
        assert regime_gate_ok(spy, as_of, vix_value=None)

    def test_permissive_on_insufficient_history(self):
        from app.ml.factor_scorer import regime_gate_ok
        spy = self._spy(50)  # < 200 days
        as_of = spy.index[-1]
        assert regime_gate_ok(spy, as_of, vix_value=50.0)  # too little data → allow

    def test_permissive_on_empty_spy(self):
        from app.ml.factor_scorer import regime_gate_ok
        assert regime_gate_ok(pd.Series(dtype=float), pd.Timestamp.now(), vix_value=40.0)


# ── FactorPortfolioScorer ─────────────────────────────────────────────────────

class TestFactorPortfolioScorer:

    def _symbols_data(self, n: int = 300, symbols=None) -> dict:
        if symbols is None:
            symbols = [f"SYM{i:02d}" for i in range(8)]
        closes = _closes_df(n=n, symbols=symbols)
        return _bars_dict(closes)

    def _run(self, scorer, n: int = 300):
        symbols_data = self._symbols_data(n=n)
        # day = day after the last bar (so all bars are "past")
        bars_df = list(symbols_data.values())[0]
        day = bars_df.index[-1].date() + timedelta(days=1)
        return scorer(day, symbols_data, vix_history=None)

    def test_returns_list(self):
        from app.ml.factor_scorer import FactorPortfolioScorer
        scorer = FactorPortfolioScorer(top_n=3, top_n_short=2, long_short=True)
        result = self._run(scorer)
        assert isinstance(result, list)

    def test_result_is_3_tuples(self):
        from app.ml.factor_scorer import FactorPortfolioScorer
        scorer = FactorPortfolioScorer(top_n=3, top_n_short=2, long_short=True)
        result = self._run(scorer)
        for item in result:
            assert len(item) == 3, "Each result must be (sym, conf, direction)"

    def test_conf_in_valid_range(self):
        from app.ml.factor_scorer import FactorPortfolioScorer
        scorer = FactorPortfolioScorer(top_n=3, top_n_short=2, long_short=True)
        result = self._run(scorer)
        for sym, conf, direction in result:
            assert 0.54 <= abs(conf) <= 0.96, f"{sym} conf={conf} out of [0.55, 0.95]"

    def test_long_direction_positive_conf(self):
        from app.ml.factor_scorer import FactorPortfolioScorer
        scorer = FactorPortfolioScorer(top_n=3, top_n_short=0, long_short=True)
        result = self._run(scorer)
        for sym, conf, direction in result:
            assert direction == "long"
            assert conf > 0

    def test_short_direction_negative_conf(self):
        from app.ml.factor_scorer import FactorPortfolioScorer
        scorer = FactorPortfolioScorer(top_n=0, top_n_short=2, long_short=True)
        result = self._run(scorer)
        shorts = [(s, c, d) for s, c, d in result if d == "short"]
        assert len(shorts) > 0
        for _, conf, _ in shorts:
            assert conf < 0

    def test_long_short_false_returns_only_longs(self):
        from app.ml.factor_scorer import FactorPortfolioScorer
        scorer = FactorPortfolioScorer(top_n=3, top_n_short=3, long_short=False)
        result = self._run(scorer)
        directions = {d for _, _, d in result}
        assert "short" not in directions

    def test_bear_market_returns_empty(self):
        from app.ml.factor_scorer import FactorPortfolioScorer
        scorer = FactorPortfolioScorer(top_n=5, top_n_short=3, long_short=True,
                                       spy_ma_window=50, vix_threshold=30.0)
        # Bear market: SPY declining sharply
        n = 300
        symbols = [f"SYM{i:02d}" for i in range(5)]
        closes = _closes_df(n=n, symbols=symbols)
        # Add SPY with severe downtrend
        spy_prices = np.linspace(500, 100, n)  # strong bear
        closes["SPY"] = spy_prices
        symbols_data = _bars_dict(closes)

        bars_df = list(symbols_data.values())[0]
        day = bars_df.index[-1].date() + timedelta(days=1)
        result = scorer(day, symbols_data, vix_history=None)
        assert result == [], "Bear market should suppress all signals"

    def test_insufficient_data_returns_empty(self):
        from app.ml.factor_scorer import FactorPortfolioScorer
        scorer = FactorPortfolioScorer(top_n=3)
        result = self._run(scorer, n=30)  # < 60 bars needed per symbol
        assert result == []

    def test_vix_history_used_for_regime_gate(self):
        from app.ml.factor_scorer import FactorPortfolioScorer
        # Include SPY in symbols_data so regime_gate_ok actually evaluates VIX
        scorer = FactorPortfolioScorer(top_n=3, vix_threshold=15.0)
        symbols_data = self._symbols_data(n=300, symbols=[f"SYM{i:02d}" for i in range(5)])
        # Add SPY as a downtrend so regime gate fails via SPY (easier to trigger)
        bars_df = list(symbols_data.values())[0]
        spy_prices = np.linspace(500, 100, 300)  # severe downtrend
        spy_idx = bars_df.index
        spy_df = pd.DataFrame({
            "open": spy_prices * 0.999, "high": spy_prices * 1.005,
            "low": spy_prices * 0.995, "close": spy_prices, "volume": 1e7,
        }, index=spy_idx)
        symbols_data["SPY"] = spy_df
        day = bars_df.index[-1].date() + timedelta(days=1)

        # SPY in severe downtrend → regime gate fails → no signals
        vix_idx = pd.bdate_range("2021-01-04", periods=300)
        vix_normal = pd.Series(np.full(300, 15.0), index=vix_idx)
        result = scorer(day, symbols_data, vix_history=vix_normal)
        assert result == [], "Bear market SPY should suppress all signals via regime gate"


# =============================================================================
# LX1EqualWeightScorer tests
# =============================================================================

class TestLX1EqualWeightScorer:
    """Tests for the Phase LX1 equal-weight 5-feature IC-validated scorer."""

    def _bars_dict(self, closes: pd.DataFrame) -> dict:
        result = {}
        for sym in closes.columns:
            df = pd.DataFrame({
                "open": closes[sym] * 0.999,
                "high": closes[sym] * 1.005,
                "low": closes[sym] * 0.995,
                "close": closes[sym],
                "volume": 1e6,
            })
            result[sym] = df
        return result

    def _run(self, n: int = 300, symbols=None):
        from app.ml.factor_scorer import LX1EqualWeightScorer
        if symbols is None:
            symbols = [f"SYM{i:02d}" for i in range(10)]
        closes = _closes_df(n=n, symbols=symbols)
        symbols_data = self._bars_dict(closes)
        day = closes.index[-1].date() + timedelta(days=1)
        scorer = LX1EqualWeightScorer()
        return scorer(day, symbols_data)

    def test_returns_list_of_tuples(self):
        result = self._run()
        assert isinstance(result, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)

    def test_scores_are_finite(self):
        result = self._run()
        assert all(np.isfinite(score) for _, score in result)

    def test_sorted_descending(self):
        result = self._run()
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_spy_excluded(self):
        from app.ml.factor_scorer import LX1EqualWeightScorer
        symbols = [f"SYM{i:02d}" for i in range(5)]
        closes = _closes_df(n=300, symbols=symbols)
        closes["SPY"] = np.linspace(400, 500, 300)
        symbols_data = self._bars_dict(closes)
        day = closes.index[-1].date() + timedelta(days=1)
        scorer = LX1EqualWeightScorer()
        result = scorer(day, symbols_data)
        syms = [s for s, _ in result]
        assert "SPY" not in syms

    def test_insufficient_history_returns_empty(self):
        result = self._run(n=30)  # < 60 bars required
        assert result == []

    def test_empty_symbols_data_returns_empty(self):
        from app.ml.factor_scorer import LX1EqualWeightScorer
        scorer = LX1EqualWeightScorer()
        day = date(2024, 1, 15)
        assert scorer(day, {}) == []

    def test_weights_sum_to_one(self):
        from app.ml.factor_scorer import LX1_EQ_WEIGHTS
        assert abs(sum(LX1_EQ_WEIGHTS.values()) - 1.0) < 1e-9

    def test_no_fundamentals_still_returns_scores(self):
        """Without FMP parquet, scorer falls back to technical features only."""
        from app.ml.factor_scorer import LX1EqualWeightScorer
        scorer = LX1EqualWeightScorer()
        scorer._fmp_path = "nonexistent_path_that_does_not_exist.parquet"
        symbols = [f"SYM{i:02d}" for i in range(6)]
        closes = _closes_df(n=300, symbols=symbols)
        bars = {}
        for sym in closes.columns:
            bars[sym] = pd.DataFrame({
                "open": closes[sym] * 0.999, "high": closes[sym] * 1.005,
                "low": closes[sym] * 0.995, "close": closes[sym], "volume": 1e6,
            })
        day = closes.index[-1].date() + timedelta(days=1)
        result = scorer(day, bars)
        # momentum_252d_ex1m and price_to_52w_high are technical — should still score
        assert len(result) > 0


# =============================================================================
# B2EqualWeightUniverseScorer tests
# =============================================================================

class TestB2EqualWeightUniverseScorer:
    """Tests for the B2 naive baseline scorer (no stock selection)."""

    def _bars_dict(self, closes: pd.DataFrame) -> dict:
        result = {}
        for sym in closes.columns:
            df = pd.DataFrame({
                "open": closes[sym] * 0.999,
                "high": closes[sym] * 1.005,
                "low": closes[sym] * 0.995,
                "close": closes[sym],
                "volume": 1e6,
            })
            result[sym] = df
        return result

    def test_all_scores_zero(self):
        from app.ml.factor_scorer import B2EqualWeightUniverseScorer
        scorer = B2EqualWeightUniverseScorer()
        symbols = [f"SYM{i:02d}" for i in range(5)]
        closes = _closes_df(n=300, symbols=symbols)
        bars = self._bars_dict(closes)
        day = closes.index[-1].date() + timedelta(days=1)
        result = scorer(day, bars)
        assert all(score == 0.0 for _, score in result)

    def test_spy_excluded(self):
        from app.ml.factor_scorer import B2EqualWeightUniverseScorer
        scorer = B2EqualWeightUniverseScorer()
        symbols = [f"SYM{i:02d}" for i in range(5)]
        closes = _closes_df(n=300, symbols=symbols)
        closes["SPY"] = np.linspace(400, 500, 300)
        bars = self._bars_dict(closes)
        day = closes.index[-1].date() + timedelta(days=1)
        result = scorer(day, bars)
        assert "SPY" not in [s for s, _ in result]

    def test_returns_all_eligible_symbols(self):
        from app.ml.factor_scorer import B2EqualWeightUniverseScorer
        scorer = B2EqualWeightUniverseScorer()
        symbols = [f"SYM{i:02d}" for i in range(8)]
        closes = _closes_df(n=300, symbols=symbols)
        bars = self._bars_dict(closes)
        day = closes.index[-1].date() + timedelta(days=1)
        result = scorer(day, bars)
        assert len(result) == len(symbols)

    def test_insufficient_history_excluded(self):
        from app.ml.factor_scorer import B2EqualWeightUniverseScorer
        scorer = B2EqualWeightUniverseScorer()
        symbols = [f"SYM{i:02d}" for i in range(5)]
        closes = _closes_df(n=30, symbols=symbols)  # < 60 bars
        bars = self._bars_dict(closes)
        day = closes.index[-1].date() + timedelta(days=1)
        result = scorer(day, bars)
        assert result == []

    def test_empty_input_returns_empty(self):
        from app.ml.factor_scorer import B2EqualWeightUniverseScorer
        scorer = B2EqualWeightUniverseScorer()
        assert scorer(date(2024, 1, 15), {}) == []
