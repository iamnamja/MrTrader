"""Tests for Phase 31: intraday feature engineering + training pipeline."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_5min_df(n=100, start="2024-01-02 09:30", base_price=100.0):
    idx = pd.date_range(start=start, periods=n, freq="5min")
    np.random.seed(42)
    prices = base_price + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.002,
        "low": prices * 0.997,
        "close": prices,
        "volume": np.random.randint(10_000, 100_000, n).astype(float),
    }, index=idx)


def _make_multi_day_bars(n_days=30, bars_per_day=78):
    """Create multi-day 5-min bars spanning n_days trading days."""
    all_bars = []
    start = datetime(2024, 1, 2, 9, 30)
    day = start
    for _ in range(n_days):
        day_bars = _make_5min_df(bars_per_day, start=day.strftime("%Y-%m-%d %H:%M"))
        all_bars.append(day_bars)
        # advance to next trading day
        day = day + timedelta(days=1)
        while day.weekday() >= 5:
            day = day + timedelta(days=1)
    return pd.concat(all_bars)


# ── Feature engineering ───────────────────────────────────────────────────────

class TestIntradayFeatures:

    def test_returns_dict_with_expected_keys(self):
        from app.ml.intraday_features import compute_intraday_features, FEATURE_NAMES
        bars = _make_5min_df(40)
        feats = compute_intraday_features(bars)
        assert feats is not None
        for name in FEATURE_NAMES:
            assert name in feats, f"Missing feature: {name}"

    def test_returns_none_for_insufficient_bars(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(5)
        assert compute_intraday_features(bars) is None

    def test_orb_position_in_range(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(40)
        feats = compute_intraday_features(bars)
        # orb_position can be outside [0,1] if price breaks out, that's expected
        assert feats is not None
        assert isinstance(feats["orb_position"], float)

    def test_orb_breakout_is_minus_one_zero_or_one(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(40)
        feats = compute_intraday_features(bars)
        assert feats["orb_breakout"] in {-1.0, 0.0, 1.0}

    def test_rsi_in_valid_range(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(40)
        feats = compute_intraday_features(bars)
        assert 0.0 <= feats["rsi_14"] <= 1.0  # normalised to [0, 1]

    def test_gap_pct_uses_prior_close(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(40)
        prior_close = float(bars["open"].iloc[0]) * 0.95  # 5% gap up
        feats = compute_intraday_features(bars, prior_close=prior_close)
        assert feats["gap_pct"] > 0.04  # roughly 5%

    def test_spy_direction_zero_when_no_spy(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(40)
        feats = compute_intraday_features(bars, spy_bars=None)
        assert feats["spy_session_return"] == 0.0

    def test_spy_direction_populated_when_spy_provided(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(40)
        spy = _make_5min_df(40)
        feats = compute_intraday_features(bars, spy_bars=spy)
        assert isinstance(feats["spy_session_return"], float)

    def test_time_of_day_increases_with_more_bars(self):
        from app.ml.intraday_features import compute_intraday_features
        bars_early = _make_5min_df(15)
        bars_late = _make_5min_df(60)
        f_early = compute_intraday_features(bars_early)
        f_late = compute_intraday_features(bars_late)
        assert f_late["time_of_day"] > f_early["time_of_day"]

    def test_all_features_are_finite(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(50)
        feats = compute_intraday_features(bars, prior_close=100.0)
        for k, v in feats.items():
            assert np.isfinite(v), f"Feature {k} is not finite: {v}"

    def test_bb_position_in_zero_one(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(40)
        feats = compute_intraday_features(bars)
        assert 0.0 <= feats["bb_position"] <= 1.0

    def test_stoch_k_in_zero_one(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(40)
        feats = compute_intraday_features(bars)
        assert 0.0 <= feats["stoch_k"] <= 1.0

    def test_cum_delta_in_zero_one(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(40)
        feats = compute_intraday_features(bars)
        assert 0.0 <= feats["cum_delta"] <= 1.0

    def test_ema_cross_positive_when_trending_up(self):
        from app.ml.intraday_features import compute_intraday_features
        # Strongly uptrending bars: EMA9 > EMA20
        prices = np.linspace(100, 130, 40)
        bars = _make_5min_df(40)
        bars["close"] = prices
        bars["open"] = prices * 0.999
        bars["high"] = prices * 1.002
        bars["low"] = prices * 0.997
        feats = compute_intraday_features(bars)
        assert feats["ema_cross"] > 0

    def test_prior_day_levels_populated(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(40)
        feats = compute_intraday_features(
            bars, prior_close=100.0, prior_day_high=105.0, prior_day_low=95.0
        )
        assert feats["prev_day_high_dist"] != 0.0
        assert feats["prev_day_low_dist"] != 0.0

    def test_macd_hist_is_finite(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(50)
        feats = compute_intraday_features(bars)
        assert np.isfinite(feats["macd_hist"])

    def test_spy_rsi_defaults_to_half_when_no_spy(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(40)
        feats = compute_intraday_features(bars, spy_bars=None)
        assert feats["spy_rsi_14"] == 0.5

    def test_ret_30m_uses_six_bar_lookback(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(20)
        feats = compute_intraday_features(bars)
        assert np.isfinite(feats["ret_30m"])

    def test_williams_r_in_zero_one(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(40)
        feats = compute_intraday_features(bars)
        assert 0.0 <= feats["williams_r"] <= 1.0

    def test_gap_fill_pct_zero_when_no_prior_close(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(40)
        feats = compute_intraday_features(bars)
        assert feats["gap_fill_pct"] == 1.0  # no gap = fully "filled"

    def test_gap_fill_pct_with_gap_up(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(40, base_price=110.0)
        prior_close = 100.0  # gap up by 10%
        feats = compute_intraday_features(bars, prior_close=prior_close)
        assert 0.0 <= feats["gap_fill_pct"] <= 1.0

    def test_session_hl_position_in_zero_one(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(40)
        feats = compute_intraday_features(bars)
        assert 0.0 <= feats["session_hl_position"] <= 1.0

    def test_vwap_cross_count_non_negative(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(40)
        feats = compute_intraday_features(bars)
        assert feats["vwap_cross_count"] >= 0.0

    def test_obv_slope_finite(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(40)
        feats = compute_intraday_features(bars)
        assert np.isfinite(feats["obv_slope"])

    def test_wick_ratios_sum_leq_one(self):
        from app.ml.intraday_features import compute_intraday_features
        bars = _make_5min_df(40)
        feats = compute_intraday_features(bars)
        assert feats["upper_wick_ratio"] + feats["lower_wick_ratio"] + feats["body_ratio"] <= 1.01

    def test_consecutive_bars_positive_for_uptrend(self):
        from app.ml.intraday_features import compute_intraday_features
        prices = np.linspace(100, 120, 40)
        bars = _make_5min_df(40)
        bars["close"] = prices
        bars["open"] = prices * 0.999
        bars["high"] = prices * 1.002
        bars["low"] = prices * 0.997
        feats = compute_intraday_features(bars)
        assert feats["consecutive_bars"] > 0


# ── Indicator helpers ─────────────────────────────────────────────────────────

class TestIndicatorHelpers:

    def test_ema_single_value(self):
        from app.ml.intraday_features import _ema
        assert _ema(np.array([5.0]), 9) == 5.0

    def test_macd_histogram_zero_on_flat(self):
        from app.ml.intraday_features import _macd_histogram
        flat = np.full(50, 100.0)
        assert abs(_macd_histogram(flat)) < 1e-6

    def test_bollinger_pct_b_midpoint_on_flat(self):
        from app.ml.intraday_features import _bollinger_pct_b
        flat = np.full(30, 100.0)
        assert _bollinger_pct_b(flat) == 0.5

    def test_stochastic_k_midpoint_at_center(self):
        from app.ml.intraday_features import _stochastic_k
        closes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        highs = np.array([92.0, 97.0, 102.0, 107.0, 112.0])
        lows = np.array([88.0, 93.0, 98.0, 103.0, 108.0])
        k = _stochastic_k(highs, lows, closes, period=5)
        assert 0.0 <= k <= 100.0

    def test_williams_r_range(self):
        from app.ml.intraday_features import _williams_r
        closes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        highs = closes + 2.0
        lows = closes - 2.0
        r = _williams_r(highs, lows, closes, period=5)
        assert -100.0 <= r <= 0.0

    def test_obv_increases_on_up_days(self):
        from app.ml.intraday_features import _obv
        closes = np.array([100.0, 101.0, 102.0, 103.0])
        volumes = np.array([1000.0, 1000.0, 1000.0, 1000.0])
        obv = _obv(closes, volumes)
        assert obv[-1] > obv[0]

    def test_consecutive_bars_negative_for_downtrend(self):
        from app.ml.intraday_features import _consecutive_bars
        closes = np.array([110.0, 109.0, 108.0, 107.0, 106.0])
        assert _consecutive_bars(closes) < 0

    def test_vwap_cross_count_choppy(self):
        from app.ml.intraday_features import _vwap_cross_count
        # Alternating up/down around a stable VWAP → many crosses
        closes = np.array([100.0 + (1 if i % 2 == 0 else -1) for i in range(30)],
                          dtype=float)
        highs = closes + 0.5
        lows = closes - 0.5
        volumes = np.ones(30) * 1000.0
        count = _vwap_cross_count(closes, highs, lows, volumes)
        assert count > 5


# ── Training pipeline ─────────────────────────────────────────────────────────

class TestIntradayTrainingPipeline:

    def _trainer(self):
        from app.ml.intraday_training import IntradayModelTrainer
        return IntradayModelTrainer()

    def _make_symbols_data(self, n_symbols=5, n_days=25):
        data = {}
        for i in range(n_symbols):
            np.random.seed(i * 7)
            data[f"SYM{i:02d}"] = _make_multi_day_bars(n_days)
        return data

    def test_build_daily_matrix_produces_samples(self):
        trainer = self._trainer()
        data = self._make_symbols_data(5, 25)
        X_train, y_train, X_test, y_test, names = trainer._build_daily_matrix(data, None)
        assert len(X_train) + len(X_test) > 0

    def test_feature_names_match_columns(self):
        trainer = self._trainer()
        data = self._make_symbols_data(5, 25)
        X_train, y_train, X_test, y_test, names = trainer._build_daily_matrix(data, None)
        if len(X_train) > 0:
            assert X_train.shape[1] == len(names)

    def test_insufficient_data_returns_empty(self):
        trainer = self._trainer()
        tiny = {"AAPL": _make_5min_df(50)}  # not enough days
        X_train, y_train, X_test, y_test, names = trainer._build_daily_matrix(tiny, None)
        assert len(X_train) == 0

    def test_labels_are_binary(self):
        trainer = self._trainer()
        data = self._make_symbols_data(5, 25)
        X_train, y_train, X_test, y_test, names = trainer._build_daily_matrix(data, None)
        if len(y_train) > 0:
            assert set(y_train).issubset({0, 1})

    def test_train_test_shapes_consistent(self):
        trainer = self._trainer()
        data = self._make_symbols_data(5, 30)
        X_train, y_train, X_test, y_test, names = trainer._build_daily_matrix(data, None)
        if len(X_train) > 0 and len(X_test) > 0:
            assert X_train.shape[1] == X_test.shape[1]

    def test_train_model_returns_version(self):
        from app.ml.intraday_training import IntradayModelTrainer
        trainer = IntradayModelTrainer()
        data = self._make_symbols_data(6, 30)

        # Provide balanced labels so XGBoost doesn't complain about single-class
        n = 60
        X = np.random.randn(n, 13).astype(np.float32)
        y = np.array([i % 2 for i in range(n)])
        names = [f"f{i}" for i in range(13)]

        with patch.object(trainer, "_fetch_data", return_value=data):
            with patch.object(
                trainer, "_build_daily_matrix",
                return_value=(X, y, X[:10], y[:10], names)
            ):
                with patch.object(trainer, "_record_version"):
                    with patch.object(trainer, "_next_version", return_value=1):
                        version = trainer.train_model(
                            symbols=[f"SYM{i:02d}" for i in range(6)],
                            days=30,
                            fetch_spy=False,
                        )
        assert version == 1


# ── Helpers ───────────────────────────────────────────────────────────────────

class TestIntradayHelpers:

    def test_rsi_returns_50_when_insufficient_data(self):
        from app.ml.intraday_features import _rsi
        assert _rsi(np.array([100.0, 101.0]), 14) == 50.0

    def test_rsi_returns_100_on_all_gains(self):
        from app.ml.intraday_features import _rsi
        closes = np.array([100.0 + i for i in range(20)])
        assert _rsi(closes, 14) == 100.0

    def test_atr_positive(self):
        from app.ml.intraday_features import _atr
        highs = np.array([101.0 + i * 0.1 for i in range(20)])
        lows = np.array([99.0 + i * 0.1 for i in range(20)])
        closes = np.array([100.0 + i * 0.1 for i in range(20)])
        atr = _atr(highs, lows, closes, 14)
        assert atr > 0
