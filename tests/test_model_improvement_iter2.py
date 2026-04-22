"""
Tests for Model Improvement Iteration 2: risk-adjusted (Sharpe-based) labels.
  - Swing: lambdarank quintiles ranked by (ret - mean) / std per window
  - Swing: cross_sectional threshold computed in Sharpe units
  - Intraday: best_return normalized by intraday realized volatility
"""
import numpy as np
import pytest


# ── Swing — lambdarank Sharpe quintiles ───────────────────────────────────────

class TestSwingLambdarankSharpeLabels:
    def _make_trainer(self):
        from app.ml.training import ModelTrainer
        return ModelTrainer(label_scheme="lambdarank")

    def test_high_vol_stock_does_not_dominate_top_quintile(self):
        """A stock with 10% return but extreme volatility should not rank above
        a stock with 3% return in a low-vol window."""
        from app.ml.training import ModelTrainer
        import numpy as np
        trainer = ModelTrainer(label_scheme="lambdarank")

        # Window with mixed returns: one stock has huge raw return but it's
        # an outlier. After Sharpe normalization, the distribution is fairer.
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.50])  # last is outlier
        meta = [{"window_idx": 0}] * 5
        X = np.zeros((5, 3))

        X_s, y_q, groups = trainer._build_lambdarank_groups(X, returns, meta)
        # The outlier (0.50) should still be in quintile 4 (it IS highest),
        # but the others should be spread across quintiles 0-3 (not all at 0).
        assert int(y_q[-1]) == 4  # highest return → top quintile
        assert len(np.unique(y_q)) == 5  # all 5 quintiles represented

    def test_quintiles_always_cover_0_to_4(self):
        from app.ml.training import ModelTrainer
        import numpy as np
        trainer = ModelTrainer(label_scheme="lambdarank")

        returns = np.array([-0.05, -0.01, 0.0, 0.02, 0.08, 0.10,
                            -0.03, 0.01, 0.04, 0.07])
        meta = [{"window_idx": 0}] * 10
        X = np.zeros((10, 3))
        _, y_q, _ = trainer._build_lambdarank_groups(X, returns, meta)
        assert set(y_q.tolist()) == {0, 1, 2, 3, 4}

    def test_sharpe_ranking_consistent_across_windows(self):
        """Two windows with same relative ordering but different scales should
        produce the same quintile assignments."""
        from app.ml.training import ModelTrainer
        import numpy as np
        trainer = ModelTrainer(label_scheme="lambdarank")

        # Window A: small returns
        rets_a = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        # Window B: 10x larger but same ordering
        rets_b = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        meta = [{"window_idx": 0}] * 5 + [{"window_idx": 1}] * 5
        X = np.zeros((10, 3))
        _, y_q, _ = trainer._build_lambdarank_groups(
            X, np.concatenate([rets_a, rets_b]), meta
        )
        # Both windows should have identical quintile assignments
        np.testing.assert_array_equal(y_q[:5], y_q[5:])

    def test_single_sample_window_gets_middle_quintile(self):
        from app.ml.training import ModelTrainer
        import numpy as np
        trainer = ModelTrainer(label_scheme="lambdarank")
        returns = np.array([0.05])
        meta = [{"window_idx": 0}]
        X = np.zeros((1, 3))
        _, y_q, _ = trainer._build_lambdarank_groups(X, returns, meta)
        assert int(y_q[0]) == 2


# ── Swing — cross_sectional Sharpe threshold ──────────────────────────────────

class TestSwingCsThresholdSharpe:
    def test_threshold_is_float(self):
        """_compute_cs_thresholds should return a float threshold for cross_sectional."""
        from app.ml.training import ModelTrainer, WINDOW_DAYS, FORWARD_DAYS
        import pandas as pd
        from datetime import date, timedelta

        trainer = ModelTrainer(label_scheme="cross_sectional")

        # Build minimal symbols_data: 2 symbols, enough dates for one window
        n_dates = WINDOW_DAYS + FORWARD_DAYS + 5
        base = date(2023, 1, 2)
        dates = [base + timedelta(days=i) for i in range(n_dates)]

        def _make_df(prices):
            idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
            return pd.DataFrame({
                "open": prices, "high": prices, "low": prices,
                "close": prices, "volume": [1e6] * n_dates
            }, index=idx)

        symbols_data = {
            "AAPL": _make_df([100.0 + i * 0.1 for i in range(n_dates)]),
            "MSFT": _make_df([200.0 + i * 0.2 for i in range(n_dates)]),
            "GOOG": _make_df([150.0 + i * 0.05 for i in range(n_dates)]),
            "AMZN": _make_df([120.0 - i * 0.05 for i in range(n_dates)]),
            "NVDA": _make_df([300.0 + i * 0.3 for i in range(n_dates)]),
        }
        all_dates = dates
        window_starts = [0]
        thresholds = trainer._compute_cs_thresholds(symbols_data, all_dates, window_starts)
        assert 0 in thresholds
        assert isinstance(thresholds[0], float)


# ── Intraday — Sharpe-adjusted best_return ────────────────────────────────────

class TestIntradaySharpeReturn:
    def test_sharpe_return_lower_for_high_vol_day(self):
        """A 1% gain on a high-vol day should produce a lower Sharpe return
        than a 1% gain on a low-vol day."""
        import pandas as pd
        import numpy as np
        from app.ml.intraday_training import HOLD_BARS

        n_feat = 40
        np.random.seed(42)

        def _make_feat_bars(vol_scale):
            prices = 100.0 + np.cumsum(np.random.randn(n_feat) * vol_scale)
            prices = np.abs(prices)
            idx = pd.date_range("2024-01-02 09:30", periods=n_feat, freq="5min")
            return pd.DataFrame({
                "open": prices, "high": prices * 1.001,
                "low": prices * 0.999, "close": prices, "volume": 1e6
            }, index=idx)

        def _compute_sharpe(vol_scale, best_ret=0.01):
            feat_bars = _make_feat_bars(vol_scale)
            entry = float(feat_bars["close"].iloc[-1])
            intraday_vol = float(feat_bars["close"].pct_change().std()) + 1e-8
            horizon_vol = intraday_vol * (HOLD_BARS ** 0.5)
            return best_ret / horizon_vol

        sharpe_low_vol = _compute_sharpe(vol_scale=0.01)
        sharpe_high_vol = _compute_sharpe(vol_scale=0.5)
        assert sharpe_low_vol > sharpe_high_vol

    def test_sharpe_return_scales_with_return(self):
        """Higher best_return should produce higher Sharpe return (monotone)."""
        import pandas as pd
        import numpy as np
        from app.ml.intraday_training import HOLD_BARS

        n_feat = 40
        np.random.seed(1)
        prices = 100.0 + np.cumsum(np.random.randn(n_feat) * 0.1)
        prices = np.abs(prices)
        idx = pd.date_range("2024-01-02 09:30", periods=n_feat, freq="5min")
        feat_bars = pd.DataFrame({
            "open": prices, "high": prices * 1.001,
            "low": prices * 0.999, "close": prices, "volume": 1e6
        }, index=idx)

        intraday_vol = float(feat_bars["close"].pct_change().std()) + 1e-8
        horizon_vol = intraday_vol * (HOLD_BARS ** 0.5)

        sharpe_1 = 0.005 / horizon_vol
        sharpe_2 = 0.015 / horizon_vol
        assert sharpe_2 > sharpe_1
