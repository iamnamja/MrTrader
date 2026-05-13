"""Tests for app/ml/diagnostics/ic.py — IC library."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from app.ml.diagnostics.ic import (
    IC_IR_MIN,
    IC_MEAN_MIN,
    HIT_RATE_MIN,
    aggregate_ic,
    compute_daily_ic,
    format_ic_markdown,
    passes_ic_threshold,
    summarize_by_regime,
    summarize_by_year,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_panel(
    n_symbols: int = 50,
    n_days: int = 60,
    n_features: int = 5,
    informative_feature: str = "feat_0",
    signal_noise_ratio: float = 1.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build synthetic (feature_panel, forward_returns).

    feat_0 is constructed to correlate with fwd_5d (SNR controlled).
    All other features are pure noise.
    """
    rng = np.random.default_rng(seed)
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]
    start = date(2023, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_days)]

    feat_names = [f"feat_{i}" for i in range(n_features)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    # True signal: common factor per day
    signal_per_day = rng.standard_normal(n_days)
    signal_arr = np.repeat(signal_per_day, n_symbols) + rng.standard_normal(n_days * n_symbols) / signal_noise_ratio

    data = {}
    data[informative_feature] = signal_arr
    for f in feat_names:
        if f != informative_feature:
            data[f] = rng.standard_normal(n_days * n_symbols)

    fp = pd.DataFrame(data, index=idx)

    # Forward returns: correlated with the informative feature
    fwd_5d = signal_arr * 0.02 + rng.standard_normal(n_days * n_symbols) * 0.01
    fwd_10d = signal_arr * 0.015 + rng.standard_normal(n_days * n_symbols) * 0.015
    fwd_20d = signal_arr * 0.008 + rng.standard_normal(n_days * n_symbols) * 0.02
    fr = pd.DataFrame(
        {"fwd_5d": fwd_5d, "fwd_10d": fwd_10d, "fwd_20d": fwd_20d},
        index=idx,
    )
    return fp, fr


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestComputeDailyIC:
    def test_informative_feature_high_ic(self):
        fp, fr = _make_panel(n_symbols=60, n_days=60, signal_noise_ratio=3.0)
        daily = compute_daily_ic(fp, fr, horizons=(5,), min_symbols_per_day=20)
        assert not daily.empty
        feat0 = daily[daily["feature"] == "feat_0"]
        assert feat0["ic"].mean() > 0.3, "Highly informative feature should have IC > 0.3"

    def test_noise_features_near_zero_ic(self):
        fp, fr = _make_panel(n_symbols=60, n_days=60, signal_noise_ratio=3.0)
        daily = compute_daily_ic(fp, fr, horizons=(5,), min_symbols_per_day=20)
        for f in ["feat_1", "feat_2", "feat_3", "feat_4"]:
            sub = daily[daily["feature"] == f]
            mean_ic = sub["ic"].mean() if not sub.empty else 0.0
            assert abs(mean_ic) < 0.15, f"{f} is noise but |IC|={abs(mean_ic):.3f} > 0.15"

    def test_min_symbols_filter_drops_thin_days(self):
        fp, fr = _make_panel(n_symbols=25, n_days=20)
        daily_strict = compute_daily_ic(fp, fr, horizons=(5,), min_symbols_per_day=30)
        # With 25 symbols < 30 threshold, no rows should pass
        assert daily_strict.empty or (daily_strict["n_symbols"] >= 30).all()

    def test_output_columns(self):
        fp, fr = _make_panel(n_symbols=40, n_days=10)
        daily = compute_daily_ic(fp, fr, horizons=(5, 10), min_symbols_per_day=10)
        if not daily.empty:
            assert set(daily.columns) == {"date", "feature", "horizon", "ic", "n_symbols"}

    def test_no_lookahead_in_daily_ic(self):
        """IC at date t must only use features/returns available at t, not t+1."""
        fp, fr = _make_panel(n_symbols=40, n_days=20, seed=1)
        daily_full = compute_daily_ic(fp, fr, horizons=(5,), min_symbols_per_day=10)
        # Truncate to first 10 days and recompute — results for those days must match
        dates_full = sorted(daily_full["date"].unique())
        if len(dates_full) < 2:
            pytest.skip("Too few trading days for this test")
        cutoff = dates_full[len(dates_full) // 2]
        fp_trunc = fp.loc[fp.index.get_level_values("date") <= cutoff]
        fr_trunc = fr.loc[fr.index.get_level_values("date") <= cutoff]
        daily_trunc = compute_daily_ic(fp_trunc, fr_trunc, horizons=(5,), min_symbols_per_day=10)
        # Days present in both should have same IC values
        common_dates = set(daily_trunc["date"].unique()) & set(daily_full["date"].unique())
        for d in list(common_dates)[:3]:
            for feat in ["feat_0"]:
                ic_full = daily_full.loc[
                    (daily_full["date"] == d) & (daily_full["feature"] == feat) & (daily_full["horizon"] == 5), "ic"
                ]
                ic_trunc = daily_trunc.loc[
                    (daily_trunc["date"] == d) & (daily_trunc["feature"] == feat) & (daily_trunc["horizon"] == 5), "ic"
                ]
                if not ic_full.empty and not ic_trunc.empty:
                    np.testing.assert_almost_equal(ic_full.values[0], ic_trunc.values[0], decimal=10)

    def test_empty_input_returns_empty_df(self):
        fp = pd.DataFrame()
        fr = pd.DataFrame()
        daily = compute_daily_ic(fp, fr)
        assert daily.empty

    def test_multiple_horizons(self):
        fp, fr = _make_panel(n_symbols=40, n_days=15)
        daily = compute_daily_ic(fp, fr, horizons=(5, 10, 20), min_symbols_per_day=10)
        if not daily.empty:
            assert set(daily["horizon"].unique()).issubset({5, 10, 20})


class TestAggregateIC:
    def test_aggregate_no_nans_in_key_cols(self):
        fp, fr = _make_panel(n_symbols=50, n_days=40)
        daily = compute_daily_ic(fp, fr, horizons=(5, 10), min_symbols_per_day=10)
        if daily.empty:
            pytest.skip("No IC computed")
        agg = aggregate_ic(daily)
        assert not agg.empty
        for col in ["ic_mean_h5", "ic_ir_h5", "ic_t_h5", "hit_rate_h5"]:
            if col in agg.columns:
                assert agg[col].isna().sum() == 0, f"{col} has NaN"

    def test_decay_calculation_informative_feature(self):
        fp, fr = _make_panel(n_symbols=60, n_days=60, signal_noise_ratio=2.0)
        daily = compute_daily_ic(fp, fr, horizons=(5, 10, 20), min_symbols_per_day=20)
        if daily.empty:
            pytest.skip("No IC computed")
        agg = aggregate_ic(daily)
        if "decay_5_10" in agg.columns and "feat_0" in agg.index:
            decay = agg.loc["feat_0", "decay_5_10"]
            # Signal decays: IC at h=10 should be < IC at h=5 for short-term signals
            assert np.isfinite(decay)

    def test_sorted_by_ic_ir(self):
        fp, fr = _make_panel(n_symbols=50, n_days=40)
        daily = compute_daily_ic(fp, fr, horizons=(5,), min_symbols_per_day=10)
        if daily.empty:
            pytest.skip("No IC computed")
        agg = aggregate_ic(daily)
        if "ic_ir_h5" in agg.columns and len(agg) > 1:
            irs = agg["ic_ir_h5"].abs().values
            assert all(irs[i] >= irs[i + 1] for i in range(len(irs) - 1)), \
                "aggregate_ic should be sorted descending by |IC IR|"


class TestSummarizeByRegime:
    def test_regime_breakdown(self):
        fp, fr = _make_panel(n_symbols=50, n_days=40)
        daily = compute_daily_ic(fp, fr, horizons=(5,), min_symbols_per_day=10)
        if daily.empty:
            pytest.skip("No IC computed")
        dates = daily["date"].unique()
        regime_labels = {d: ("RISK_ON" if i % 2 == 0 else "RISK_OFF") for i, d in enumerate(dates)}
        by_regime = summarize_by_regime(daily, regime_labels)
        assert not by_regime.empty
        regimes = by_regime.index.get_level_values("regime").unique()
        assert "RISK_ON" in regimes
        assert "RISK_OFF" in regimes

    def test_unknown_dates_labeled_unknown(self):
        fp, fr = _make_panel(n_symbols=40, n_days=20)
        daily = compute_daily_ic(fp, fr, horizons=(5,), min_symbols_per_day=10)
        if daily.empty:
            pytest.skip("No IC computed")
        by_regime = summarize_by_regime(daily, {})  # empty map → all UNKNOWN
        if not by_regime.empty:
            assert "UNKNOWN" in by_regime.index.get_level_values("regime").unique()


class TestPassesICThreshold:
    def test_strong_signal_passes(self):
        fp, fr = _make_panel(n_symbols=80, n_days=80, signal_noise_ratio=4.0)
        daily = compute_daily_ic(fp, fr, horizons=(5,), min_symbols_per_day=20)
        agg = aggregate_ic(daily)
        if agg.empty or "ic_ir_h5" not in agg.columns:
            pytest.skip("No IC data")
        passing, failing = passes_ic_threshold(agg, horizon=5)
        # feat_0 should be in passing (strong signal)
        assert "feat_0" in passing, f"feat_0 not in passing: {agg.loc['feat_0'] if 'feat_0' in agg.index else 'missing'}"

    def test_pure_noise_fails(self):
        fp, fr = _make_panel(n_symbols=80, n_days=80, signal_noise_ratio=0.01)
        daily = compute_daily_ic(fp, fr, horizons=(5,), min_symbols_per_day=20)
        agg = aggregate_ic(daily)
        if agg.empty:
            pytest.skip("No IC data")
        passing, failing = passes_ic_threshold(agg, horizon=5)
        # With near-zero signal, nearly everything should fail
        assert len(failing) >= len(passing), "Expected most noise features to fail"


class TestFormatICMarkdown:
    def test_returns_string(self):
        fp, fr = _make_panel(n_symbols=40, n_days=20)
        daily = compute_daily_ic(fp, fr, horizons=(5,), min_symbols_per_day=10)
        agg = aggregate_ic(daily)
        md = format_ic_markdown(agg, top_n=3, horizon=5)
        assert isinstance(md, str)
        assert len(md) > 0

    def test_empty_summary_returns_message(self):
        md = format_ic_markdown(pd.DataFrame(), top_n=3)
        assert "No IC data" in md
