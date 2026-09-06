"""2026-09-06 — the regime gate could not fail, and the VIX3M feed was silently stale.

Four stacked defects, each of which hid the next:

1. `load_dataset` filters to snapshot_trigger == "backfill"; the backfill stopped on
   2026-05-07, so the weekly retrain re-fit the same 2179 rows for four months.
2. `train`/`_write_model_version` recorded the REQUESTED end (`date.today()`) as
   `train_end`, so every weekly row claimed data it did not have. This hid (1).
3. `_FOLDS` was a hardcoded literal whose last test window ended 2026-04-30, so the
   walk-forward re-scored the same three windows forever — versions v35..v42 all
   recorded wf_auc_mean=0.9563 / wf_auc_min=0.9062 / brier=0.0569, byte-identical.
   The promotion gate was applied to a constant.
4. `RegimeFeatureBuilder` had no FRED backstop for ^VIX3M and no staleness bound, so
   `_slice_to_date(...).iloc[-1]` carried the last available close forward forever.
   Measured: build(2026-09-04) returned vix_term_ratio=0.7074 off the 2026-07-17
   close; the FRED-backed answer is 14.53/17.61 = 0.8251.

These tests pin the behaviour that makes each of those loud instead of silent.
"""
from datetime import date

import pandas as pd
import pytest


# ── (3) the fold schedule must reach the present ──────────────────────────────

class TestBuildFolds:
    def test_fixed_folds_are_unchanged(self):
        """The three historical folds are frozen — that is what keeps the
        version-over-version metric series comparable."""
        from app.ml.regime_training import _FIXED_FOLDS

        assert _FIXED_FOLDS == [
            (date(2018, 1, 1), date(2023, 12, 31), date(2024, 12, 31)),
            (date(2018, 1, 1), date(2024, 12, 31), date(2025, 9, 30)),
            (date(2018, 1, 1), date(2025, 9, 30), date(2026, 4, 30)),
        ]

    def test_rolling_fold_added_when_data_is_current(self):
        from app.ml.regime_training import build_folds, _FIXED_FOLDS

        folds = build_folds(date(2026, 9, 4))
        assert len(folds) == len(_FIXED_FOLDS) + 1
        train_start, train_end, test_end = folds[-1]
        assert train_end == date(2026, 4, 30)
        assert test_end == date(2026, 9, 4)
        # the rolling fold must be genuinely out-of-sample: walk_forward selects the
        # test set with `> train_end`, so the windows cannot overlap
        assert train_end < test_end

    def test_no_rolling_fold_when_data_stops_at_the_boundary(self):
        from app.ml.regime_training import build_folds, _FIXED_FOLDS

        assert len(build_folds(date(2026, 4, 30))) == len(_FIXED_FOLDS)

    def test_no_rolling_fold_when_data_is_stale(self):
        """The v35..v42 situation: data older than the boundary yields the frozen gate."""
        from app.ml.regime_training import build_folds, _FIXED_FOLDS

        assert len(build_folds(date(2026, 1, 15))) == len(_FIXED_FOLDS)

    def test_none_data_end_is_tolerated(self):
        from app.ml.regime_training import build_folds, _FIXED_FOLDS

        assert len(build_folds(None)) == len(_FIXED_FOLDS)

    def test_build_folds_does_not_read_the_clock(self):
        """Regression: the bug was that `train_end` came from date.today() while the
        DATA ended four months earlier. build_folds must depend only on its argument."""
        from app.ml.regime_training import build_folds

        assert build_folds(date(2026, 6, 30))[-1][2] == date(2026, 6, 30)


class TestWalkForwardWarnsWhenGateIsFrozen:
    """A missing rolling fold silently restores the un-failable gate, so it must WARN."""

    def _df(self, last_date: date, n: int = 60):
        """Deliberately too small for ANY fold to fit — this test is about the banner,
        so every fold must bail at the min-rows guard before touching feature columns."""
        dates = pd.date_range(end=pd.Timestamp(last_date), periods=n, freq="D").date
        return pd.DataFrame({"snapshot_date": list(dates)})

    def test_warns_when_dataset_predates_the_rolling_boundary(self, caplog):
        from app.ml.regime_training import RegimeModelTrainer, _ROLLING_FOLD_TRAIN_END

        trainer = RegimeModelTrainer()
        df = self._df(date(2023, 6, 30))
        with caplog.at_level("WARNING"):
            # every fold will bail on insufficient data; we only assert on the banner
            trainer.walk_forward(df)
        assert any("NO ROLLING FOLD" in r.message or "NO ROLLING FOLD" in r.getMessage()
                   for r in caplog.records)
        assert _ROLLING_FOLD_TRAIN_END == date(2026, 4, 30)


# ── (4) VIX3M staleness + FRED backstop ───────────────────────────────────────

class TestVix3mAsOf:
    def _series(self, pairs):
        idx = pd.to_datetime([p[0] for p in pairs])
        return pd.Series([p[1] for p in pairs], index=idx, name="close")

    def test_fresh_yfinance_value_is_used(self):
        from app.ml.regime_features import _vix3m_as_of

        s = self._series([("2026-09-03", 17.4), ("2026-09-04", 17.6)])
        assert _vix3m_as_of(s, date(2026, 9, 4)) == pytest.approx(17.6)

    def test_value_within_the_staleness_bound_is_used(self):
        """A long weekend must not invalidate Friday's settled close."""
        from app.ml.regime_features import _vix3m_as_of

        s = self._series([("2026-09-04", 17.6)])
        assert _vix3m_as_of(s, date(2026, 9, 7)) == pytest.approx(17.6)

    def test_stale_yfinance_value_is_rejected_not_carried_forward(self, monkeypatch):
        """THE BUG: .iloc[-1] returned the 2026-07-17 close for a 2026-09-04 as_of."""
        import app.ml.regime_features as rf

        monkeypatch.setattr(rf, "_macro_vix3m_map", lambda: {})
        s = self._series([("2026-07-16", 20.4), ("2026-07-17", 20.54)])
        assert rf._vix3m_as_of(s, date(2026, 9, 4)) is None

    def test_falls_back_to_macro_history_when_yfinance_is_stale(self, monkeypatch):
        import app.ml.regime_features as rf

        monkeypatch.setattr(rf, "_macro_vix3m_map", lambda: {"2026-09-04": 17.61})
        s = self._series([("2026-07-17", 20.54)])
        assert rf._vix3m_as_of(s, date(2026, 9, 4)) == pytest.approx(17.61)

    def test_macro_fallback_walks_back_to_the_last_settled_close(self, monkeypatch):
        """as_of on a weekend resolves to Friday, not to None."""
        import app.ml.regime_features as rf

        monkeypatch.setattr(rf, "_macro_vix3m_map", lambda: {"2026-09-04": 17.61})
        assert rf._vix3m_as_of(None, date(2026, 9, 6)) == pytest.approx(17.61)

    def test_macro_fallback_respects_the_same_staleness_bound(self, monkeypatch):
        """A stale FRED series must not be carried forward either."""
        import app.ml.regime_features as rf

        monkeypatch.setattr(rf, "_macro_vix3m_map", lambda: {"2026-07-17": 20.54})
        assert rf._vix3m_as_of(None, date(2026, 9, 4)) is None

    def test_yfinance_wins_when_both_are_fresh(self, monkeypatch):
        """yfinance stays PRIMARY — FRED only fills what yfinance is missing."""
        import app.ml.regime_features as rf

        monkeypatch.setattr(rf, "_macro_vix3m_map", lambda: {"2026-09-04": 99.0})
        s = self._series([("2026-09-04", 17.6)])
        assert rf._vix3m_as_of(s, date(2026, 9, 4)) == pytest.approx(17.6)

    def test_all_nan_series_falls_through_to_macro(self, monkeypatch):
        """The prefetch path returns rows present but NaN — that is 'missing', not 'fresh'."""
        import app.ml.regime_features as rf

        monkeypatch.setattr(rf, "_macro_vix3m_map", lambda: {"2026-09-04": 17.61})
        s = self._series([("2026-09-03", float("nan")), ("2026-09-04", float("nan"))])
        assert rf._vix3m_as_of(s, date(2026, 9, 4)) == pytest.approx(17.61)

    def test_no_data_anywhere_returns_none(self, monkeypatch):
        import app.ml.regime_features as rf

        monkeypatch.setattr(rf, "_macro_vix3m_map", lambda: {})
        assert rf._vix3m_as_of(None, date(2026, 9, 4)) is None

    def test_future_dated_value_is_not_used(self, monkeypatch):
        """A close dated after as_of would be lookahead; reject it."""
        import app.ml.regime_features as rf

        monkeypatch.setattr(rf, "_macro_vix3m_map", lambda: {})
        s = self._series([("2026-09-10", 17.6)])
        assert rf._vix3m_as_of(s, date(2026, 9, 4)) is None


class TestVixTermRatioFeature:
    """The feature itself must go NULL rather than wrong when VIX3M is unavailable."""

    def test_vix_term_ratio_is_null_when_no_fresh_vix3m(self, monkeypatch):
        import app.ml.regime_features as rf

        monkeypatch.setattr(rf, "_macro_vix3m_map", lambda: {})
        builder = rf.RegimeFeatureBuilder()
        feats = {k: float("nan") for k in rf.REGIME_FEATURE_NAMES}
        vix_s = pd.Series(
            [15.0] * 60,
            index=pd.date_range(end=pd.Timestamp("2026-09-04"), periods=60, freq="D"),
        )
        stale = pd.Series([20.54], index=pd.to_datetime(["2026-07-17"]))
        builder._add_vix_features(feats, vix_s, stale, date(2026, 9, 4))

        assert feats["vix_level"] == pytest.approx(15.0)
        # NaN, not 15.0/20.54 = 0.73
        assert feats["vix_term_ratio"] != feats["vix_term_ratio"]

    def test_vix_term_ratio_uses_the_fred_backed_value(self, monkeypatch):
        import app.ml.regime_features as rf

        monkeypatch.setattr(rf, "_macro_vix3m_map", lambda: {"2026-09-04": 17.61})
        builder = rf.RegimeFeatureBuilder()
        feats = {k: float("nan") for k in rf.REGIME_FEATURE_NAMES}
        vix_s = pd.Series(
            [14.53] * 60,
            index=pd.date_range(end=pd.Timestamp("2026-09-04"), periods=60, freq="D"),
        )
        stale = pd.Series([20.54], index=pd.to_datetime(["2026-07-17"]))
        builder._add_vix_features(feats, vix_s, stale, date(2026, 9, 4))

        assert feats["vix_term_ratio"] == pytest.approx(0.8251, abs=1e-4)
