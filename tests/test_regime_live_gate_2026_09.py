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
from datetime import date, timedelta

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
        from app.ml.regime_training import (
            build_folds, _FIXED_FOLDS, ROLLING_TEST_WINDOW_DAYS)
        from datetime import timedelta

        folds = build_folds(date(2026, 9, 4))
        assert len(folds) == len(_FIXED_FOLDS) + 1
        _, train_end, test_end = folds[-1]
        assert test_end == date(2026, 9, 4)
        assert train_end == date(2026, 9, 4) - timedelta(days=ROLLING_TEST_WINDOW_DAYS)
        # genuinely out-of-sample: walk_forward selects the test set with `> train_end`
        assert train_end < test_end

    def test_rolling_fold_train_end_rolls_too(self):
        """REGRESSION: the first cut pinned the rolling TRAIN end at 2026-04-30 and let
        only the test end advance, which recreates the frozen-window defect on a slower
        clock — by 2027 it would score a year-stale fit against a year of unseen data."""
        from app.ml.regime_training import build_folds

        a_train_end = build_folds(date(2026, 9, 4))[-1][1]
        b_train_end = build_folds(date(2027, 9, 4))[-1][1]
        assert b_train_end > a_train_end
        assert (b_train_end - a_train_end).days == 365

    def test_rolling_test_window_length_is_stable_over_time(self):
        from app.ml.regime_training import build_folds

        for d in (date(2026, 9, 4), date(2027, 9, 4), date(2030, 1, 15)):
            _, train_end, test_end = build_folds(d)[-1]
            assert (test_end - train_end).days == 120

    def test_no_rolling_fold_without_enough_training_history(self):
        """A rolling fold whose train window is shorter than fold 1's is refused."""
        from app.ml.regime_training import build_folds, _FIXED_FOLDS

        assert len(build_folds(date(2024, 3, 1))) == len(_FIXED_FOLDS)

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

    def test_warns_when_dataset_cannot_support_a_rolling_fold(self, caplog):
        from app.ml.regime_training import RegimeModelTrainer

        trainer = RegimeModelTrainer()
        df = self._df(date(2023, 6, 30))
        with caplog.at_level("WARNING"):
            # every fold will bail on insufficient data; we only assert on the banner
            trainer.walk_forward(df)
        assert any("NO ROLLING FOLD" in r.getMessage() for r in caplog.records)


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
        stale = self._series([("2026-07-17", 20.54)])
        assert rf._vix3m_as_of(stale, date(2026, 9, 6)) == pytest.approx(17.61)

    def test_macro_fallback_respects_the_same_staleness_bound(self, monkeypatch):
        """A stale FRED series must not be carried forward either."""
        import app.ml.regime_features as rf

        monkeypatch.setattr(rf, "_macro_vix3m_map", lambda: {"2026-07-17": 20.54})
        stale = self._series([("2026-07-17", 20.54)])
        assert rf._vix3m_as_of(stale, date(2026, 9, 4)) is None

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

    def test_none_or_empty_primary_still_reaches_the_fred_backstop(self, monkeypatch):
        """`_fetch_single` returns None on ANY yfinance failure, so refusing the fallback
        for a None/empty primary would disable the backstop in exactly the outage it
        exists for. There is no look-ahead cost: the fallback is probed by as_of_date."""
        import app.ml.regime_features as rf

        monkeypatch.setattr(rf, "_macro_vix3m_map", lambda: {"2026-09-04": 17.61})
        assert rf._vix3m_as_of(None, date(2026, 9, 4)) == pytest.approx(17.61)
        assert rf._vix3m_as_of(pd.Series([], dtype=float),
                               date(2026, 9, 4)) == pytest.approx(17.61)

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


# ── (1) NaN-blind coalescing in the label rule (found in review) ──────────────

class TestLabelRuleIsNanSafe:
    """`row.get(k) or default` is NaN-blind: NaN is truthy, so `NaN or 1.0` is NaN and
    every later comparison silently becomes False. Latent until `_vix3m_as_of` started
    (correctly) returning None on a dead feed — at which point a VIX3M outage would have
    force-labelled every affected day RISK_CAUTION."""

    def _row(self, **kw):
        base = {
            "vix_level": 18.0, "vix_pct_1y": 0.40, "vix_term_ratio": 0.95,
            "spy_ma50_dist": 0.03, "spy_ma200_dist": 0.05,
            "credit_hyg_ief_20d": 0.002, "breadth_rsp_spy_ratio_20d": 0.01,
            "spy_20d_return": 0.03,
        }
        base.update(kw)
        return base

    def test_nan_behaves_like_none_not_like_a_value(self):
        from app.ml.regime_features import label_regime_day

        nan = float("nan")
        assert label_regime_day(self._row(vix_term_ratio=None)) == 2
        # THE BUG: this returned 1 (RISK_CAUTION) because NaN <= 1.0 is False
        assert label_regime_day(self._row(vix_term_ratio=nan)) == 2
        assert label_regime_day(self._row(vix_term_ratio=0.95)) == 2

    @pytest.mark.parametrize("field", [
        "vix_level", "vix_pct_1y", "vix_term_ratio", "spy_ma50_dist",
        "spy_ma200_dist", "credit_hyg_ief_20d", "breadth_rsp_spy_ratio_20d",
        "spy_20d_return",
    ])
    def test_every_coalesced_field_is_nan_safe(self, field):
        """All eight shared the same idiom, so all eight are pinned."""
        from app.ml.regime_features import label_regime_day

        assert (label_regime_day(self._row(**{field: float("nan")}))
                == label_regime_day(self._row(**{field: None})))

    def test_real_backwardation_still_triggers_risk_off(self):
        """The NaN fix must not blunt the signal it was masking."""
        from app.ml.regime_features import label_regime_day

        assert label_regime_day(self._row(vix_pct_1y=0.90, vix_term_ratio=1.10)) == 0

    def test_coalesce_handles_unparseable_values(self):
        from app.ml.regime_features import _coalesce

        assert _coalesce({"x": "abc"}, "x", 1.0) == 1.0
        assert _coalesce({}, "x", 1.0) == 1.0
        assert _coalesce({"x": None}, "x", 1.0) == 1.0
        assert _coalesce({"x": float("nan")}, "x", 1.0) == 1.0
        assert _coalesce({"x": "2.5"}, "x", 1.0) == 2.5


# ── (6) the VIX numerator needs the same guard as the denominator ─────────────

class TestVixNumeratorStaleness:
    def test_stale_vix_yields_null_features_not_a_stale_ratio(self, monkeypatch):
        """Guarding only VIX3M would still let a weeks-old VIX pair with a fresh
        denominator — the same wrong number, merely relocated."""
        import app.ml.regime_features as rf

        monkeypatch.setattr(rf, "_macro_series", lambda field: None)
        monkeypatch.setattr(rf, "_macro_vix3m_map", lambda: {"2026-09-04": 17.61})
        builder = rf.RegimeFeatureBuilder()
        feats = {k: float("nan") for k in rf.REGIME_FEATURE_NAMES}
        stale_vix = pd.Series(
            [15.0] * 60,
            index=pd.date_range(end=pd.Timestamp("2026-07-17"), periods=60, freq="D"),
        )
        builder._add_vix_features(feats, stale_vix, None, date(2026, 9, 4))

        assert feats["vix_level"] != feats["vix_level"]        # NaN
        assert feats["vix_term_ratio"] != feats["vix_term_ratio"]

    def test_fresh_vix_still_populates(self, monkeypatch):
        import app.ml.regime_features as rf

        monkeypatch.setattr(rf, "_macro_series", lambda field: None)
        monkeypatch.setattr(rf, "_macro_vix3m_map", lambda: {})
        builder = rf.RegimeFeatureBuilder()
        feats = {k: float("nan") for k in rf.REGIME_FEATURE_NAMES}
        vix = pd.Series(
            [15.0] * 60,
            index=pd.date_range(end=pd.Timestamp("2026-09-04"), periods=60, freq="D"),
        )
        builder._add_vix_features(feats, vix, None, date(2026, 9, 4))
        assert feats["vix_level"] == pytest.approx(15.0)


# ── (2) a rolling fold that is BUILT but SKIPPED must not pass silently ───────

class TestRegimeGateRequiresRecentEvidence:
    """The `NO ROLLING FOLD` banner keys on the fold COUNT, so it cannot fire when a
    rolling fold is built and then dropped by the min-rows guard. The gate closes that
    hole: no rolling evidence => FAIL, whatever the reason."""

    def _base(self, **kw):
        d = {"wf_auc_min": 0.9062, "wf_log_loss_mean": 0.0569,
             "rolling_log_loss": 0.0001}
        d.update(kw)
        return d

    def test_passes_with_recent_evidence(self):
        from app.ml.regime_training import regime_gate

        assert regime_gate(self._base()) == (True, [])

    def test_missing_rolling_key_fails(self):
        from app.ml.regime_training import regime_gate

        payload = self._base()
        del payload["rolling_log_loss"]
        ok, failures = regime_gate(payload)
        assert not ok
        assert any("rolling fold not evaluated" in f for f in failures)

    def test_none_rolling_fails(self):
        from app.ml.regime_training import regime_gate

        ok, failures = regime_gate(self._base(rolling_log_loss=None))
        assert not ok
        assert any("rolling fold not evaluated" in f for f in failures)

    def test_nan_rolling_fails(self):
        from app.ml.regime_training import regime_gate

        ok, _ = regime_gate(self._base(rolling_log_loss=float("nan")))
        assert not ok

    def test_bad_rolling_log_loss_fails_even_when_fixed_folds_pass(self):
        """The whole point: the fixed folds always pass, so only this term can fail."""
        from app.ml.regime_training import regime_gate

        ok, failures = regime_gate(self._base(rolling_log_loss=0.9))
        assert not ok
        assert any("rolling_log_loss" in f for f in failures)

    def test_the_v42_payload_shape_no_longer_passes(self):
        """v35..v42 carried no rolling term. Re-evaluating one must not read as healthy."""
        from app.ml.regime_training import regime_gate

        ok, _ = regime_gate({"wf_auc_min": 0.9062, "wf_log_loss_mean": 0.0569})
        assert not ok


# ── second-review fixes ──────────────────────────────────────────────────────

class TestStalenessIsAGateFailure:
    """The rolling fold derives from the DATA's end, so a dataset that stops advancing
    yields a rolling fold that also stops advancing — the un-failable gate returning one
    level down. Only `requested_end` is calendar-anchored, so it is what detects a freeze."""

    def _p(self, **kw):
        d = {"wf_auc_min": 0.9062, "wf_log_loss_mean": 0.0569,
             "rolling_log_loss": 0.0001,
             "train_end": "2026-09-04", "requested_end": "2026-09-06"}
        d.update(kw)
        return d

    def test_current_data_passes(self):
        from app.ml.regime_training import regime_gate

        assert regime_gate(self._p()) == (True, [])

    def test_the_original_defect_now_fails_the_gate(self):
        """Exactly the v35..v42 condition: data ends 2026-05-07, retrain asks for today."""
        from app.ml.regime_training import regime_gate

        ok, failures = regime_gate(
            self._p(train_end="2026-05-07", requested_end="2026-09-06"))
        assert not ok
        assert any("stale" in f for f in failures)

    def test_a_holiday_week_of_lag_is_tolerated(self):
        from app.ml.regime_training import regime_gate

        ok, _ = regime_gate(self._p(train_end="2026-08-30", requested_end="2026-09-06"))
        assert ok

    def test_unparseable_dates_fail_rather_than_raise(self):
        from app.ml.regime_training import regime_gate

        ok, failures = regime_gate(self._p(train_end="not-a-date"))
        assert not ok
        assert any("unparseable" in f for f in failures)


class TestRegistryAndPickleAgree:
    """`_write_model_version` recomputing wf_auc_min over ALL folds would write a different
    value into the registry than the pickle carries — contaminating the very series the
    fixed folds exist to preserve."""

    def test_write_model_version_takes_f1_min_rather_than_recomputing(self):
        import inspect
        from app.ml.regime_training import RegimeModelTrainer

        sig = inspect.signature(RegimeModelTrainer._write_model_version)
        assert "f1_min" in sig.parameters
        src = inspect.getsource(RegimeModelTrainer._write_model_version)
        assert "wf_auc_min=f1_min" in src
        assert "for r in fold_results" not in src.split("wf_auc_min")[1][:200]


class TestEmptyDatasetIsDiagnosed:
    def test_all_rows_dropped_raises_a_named_error_not_a_bare_max(self, monkeypatch):
        """train() does max(df['snapshot_date']); an all-dropped frame would die there with
        'max() arg is an empty sequence', naming neither cause nor subsystem. Newly
        reachable because the staleness guard NULLs vix_level instead of carrying a stale
        value forward."""
        import contextlib
        import app.database.session as sess
        import app.ml.regime_training as rt

        class _Row:
            snapshot_date = date(2026, 9, 4)

            def __getattr__(self, name):
                return float("nan")     # every feature missing -> every row dropped

        class _Q:
            def filter(self, *a, **k):
                return self

            def order_by(self, *a, **k):
                return self

            def all(self):
                return [_Row(), _Row()]

        class _S:
            def query(self, *a, **k):
                return _Q()

        @contextlib.contextmanager
        def _fake_session():
            yield _S()

        monkeypatch.setattr(sess, "get_session", _fake_session)
        monkeypatch.setattr(sess, "init_db", lambda *a, **k: None)

        with pytest.raises(ValueError, match="dropped for missing core features"):
            rt.RegimeModelTrainer().load_dataset(date(2018, 1, 1), date(2026, 9, 6))


# ── third-review fixes ───────────────────────────────────────────────────────

class TestBackfillResumesFromTheLastRow:
    """A fixed lookback cannot close a gap longer than itself: it writes the recent tail,
    leaves the older hole empty forever, and drags max(snapshot_date) up to today so the
    staleness gate reads CURRENT and passes over the hole. A false green is worse than the
    stale red it replaced."""

    def test_start_none_resumes_from_the_last_existing_row(self, monkeypatch):
        import scripts.backfill_regime_snapshots as bf

        captured = {}
        monkeypatch.setattr(bf, "last_backfill_date", lambda: date(2026, 5, 7))

        def _fake_days(start, end):
            captured["start"] = start
            return []
        monkeypatch.setattr(bf, "_trading_days_between", _fake_days)

        bf.extend_backfill(None, date(2026, 9, 5))
        # resumes the day AFTER the last row — not today-30
        assert captured["start"] == date(2026, 5, 8)

    def test_start_none_on_an_empty_table_falls_back_to_a_lookback(self, monkeypatch):
        import scripts.backfill_regime_snapshots as bf

        captured = {}
        monkeypatch.setattr(bf, "last_backfill_date", lambda: None)

        def _fake_days(start, end):
            captured["start"] = start
            return []
        monkeypatch.setattr(bf, "_trading_days_between", _fake_days)

        bf.extend_backfill(None, date(2026, 9, 5))
        assert captured["start"] == date(2026, 9, 5) - timedelta(days=30)


class TestRollingFoldMinimumBinds:
    def test_minimum_is_proportionate_to_the_window(self):
        """20 was both a no-op (it equalled the generic per-fold minimum) and ~4x below
        the ~82 trading days a 120-calendar-day window holds."""
        from app.ml.regime_training import (
            _MIN_ROLLING_TEST_ROWS, ROLLING_TEST_WINDOW_DAYS)

        assert _MIN_ROLLING_TEST_ROWS > 20
        approx_trading_days = ROLLING_TEST_WINDOW_DAYS * 5 / 7
        assert _MIN_ROLLING_TEST_ROWS > 0.5 * approx_trading_days

    def test_a_sparse_rolling_window_blocks_promotion(self):
        """Dropping the fold makes rolling_log_loss None, which FAILS the gate — so a hole
        in the recent snapshots cannot be certified over."""
        from app.ml.regime_training import regime_gate

        ok, failures = regime_gate({
            "wf_auc_min": 0.9062, "wf_log_loss_mean": 0.0569,
            "rolling_log_loss": None,
            "train_end": "2026-09-04", "requested_end": "2026-09-06",
        })
        assert not ok
        assert any("rolling fold not evaluated" in f for f in failures)


class TestLiveScoringFeatureContract:
    def test_nis_features_are_imputed_exactly_as_the_trainer_does(self):
        import inspect
        from app.ml.regime_model import RegimeModel

        src = inspect.getsource(RegimeModel.score)
        assert '("nis_risk_numeric", 0.5)' in src
        assert '("nis_sizing_factor", 1.0)' in src

    def test_missing_vix_level_falls_back_to_neutral(self):
        """vix_level is in the trainer's core dropna set, so the model never saw it
        missing and has no learned branch — scoring anyway is arbitrary-but-confident."""
        import inspect
        from app.ml.regime_model import RegimeModel

        src = inspect.getsource(RegimeModel.score)
        assert "_vix_level" in src and "_legacy_fallback" in src
