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

class TestVixTermResolution:
    """Targets the SHIPPING path (`_vix_term_ratio_as_of`). An earlier cut of these tests
    exercised a single-leg `_vix3m_as_of` helper that production no longer calls, so the
    staleness logic that actually ships was largely untested and several "disable the FRED
    backstop" monkeypatches were silent no-ops.

    Every test here stubs `_macro_series` — the ONE seam the shipping path uses. Stubbing
    anything else lets the real, gitignored `data/macro/macro_history.parquet` leak in,
    which passes locally and fails on CI.
    """

    def _s(self, pairs):
        return pd.Series([p[1] for p in pairs],
                         index=pd.to_datetime([p[0] for p in pairs]), name="close")

    def _no_macro(self, monkeypatch):
        import app.ml.regime_features as rf
        monkeypatch.setattr(rf, "_macro_series", lambda field: None)

    def test_fresh_yfinance_pair_is_used(self, monkeypatch):
        import app.ml.regime_features as rf

        self._no_macro(monkeypatch)
        vix = self._s([("2026-09-03", 14.32), ("2026-09-04", 14.53)])
        v3m = self._s([("2026-09-03", 17.42), ("2026-09-04", 17.61)])
        assert rf._vix_term_ratio_as_of(vix, v3m, date(2026, 9, 4)) == pytest.approx(
            round(14.53 / 17.61, 4))

    def test_a_pair_within_the_staleness_bound_is_used(self, monkeypatch):
        """A long weekend must not invalidate Friday's settled closes."""
        import app.ml.regime_features as rf

        self._no_macro(monkeypatch)
        vix = self._s([("2026-09-04", 14.53)])
        v3m = self._s([("2026-09-04", 17.61)])
        assert rf._vix_term_ratio_as_of(vix, v3m, date(2026, 9, 7)) == pytest.approx(
            round(14.53 / 17.61, 4))

    def test_stale_pair_is_rejected_not_carried_forward(self, monkeypatch):
        """THE ORIGINAL BUG: .iloc[-1] returned the 2026-07-17 close for a 2026-09-04
        as_of, giving vix_term_ratio=0.7074 against a truth of 0.8251."""
        import app.ml.regime_features as rf

        self._no_macro(monkeypatch)
        vix = self._s([("2026-07-17", 15.0)])
        v3m = self._s([("2026-07-17", 20.54)])
        assert rf._vix_term_ratio_as_of(vix, v3m, date(2026, 9, 4)) is None

    def test_falls_back_to_the_fred_backed_series(self, monkeypatch):
        import app.ml.regime_features as rf

        macro = {"vix": self._s([("2026-09-04", 14.53)]),
                 "vix3m": self._s([("2026-09-04", 17.61)])}
        monkeypatch.setattr(rf, "_macro_series", lambda f: macro.get(f))
        stale = self._s([("2026-07-17", 20.54)])
        assert rf._vix_term_ratio_as_of(None, stale, date(2026, 9, 4)) == pytest.approx(
            round(14.53 / 17.61, 4))

    def test_macro_fallback_respects_the_same_staleness_bound(self, monkeypatch):
        import app.ml.regime_features as rf

        macro = {"vix": self._s([("2026-07-17", 15.0)]),
                 "vix3m": self._s([("2026-07-17", 20.54)])}
        monkeypatch.setattr(rf, "_macro_series", lambda f: macro.get(f))
        assert rf._vix_term_ratio_as_of(None, None, date(2026, 9, 4)) is None

    def test_no_data_anywhere_returns_none(self, monkeypatch):
        import app.ml.regime_features as rf

        self._no_macro(monkeypatch)
        assert rf._vix_term_ratio_as_of(None, None, date(2026, 9, 4)) is None

    def test_all_nan_series_falls_through_to_macro(self, monkeypatch):
        """The prefetch path returns rows that are present but all-NaN — that is
        'missing', not 'fresh'."""
        import app.ml.regime_features as rf

        macro = {"vix": self._s([("2026-09-04", 14.53)]),
                 "vix3m": self._s([("2026-09-04", 17.61)])}
        monkeypatch.setattr(rf, "_macro_series", lambda f: macro.get(f))
        nan_s = self._s([("2026-09-03", float("nan")), ("2026-09-04", float("nan"))])
        assert rf._vix_term_ratio_as_of(nan_s, nan_s, date(2026, 9, 4)) == pytest.approx(
            round(14.53 / 17.61, 4))

    def test_future_dated_value_is_not_used(self, monkeypatch):
        """A close dated after as_of would be lookahead."""
        import app.ml.regime_features as rf

        self._no_macro(monkeypatch)
        vix = self._s([("2026-09-10", 15.0)])
        v3m = self._s([("2026-09-10", 17.6)])
        assert rf._vix_term_ratio_as_of(vix, v3m, date(2026, 9, 4)) is None


class TestVixTermRatioFeature:
    """The feature itself must go NULL rather than wrong when VIX3M is unavailable."""

    def test_vix_term_ratio_is_null_when_no_fresh_vix3m(self, monkeypatch):
        import app.ml.regime_features as rf

        monkeypatch.setattr(rf, "_macro_series", lambda f: None)
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
        """Stubs `_macro_series`, the seam the shipping path uses. Patching a helper the
        shipping path no longer calls let the real (gitignored) macro parquet leak in —
        green locally, red on CI."""
        import app.ml.regime_features as rf

        _m = {"vix": pd.Series([14.53], index=pd.to_datetime(["2026-09-04"])),
              "vix3m": pd.Series([17.61], index=pd.to_datetime(["2026-09-04"]))}
        monkeypatch.setattr(rf, "_macro_series", lambda f: _m.get(f))
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
        from app.ml import regime_backfill as bf

        captured = {}
        monkeypatch.setattr(bf, "last_backfill_date", lambda: date(2026, 5, 7))

        def _fake_days(start, end):
            captured.setdefault("start", start)
            return []
        monkeypatch.setattr(bf, "trading_days_between", _fake_days)

        bf.extend_backfill(None, date(2026, 9, 5))
        # resumes the day AFTER the last row — not today-30
        assert captured["start"] == date(2026, 5, 8)

    def test_start_none_on_an_empty_table_falls_back_to_a_lookback(self, monkeypatch):
        from app.ml import regime_backfill as bf

        captured = {}
        monkeypatch.setattr(bf, "last_backfill_date", lambda: None)

        def _fake_days(start, end):
            captured.setdefault("start", start)
            return []
        monkeypatch.setattr(bf, "trading_days_between", _fake_days)

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


# ── fourth-review fixes ──────────────────────────────────────────────────────

class TestVixTermRatioIsDatePaired:
    """The two legs must come from the SAME date. Resolving them through independent
    'within 5 days' lookups divides closes from different days — and that is the DEFAULT
    live path, because yfinance ^VIX3M is ~95% NaN so the denominator comes from FRED,
    which lags a business day behind the yfinance numerator."""

    def _s(self, pairs):
        return pd.Series([p[1] for p in pairs],
                         index=pd.to_datetime([p[0] for p in pairs]), name="close")

    def test_same_date_pair_is_used(self, monkeypatch):
        import app.ml.regime_features as rf

        monkeypatch.setattr(rf, "_macro_series", lambda f: None)
        vix = self._s([("2026-09-03", 14.32), ("2026-09-04", 14.53)])
        v3m = self._s([("2026-09-03", 17.42), ("2026-09-04", 17.61)])
        assert rf._vix_term_ratio_as_of(vix, v3m, date(2026, 9, 4)) == pytest.approx(
            round(14.53 / 17.61, 4))

    def test_a_lagging_denominator_does_not_pair_with_todays_numerator(self, monkeypatch):
        """THE BUG: VIX spikes 15 -> 25 while VIX3M is only published through yesterday.
        Unpaired, 25/16 = 1.5625 crosses the 1.05 backwardation threshold and flips the
        rule label to RISK_OFF. Paired, we use the last day carrying BOTH."""
        import app.ml.regime_features as rf

        monkeypatch.setattr(rf, "_macro_series", lambda f: None)
        vix = self._s([("2026-09-03", 15.0), ("2026-09-04", 25.0)])
        v3m = self._s([("2026-09-03", 16.0)])            # denominator lags one day
        ratio = rf._vix_term_ratio_as_of(vix, v3m, date(2026, 9, 4))
        assert ratio == pytest.approx(round(15.0 / 16.0, 4))
        assert ratio < 1.05                              # NOT spurious backwardation

    def test_returns_none_when_no_date_carries_both(self, monkeypatch):
        import app.ml.regime_features as rf

        monkeypatch.setattr(rf, "_macro_series", lambda f: None)
        vix = self._s([("2026-09-04", 15.0)])
        v3m = self._s([("2026-07-17", 20.54)])           # never overlaps in-window
        assert rf._vix_term_ratio_as_of(vix, v3m, date(2026, 9, 4)) is None

    def test_yfinance_still_wins_per_date_over_macro(self, monkeypatch):
        import app.ml.regime_features as rf

        macro = {"vix": self._s([("2026-09-04", 99.0)]),
                 "vix3m": self._s([("2026-09-04", 17.61)])}
        monkeypatch.setattr(rf, "_macro_series", lambda f: macro.get(f))
        vix = self._s([("2026-09-04", 14.53)])
        assert rf._vix_term_ratio_as_of(vix, None, date(2026, 9, 4)) == pytest.approx(
            round(14.53 / 17.61, 4))


class TestSpyStalenessGuard:
    def test_stale_spy_leaves_trend_features_null(self, monkeypatch):
        """spy_ma200_dist and spy_20d_return are in the trainer's core set AND drive
        label_regime_day — a frozen SPY feed must not pin them to their last value."""
        import app.ml.regime_features as rf

        monkeypatch.setattr(rf, "_macro_series", lambda f: None)
        builder = rf.RegimeFeatureBuilder()
        feats = {k: float("nan") for k in rf.REGIME_FEATURE_NAMES}
        idx = pd.date_range(end=pd.Timestamp("2026-07-17"), periods=300, freq="D")
        stale = pd.DataFrame({"close": [500.0] * len(idx)}, index=idx)
        builder._add_spy_features(feats, stale, date(2026, 9, 4))
        assert feats["spy_ma200_dist"] != feats["spy_ma200_dist"]     # NaN
        assert feats["spy_20d_return"] != feats["spy_20d_return"]


class TestBackfillNeverWritesUnusableRows:
    """fetch_all_prefetched returns {} on a batch-download failure; {} still takes the
    prefetched branch and build() yields a full dict of NaN — NOT None. Under a forced
    rewrite that would NULL a fortnight of good rows AND advance last_backfill_date past
    them, making the damage permanent and invisible."""

    def test_empty_prefetch_writes_nothing(self, monkeypatch):
        from app.ml import regime_backfill as rb

        monkeypatch.setattr(rb, "last_backfill_date", lambda: date(2026, 8, 20))
        monkeypatch.setattr(
            rb.RegimeFeatureBuilder, "fetch_all_prefetched",
            staticmethod(lambda *a, **k: {}))
        wrote = []
        out = rb.extend_backfill(None, date(2026, 9, 5), rewrite_recent_days=14,
                                 upsert=lambda *a, **k: wrote.append(1) or True)
        assert wrote == []
        assert out["ok"] == 0 and out["unusable"] == out["days"] > 0

    def test_all_nan_features_are_not_written(self, monkeypatch):
        from app.ml import regime_backfill as rb

        monkeypatch.setattr(rb, "last_backfill_date", lambda: date(2026, 9, 1))
        monkeypatch.setattr(
            rb.RegimeFeatureBuilder, "fetch_all_prefetched",
            staticmethod(lambda *a, **k: {"SPY": object()}))
        monkeypatch.setattr(
            rb.RegimeFeatureBuilder, "build",
            lambda self, **k: {f: float("nan") for f in rb._REQUIRED_FEATURES})
        wrote = []
        out = rb.extend_backfill(None, date(2026, 9, 5),
                                 upsert=lambda *a, **k: wrote.append(1) or True)
        assert wrote == []
        assert out["ok"] == 0 and out["unusable"] > 0

    def test_usable_rows_are_written(self, monkeypatch):
        from app.ml import regime_backfill as rb

        monkeypatch.setattr(rb, "last_backfill_date", lambda: date(2026, 9, 1))
        monkeypatch.setattr(
            rb.RegimeFeatureBuilder, "fetch_all_prefetched",
            staticmethod(lambda *a, **k: {"SPY": object()}))
        monkeypatch.setattr(
            rb.RegimeFeatureBuilder, "build",
            lambda self, **k: {f: 1.0 for f in rb._REQUIRED_FEATURES})
        wrote = []
        out = rb.extend_backfill(None, date(2026, 9, 5),
                                 upsert=lambda *a, **k: wrote.append(1) or True)
        assert len(wrote) == out["ok"] > 0
        assert out["unusable"] == 0
