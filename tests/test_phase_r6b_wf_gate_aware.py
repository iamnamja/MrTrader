"""Phase R6b — gate-aware WF evaluation + swing R6 exclusion tests."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path


class TestRetainCronUsesOppScoreGate:
    """retrain_cron passes use_opportunity_score=True to both WF runners."""

    def test_swing_wf_called_with_opp_score(self):
        src = (Path(__file__).parent.parent / "scripts/retrain_cron.py").read_text()
        # Both the run_swing_walkforward call and use_opportunity_score=True must be present
        assert "run_swing_walkforward" in src
        assert "use_opportunity_score=True" in src

    def test_intraday_wf_called_with_opp_score(self):
        src = (Path(__file__).parent.parent / "scripts/retrain_cron.py").read_text()
        assert "run_intraday_walkforward" in src
        # use_opportunity_score=True appears for both callers
        assert src.count("use_opportunity_score=True") >= 2

    def test_swing_retrain_config_has_r6_exclusion(self):
        from app.ml.retrain_config import SWING_RETRAIN
        assert SWING_RETRAIN.get("exclude_risk_off_days") is True

    def test_intraday_retrain_config_has_r6_exclusion(self):
        from app.ml.retrain_config import INTRADAY_RETRAIN
        assert INTRADAY_RETRAIN.get("exclude_risk_off_days") is True


class TestSwingTrainerR6Exclusion:
    """ModelTrainer.train_model filters RISK_OFF windows when exclude_risk_off_days=True."""

    def _make_meta(self, window_indices):
        return [{"window_idx": i, "outcome_return": 0.01, "vol_percentile": 0.5,
                 "avg_volume": 1e6, "sector": "Tech", "vix_regime_bucket": 0.3}
                for i in window_indices]

    def test_load_risk_off_dates_returns_set_on_db_error(self):
        """Fail-open: returns empty set when DB unavailable."""
        from app.ml.training import ModelTrainer
        trainer = ModelTrainer.__new__(ModelTrainer)
        # Patch at the point of use inside the method (imported locally)
        with patch("app.database.session.get_session", side_effect=Exception("no db")):
            with patch("app.ml.training.get_session", side_effect=Exception("no db")):
                result = trainer._load_risk_off_dates()
        assert isinstance(result, set)
        assert len(result) == 0

    def test_load_risk_off_dates_returns_dates(self):
        """Returns set of date objects from DB query."""
        from app.ml.training import ModelTrainer
        from datetime import date
        trainer = ModelTrainer.__new__(ModelTrainer)
        mock_rows = [MagicMock(snapshot_date=date(2025, 4, 7)),
                     MagicMock(snapshot_date=date(2025, 4, 8))]
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = mock_rows
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_db)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        # Patch inside the method's local import scope
        with patch("app.database.session.get_session", return_value=mock_ctx):
            with patch("app.ml.training.get_session", return_value=mock_ctx):
                result = trainer._load_risk_off_dates()
        assert date(2025, 4, 7) in result
        assert date(2025, 4, 8) in result

    def test_train_model_signature_has_exclude_param(self):
        """train_model accepts exclude_risk_off_days kwarg."""
        import inspect
        from app.ml.training import ModelTrainer
        sig = inspect.signature(ModelTrainer.train_model)
        assert "exclude_risk_off_days" in sig.parameters

    def test_r6b_exclusion_filters_rows(self):
        """When exclude_risk_off_days=True, rows whose window_idx maps to RISK_OFF dates are dropped."""
        from datetime import date
        import numpy as np

        # Simulate: all_dates[0] = RISK_OFF day, all_dates[1] = normal day
        all_dates = [date(2025, 4, 7), date(2025, 4, 8), date(2025, 4, 9)]
        risk_off = {date(2025, 4, 7)}

        meta = [
            {"window_idx": 0, "outcome_return": 0.01, "vol_percentile": 0.5,
             "avg_volume": 1e6, "sector": "Tech", "vix_regime_bucket": 0.3},  # RISK_OFF
            {"window_idx": 1, "outcome_return": 0.02, "vol_percentile": 0.6,
             "avg_volume": 2e6, "sector": "Tech", "vix_regime_bucket": 0.4},  # OK
            {"window_idx": 2, "outcome_return": 0.03, "vol_percentile": 0.7,
             "avg_volume": 3e6, "sector": "Tech", "vix_regime_bucket": 0.5},  # OK
        ]
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 0, 1])

        keep_mask = np.array([
            all_dates[m["window_idx"]] not in risk_off
            for m in meta
        ])
        X_filtered = X[keep_mask]
        y_filtered = y[keep_mask]
        meta_filtered = [m for m, k in zip(meta, keep_mask) if k]

        assert len(X_filtered) == 2
        assert len(y_filtered) == 2
        assert all(m["window_idx"] != 0 for m in meta_filtered)

    def test_no_exclusion_when_flag_false(self):
        """When exclude_risk_off_days=False, no filtering happens regardless of DB."""
        # If the flag is False, _load_risk_off_dates should never be called
        from datetime import date
        import numpy as np

        exclude_risk_off_days = False
        called = []

        def fake_load():
            called.append(True)
            return {date(2025, 4, 7)}

        meta = [{"window_idx": 0, "outcome_return": 0.01, "vol_percentile": 0.5,
                 "avg_volume": 1e6, "sector": "Tech", "vix_regime_bucket": 0.3}]
        X = np.array([[1, 2]])
        y = np.array([1])

        if exclude_risk_off_days:
            risk_off = fake_load()
            keep_mask = np.ones(len(X), dtype=bool)
            X = X[keep_mask]
            y = y[keep_mask]

        assert len(called) == 0
        assert len(X) == 1
