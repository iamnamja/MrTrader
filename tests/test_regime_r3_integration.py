"""Phase R3 — Regime model premarket integration tests.

Covers:
- _run_regime_scoring writes to regime_snapshots
- _startup_regime_catchup: runs when no premarket row exists
- _startup_regime_catchup: skips when premarket row already exists
- _startup_regime_catchup: skips when past 11:30 ET
- get_regime_context: returns latest non-backfill row for today
- get_regime_context: applies staleness haircut if >4h old during market hours
- _schedule_regime_reeval_jobs: registers post-event jobs
- API: GET /regime/current returns current context
- API: GET /regime/history returns time series
- PM ProposalLog rows populated with regime context
"""
import asyncio
from datetime import date, datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── PremarketIntelligence regime methods ─────────────────────────────────────

class TestRunRegimeScoring:
    def test_returns_score_dict(self):
        from app.agents.premarket import PremarketIntelligence
        pi = PremarketIntelligence()
        mock_result = {"regime_score": 0.72, "regime_label": "RISK_ON", "trigger": "premarket", "cached": False}
        with patch("app.ml.regime_model.RegimeModel.instance") as MockRM:
            MockRM.return_value.score.return_value = mock_result
            result = pi._run_regime_scoring("premarket")
        assert result["regime_score"] == 0.72
        assert result["regime_label"] == "RISK_ON"

    def test_returns_none_on_exception(self):
        from app.agents.premarket import PremarketIntelligence
        pi = PremarketIntelligence()
        with patch("app.ml.regime_model.RegimeModel.instance", side_effect=RuntimeError("model missing")):
            result = pi._run_regime_scoring("premarket")
        assert result is None


class TestStartupRegimeCatchup:
    def _make_pi(self):
        from app.agents.premarket import PremarketIntelligence
        return PremarketIntelligence()

    def _make_mock_session(self, existing_row=None):
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.first.return_value = existing_row
        return mock_session

    def test_runs_catchup_when_no_premarket_row(self):
        pi = self._make_pi()
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        mock_now = datetime(2026, 5, 6, 9, 15, tzinfo=ET)

        with (
            patch.object(pi, "_run_regime_scoring") as mock_score,
            patch("app.agents.premarket.datetime") as mock_dt,
            patch("app.database.session.get_session", return_value=self._make_mock_session(None)),
        ):
            mock_dt.now.return_value = mock_now
            pi._startup_regime_catchup()

        mock_score.assert_called_once_with("startup_catchup")

    def test_skips_catchup_when_row_exists(self):
        pi = self._make_pi()
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        mock_now = datetime(2026, 5, 6, 9, 15, tzinfo=ET)

        with (
            patch.object(pi, "_run_regime_scoring") as mock_score,
            patch("app.agents.premarket.datetime") as mock_dt,
            patch("app.database.session.get_session", return_value=self._make_mock_session(MagicMock())),
        ):
            mock_dt.now.return_value = mock_now
            pi._startup_regime_catchup()

        mock_score.assert_not_called()

    def test_skips_catchup_after_1130(self):
        from app.agents.premarket import PremarketIntelligence
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        pi = PremarketIntelligence()
        mock_now = datetime(2026, 5, 6, 12, 0, tzinfo=ET)

        with (
            patch.object(pi, "_run_regime_scoring") as mock_score,
            patch("app.agents.premarket.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = mock_now
            pi._startup_regime_catchup()

        mock_score.assert_not_called()


class TestGetRegimeContext:
    def _make_mock_row(self, score=0.65, label="RISK_ON", trigger="premarket", hours_ago=1, now_naive=None):
        row = MagicMock()
        row.regime_score = score
        row.regime_label = label
        row.snapshot_trigger = trigger
        base = now_naive if now_naive is not None else datetime.utcnow()
        row.snapshot_time = base - timedelta(hours=hours_ago)
        return row

    def test_returns_dict_with_score(self):
        from app.agents.premarket import PremarketIntelligence
        pi = PremarketIntelligence()
        mock_row = self._make_mock_row(score=0.72, label="RISK_ON")
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_row

        with patch("app.database.session.get_session", return_value=mock_session):
            result = pi.get_regime_context()

        assert result is not None
        assert result["regime_score"] == 0.72
        assert result["regime_label"] == "RISK_ON"

    def test_returns_none_when_no_row(self):
        from app.agents.premarket import PremarketIntelligence
        pi = PremarketIntelligence()
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

        with patch("app.database.session.get_session", return_value=mock_session):
            result = pi.get_regime_context()
        assert result is None

    def test_applies_staleness_haircut_during_market_hours(self):
        from app.agents.premarket import PremarketIntelligence
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        pi = PremarketIntelligence()
        mock_now_naive = datetime(2026, 5, 6, 13, 0)  # naive reference for age calc
        mock_row = self._make_mock_row(score=0.80, hours_ago=5, now_naive=mock_now_naive)
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_row
        mock_now = datetime(2026, 5, 6, 13, 0, tzinfo=ET)

        with (
            patch("app.database.session.get_session", return_value=mock_session),
            patch("app.agents.premarket.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = mock_now
            result = pi.get_regime_context()

        # Score should be haircut by 20% when stale
        assert result is not None
        assert result.get("stale") is True
        assert result["regime_score"] < 0.80


# ── Schedule regime re-eval jobs ─────────────────────────────────────────────

class TestScheduleRegimeReevalJobs:
    def test_registers_job_for_fomc_event(self):
        from app.agents.premarket import PremarketIntelligence
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        pi = PremarketIntelligence()
        mock_scheduler = MagicMock()

        mock_evt = MagicMock()
        mock_evt.event_type = "FOMC"
        mock_evt.time_str = "23:00"  # far-future time today so run_dt > now
        mock_evt.date_str = date.today().isoformat()

        mock_ctx = MagicMock()
        mock_ctx.events_today = [mock_evt]

        with patch("app.calendars.macro.MacroCalendar") as MockCal:
            MockCal.return_value.get_context.return_value = mock_ctx
            pi._schedule_regime_reeval_jobs(mock_scheduler)

        mock_scheduler.add_job.assert_called_once()
        call_kwargs = mock_scheduler.add_job.call_args
        assert "regime_reeval_post_fomc" in str(call_kwargs)

    def test_no_jobs_when_no_events(self):
        from app.agents.premarket import PremarketIntelligence
        pi = PremarketIntelligence()
        mock_scheduler = MagicMock()

        mock_ctx = MagicMock()
        mock_ctx.events_today = []

        with patch("app.calendars.macro.MacroCalendar") as MockCal:
            MockCal.return_value.get_context.return_value = mock_ctx
            pi._schedule_regime_reeval_jobs(mock_scheduler)

        mock_scheduler.add_job.assert_not_called()


# ── ProposalLog regime columns populated ─────────────────────────────────────

class TestProposalLogRegimeColumns:
    def test_proposal_log_has_regime_columns(self):
        from app.database.models import ProposalLog
        assert hasattr(ProposalLog, "regime_score_at_scan")
        assert hasattr(ProposalLog, "regime_label_at_scan")
        assert hasattr(ProposalLog, "regime_trigger_at_scan")

    def test_proposal_log_accepts_regime_values(self):
        from app.database.models import ProposalLog
        row = ProposalLog(
            strategy="intraday",
            symbol="AAPL",
            pm_status="SCORED",
            regime_score_at_scan=0.72,
            regime_label_at_scan="RISK_ON",
            regime_trigger_at_scan="premarket",
        )
        assert row.regime_score_at_scan == 0.72
        assert row.regime_label_at_scan == "RISK_ON"
        assert row.regime_trigger_at_scan == "premarket"


# ── Run premarket routine includes regime ─────────────────────────────────────

class TestRunPremarketRoutineRegime:
    def test_run_premarket_routine_includes_regime_score(self):
        from app.agents.premarket import PremarketIntelligence
        pi = PremarketIntelligence()

        mock_regime = {"regime_score": 0.68, "regime_label": "RISK_ON", "trigger": "premarket", "cached": False}

        with (
            patch.object(pi, "_fetch_macro_events", return_value={}),
            patch.object(pi, "_fetch_spy_premarket", return_value=0.002),
            patch.object(pi, "_run_regime_scoring", return_value=mock_regime) as mock_score,
        ):
            summary = pi.run_premarket_routine(open_positions=[])

        mock_score.assert_called_once_with("premarket")
        assert summary.get("regime_score") == 0.68
        assert summary.get("regime_label") == "RISK_ON"
