"""Daily liveness beacon (2026-08-05).

The beacon exists because the on-box dead-man watchdog dies with its host, so a power cut or an
OS reboot emits NOTHING. Its contract is therefore unusual and worth pinning down: it must send
even when its own probes fail, because a beacon that can be silenced by an internal error
reintroduces exactly the silent-failure mode it was built to remove.

All tests are pure-Python — every external dependency (Alpaca, DB, kill switch) is patched.
"""

import asyncio
from unittest.mock import patch

import pytest

from app.notifications.notifier import RATE_LIMITS, VALID_EVENTS, render
from app.orchestrator import AgentOrchestrator
from app.scheduler import AgentScheduler


def _run_beacon(orch):
    """Invoke the beacon, capturing the enqueued payload instead of writing to the queue."""
    captured = {}

    def _fake_enqueue(event_type, payload, dedup_key=None):
        captured["event_type"] = event_type
        captured["payload"] = payload
        return 1

    with patch("app.notifications.notifier.enqueue", _fake_enqueue):
        asyncio.run(orch._send_daily_alive())
    return captured


class TestEventRegistration:
    def test_event_is_registered(self):
        assert "daily_alive" in VALID_EVENTS
        assert RATE_LIMITS["daily_alive"] == 0, "beacon must never be throttled"

    def test_renders_ok_and_degraded_subjects(self):
        subj_ok, body_ok = render("daily_alive", {"date": "2026-08-05", "all_ok": True})
        assert "OK" in subj_ok
        # the "silence is the alert" instruction must survive into the body
        assert "stops arriving" in body_ok

        subj_bad, body_bad = render(
            "daily_alive",
            {"date": "2026-08-05", "all_ok": False, "degraded": ["equity: RuntimeError"]},
        )
        assert "DEGRADED" in subj_bad
        assert "equity: RuntimeError" in body_bad


class TestBeaconSends:
    def setup_method(self):
        self.orch = AgentOrchestrator()
        self.orch._started_at = 0.0

    def test_sends_ok_when_all_probes_succeed(self):
        with patch("app.live_trading.heartbeat.heartbeat_age_seconds", return_value=12.0), \
             patch("app.integrations.alpaca.AlpacaClient") as ac, \
             patch("app.live_trading.reconciliation.reconcile") as rec:
            ac.return_value.get_account.return_value = {"equity": "100000"}
            ac.return_value.get_positions.return_value = [{"symbol": "SPY", "qty": "10"}]
            rec.return_value.status = "MATCH"
            rec.return_value.position_breaks = []
            cap = _run_beacon(self.orch)

        assert cap["event_type"] == "daily_alive"
        assert cap["payload"]["all_ok"] is True
        assert cap["payload"]["degraded"] == []

    def test_reconciliation_break_marks_degraded_but_still_sends(self):
        """A live DB<->broker break must show up as DEGRADED, not as a missing email."""
        with patch("app.live_trading.heartbeat.heartbeat_age_seconds", return_value=12.0), \
             patch("app.integrations.alpaca.AlpacaClient") as ac, \
             patch("app.live_trading.reconciliation.reconcile") as rec:
            ac.return_value.get_account.return_value = {"equity": "100000"}
            ac.return_value.get_positions.return_value = []
            rec.return_value.status = "FAIL_CLOSED"
            rec.return_value.position_breaks = []
            cap = _run_beacon(self.orch)

        assert cap["payload"]["all_ok"] is False
        assert any("reconciliation" in d for d in cap["payload"]["degraded"])

    def test_probe_failure_degrades_rather_than_suppresses(self):
        """THE core contract: a broker outage must not silence the beacon."""
        with patch("app.live_trading.heartbeat.heartbeat_age_seconds", return_value=12.0), \
             patch("app.integrations.alpaca.AlpacaClient",
                   side_effect=RuntimeError("broker unreachable")):
            cap = _run_beacon(self.orch)

        assert cap["event_type"] == "daily_alive", "beacon must still be enqueued"
        assert cap["payload"]["all_ok"] is False
        assert cap["payload"]["equity"] == "unavailable"
        assert any("RuntimeError" in d for d in cap["payload"]["degraded"])

    def test_total_probe_failure_still_sends(self):
        """Even with every probe broken, the email goes out — that is the whole point."""
        with patch("app.live_trading.heartbeat.heartbeat_age_seconds",
                   side_effect=OSError("no heartbeat")), \
             patch("app.integrations.alpaca.AlpacaClient", side_effect=OSError("down")), \
             patch("app.live_trading.reconciliation.db_expected_positions",
                   side_effect=OSError("db down")):
            cap = _run_beacon(self.orch)

        assert cap["event_type"] == "daily_alive"
        assert cap["payload"]["all_ok"] is False
        assert len(cap["payload"]["degraded"]) >= 3


class TestScheduledAllSevenDays:
    def test_beacon_runs_on_weekends(self):
        """A beacon silent on Saturday cannot detect a host that died Friday night."""
        sched = AgentScheduler()
        try:
            sched.start()
            sched.schedule_daily_at_time(
                lambda: None, hour=7, minute=45, job_id="daily_alive_beacon",
                day_of_week="0-6",
            )
            job = sched.scheduler.get_job("daily_alive_beacon")
            assert "0-6" in str(job.trigger)
        finally:
            if sched.scheduler.running:
                sched.stop()

    def test_default_remains_weekdays_only(self):
        """The new day_of_week param must not silently change existing market-hours jobs."""
        sched = AgentScheduler()
        try:
            sched.start()
            sched.schedule_daily_at_time(lambda: None, hour=9, minute=45, job_id="wk")
            assert "0-4" in str(sched.scheduler.get_job("wk").trigger)
        finally:
            if sched.scheduler.running:
                sched.stop()


@pytest.mark.asyncio
async def test_beacon_is_registered_in_orchestrator_jobs():
    orch = AgentOrchestrator()
    with patch("app.scheduler.scheduler.schedule_daily_at_time") as daily, \
         patch("app.scheduler.scheduler.schedule_every_n_minutes"):
        orch._schedule_jobs()
    ids = [c.kwargs.get("job_id") for c in daily.call_args_list]
    assert "daily_alive_beacon" in ids
    call = next(c for c in daily.call_args_list if c.kwargs.get("job_id") == "daily_alive_beacon")
    assert call.kwargs["day_of_week"] == "0-6"
