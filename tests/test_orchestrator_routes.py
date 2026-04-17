"""
Tests for the orchestrator API routes — market status + session log.
"""
import pytz
from datetime import datetime
from unittest.mock import patch
from app.api.orchestrator_routes import _market_status, _log, _session_log

_ET = pytz.timezone("America/New_York")

# April 2026: Mon=13, Tue=14, Wed=15, Thu=16(today), Fri=17, Sat=18, Sun=19, Mon=20
def _et(month_day: int, hour: int, minute: int = 0) -> datetime:
    return _ET.localize(datetime(2026, 4, month_day, hour, minute, 0))


MODULE = "app.api.orchestrator_routes"


class TestMarketStatus:

    def test_open_during_hours(self):
        with patch(f"{MODULE}._now_et", return_value=_et(15, 10, 0)):  # Wed 10:00
            r = _market_status()
        assert r["is_open"] is True
        assert r["next_event"]["event"] == "market_close"
        assert r["next_event"]["minutes"] == (16 - 10) * 60  # 360 min

    def test_closed_before_open(self):
        with patch(f"{MODULE}._now_et", return_value=_et(15, 8, 0)):  # Wed 08:00
            r = _market_status()
        assert r["is_open"] is False
        assert r["next_event"]["event"] == "market_open"

    def test_closed_after_close(self):
        with patch(f"{MODULE}._now_et", return_value=_et(15, 17, 0)):  # Wed 17:00
            r = _market_status()
        assert r["is_open"] is False
        assert r["next_event"]["event"] == "market_open"

    def test_closed_saturday(self):
        with patch(f"{MODULE}._now_et", return_value=_et(18, 11, 0)):  # Sat Apr 18
            r = _market_status()
        assert r["is_open"] is False
        assert r["weekday"] == "Saturday"

    def test_closed_sunday(self):
        with patch(f"{MODULE}._now_et", return_value=_et(19, 11, 0)):  # Sun Apr 19
            r = _market_status()
        assert r["is_open"] is False
        assert r["weekday"] == "Sunday"

    def test_open_at_930_exactly(self):
        with patch(f"{MODULE}._now_et", return_value=_et(13, 9, 30)):  # Mon 09:30
            r = _market_status()
        assert r["is_open"] is True

    def test_closed_at_1600_exactly(self):
        with patch(f"{MODULE}._now_et", return_value=_et(13, 16, 0)):  # Mon 16:00
            r = _market_status()
        assert r["is_open"] is False

    def test_next_open_skips_weekend_from_friday_pm(self):
        with patch(f"{MODULE}._now_et", return_value=_et(17, 17, 0)):  # Fri Apr 17 pm
            r = _market_status()
        assert r["is_open"] is False
        # Next open should be Monday Apr 20
        assert r["next_event"]["date"] == "2026-04-20"

    def test_result_keys(self):
        r = _market_status()
        for key in ("is_open", "current_time_et", "weekday", "next_event"):
            assert key in r

    def test_minutes_to_close_correct(self):
        with patch(f"{MODULE}._now_et", return_value=_et(15, 15, 0)):  # Wed 15:00
            r = _market_status()
        assert r["next_event"]["minutes"] == 60  # 1h until close


class TestSessionLog:
    def setup_method(self):
        _session_log.clear()

    def test_log_adds_entry(self):
        _log("INFO", "Test message")
        assert len(_session_log) == 1
        assert _session_log[0]["message"] == "Test message"
        assert _session_log[0]["level"] == "INFO"

    def test_log_newest_first(self):
        _log("INFO", "first")
        _log("INFO", "second")
        assert _session_log[0]["message"] == "second"
        assert _session_log[1]["message"] == "first"

    def test_log_has_timestamp(self):
        _log("WARNING", "msg")
        assert "timestamp" in _session_log[0]
        assert len(_session_log[0]["timestamp"]) > 10

    def test_log_with_detail(self):
        _log("ERROR", "failed", {"reason": "timeout"})
        assert _session_log[0]["detail"]["reason"] == "timeout"

    def test_log_caps_at_200(self):
        for i in range(210):
            _log("INFO", f"msg {i}")
        assert len(_session_log) == 200

    def test_log_levels(self):
        for level in ("INFO", "WARNING", "ERROR"):
            _log(level, f"msg at {level}")
        levels = {e["level"] for e in _session_log}
        assert levels == {"INFO", "WARNING", "ERROR"}
