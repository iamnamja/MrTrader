"""
Unit tests for Phase 20: Pre-market Intelligence & Event Monitoring.

Covers:
- PremarketIntelligence.is_intraday_blocked() — SPY gap, FOMC, NFP
- PremarketIntelligence.intraday_sizing_factor() — halving on adverse SPY
- Macro event detection: FOMC hardcoded dates, NFP first-Friday heuristic
- Overnight gap analysis: OK / REEVAL / AUTO_EXIT thresholds
- 8-K monitor: EDGAR poll throttling (respects EDGAR_POLL_INTERVAL)

All tests are pure-Python — no network calls.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

ET = ZoneInfo("America/New_York")


def _fresh_pm():
    """Return a fresh PremarketIntelligence with cleared state."""
    from app.agents.premarket import PremarketIntelligence
    pm = PremarketIntelligence()
    return pm


# ─── SPY pre-market context ───────────────────────────────────────────────────

class TestSpyPremarket:
    def test_no_block_on_flat_spy(self):
        pm = _fresh_pm()
        pm._spy_premarket_pct = 0.0
        assert not pm.is_intraday_blocked()
        assert pm.intraday_sizing_factor() == 1.0

    def test_halve_on_moderate_drop(self):
        pm = _fresh_pm()
        pm._spy_premarket_pct = -0.016  # -1.6% (between -1.5% and -2.5%)
        assert not pm.is_intraday_blocked()
        assert pm.intraday_sizing_factor() == 0.5

    def test_block_on_large_drop(self):
        pm = _fresh_pm()
        pm._spy_premarket_pct = -0.03  # -3%
        assert pm.is_intraday_blocked()

    def test_no_halve_on_slight_drop(self):
        pm = _fresh_pm()
        pm._spy_premarket_pct = -0.005  # -0.5% — below threshold
        assert not pm.is_intraday_blocked()
        assert pm.intraday_sizing_factor() == 1.0


# ─── FOMC gate ────────────────────────────────────────────────────────────────

class TestFomcGate:
    def test_fomc_blocks_intraday(self):
        pm = _fresh_pm()
        pm._macro_flags = {"FOMC": {"name": "FOMC Meeting", "time": "14:00 ET"}}
        assert pm.is_intraday_blocked()
        assert pm.intraday_sizing_factor() == 0.5

    def test_non_fomc_day_not_blocked(self):
        pm = _fresh_pm()
        pm._macro_flags = {}
        assert not pm.is_intraday_blocked()

    def test_fomc_date_detected_in_hardcoded_list(self):
        pm = _fresh_pm()
        # 2026-04-29 is a known FOMC date
        events = pm._fetch_macro_events("2026-04-29")
        assert "FOMC" in events


# ─── NFP gate ─────────────────────────────────────────────────────────────────

class TestNfpGate:
    def test_nfp_first_friday_detected(self):
        pm = _fresh_pm()
        # Find the first Friday of April 2026
        d = date(2026, 4, 1)
        while d.weekday() != 4:
            d += timedelta(days=1)
        events = pm._fetch_macro_events(d.isoformat())
        assert "NFP" in events

    def test_non_first_friday_not_nfp(self):
        pm = _fresh_pm()
        # Second Friday of April 2026
        d = date(2026, 4, 1)
        while d.weekday() != 4:
            d += timedelta(days=1)
        second_friday = d + timedelta(weeks=1)
        events = pm._fetch_macro_events(second_friday.isoformat())
        assert "NFP" not in events

    def test_nfp_blocks_intraday_before_10am(self):
        pm = _fresh_pm()
        pm._macro_flags = {"NFP": {"name": "NFP", "time": "08:30 ET"}}
        with patch("app.agents.premarket.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 1, 9, 30, tzinfo=ET)
            assert pm.is_intraday_blocked()

    def test_nfp_allows_intraday_after_10am(self):
        pm = _fresh_pm()
        pm._macro_flags = {"NFP": {"name": "NFP", "time": "08:30 ET"}}
        pm._spy_premarket_pct = 0.0
        with patch("app.agents.premarket.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 1, 10, 5, tzinfo=ET)
            assert not pm.is_intraday_blocked()


# ─── Overnight gap analysis ───────────────────────────────────────────────────

class TestOvernightGapAnalysis:
    def _make_bars(self, prior_close: float, today_close: float) -> pd.DataFrame:
        return pd.DataFrame({
            "close": [prior_close * 0.98, prior_close, today_close],
            "volume": [1_000_000, 1_000_000, 1_000_000],
        })

    def test_small_gap_is_ok(self):
        pm = _fresh_pm()
        bars = self._make_bars(100.0, 101.0)  # +1% gap
        mock_client = MagicMock()
        mock_client.get_bars.return_value = bars
        with patch("app.integrations.get_alpaca_client", return_value=mock_client):
            gaps = pm._check_overnight_gaps(["AAPL"])
        assert gaps["AAPL"]["action"] == "OK"

    def test_moderate_adverse_gap_triggers_reeval(self):
        pm = _fresh_pm()
        bars = self._make_bars(100.0, 96.5)  # -3.5% gap
        mock_client = MagicMock()
        mock_client.get_bars.return_value = bars
        with patch("app.integrations.get_alpaca_client", return_value=mock_client):
            gaps = pm._check_overnight_gaps(["AAPL"])
        assert gaps["AAPL"]["action"] == "REEVAL"

    def test_large_adverse_gap_triggers_auto_exit(self):
        pm = _fresh_pm()
        bars = self._make_bars(100.0, 94.0)  # -6% gap
        mock_client = MagicMock()
        mock_client.get_bars.return_value = bars
        with patch("app.integrations.get_alpaca_client", return_value=mock_client):
            gaps = pm._check_overnight_gaps(["TSLA"])
        assert gaps["TSLA"]["action"] == "AUTO_EXIT"

    def test_reeval_sends_to_queue(self):
        pm = _fresh_pm()
        bars = self._make_bars(100.0, 96.5)  # -3.5%
        mock_client = MagicMock()
        mock_client.get_bars.return_value = bars
        send_fn = MagicMock()
        with patch("app.integrations.get_alpaca_client", return_value=mock_client):
            pm._check_overnight_gaps(["AAPL"], redis_send_fn=send_fn)
        send_fn.assert_called_once()
        args = send_fn.call_args[0]
        assert args[0] == "pm_reeval_requests"
        assert args[1]["symbol"] == "AAPL"

    def test_exit_sends_to_trader_queue(self):
        pm = _fresh_pm()
        bars = self._make_bars(100.0, 93.0)  # -7%
        mock_client = MagicMock()
        mock_client.get_bars.return_value = bars
        send_fn = MagicMock()
        with patch("app.integrations.get_alpaca_client", return_value=mock_client):
            pm._check_overnight_gaps(["NVDA"], redis_send_fn=send_fn)
        send_fn.assert_called_once()
        args = send_fn.call_args[0]
        assert args[0] == "trader_exit_requests"
        assert args[1]["action"] == "EXIT"

    def test_empty_symbols_returns_empty(self):
        pm = _fresh_pm()
        result = pm._check_overnight_gaps([])
        assert result == {}


# ─── 8-K poll throttling ──────────────────────────────────────────────────────

class TestEdgar8KThrottling:
    def test_skips_if_checked_recently(self):
        import time
        pm = _fresh_pm()
        pm._last_8k_check = time.monotonic()  # just checked
        result = pm._check_8k_filings(["AAPL"])
        assert result == []

    def test_polls_when_interval_elapsed(self):
        pm = _fresh_pm()
        from app.agents.premarket import EDGAR_POLL_INTERVAL
        pm._last_8k_check = -(EDGAR_POLL_INTERVAL + 1)  # guaranteed old
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"hits": {"hits": []}}
            mock_get.return_value = mock_resp
            result = pm._check_8k_filings(["AAPL"])
        mock_get.assert_called()
        assert isinstance(result, list)
