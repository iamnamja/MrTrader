"""Alpha-v9 2026-06-16 dashboard/UI audit fixes:
  1. Alpaca transient-connection retry (RemoteDisconnected/ConnectionError blips).
  2. Intraday force-close phantom guard (don't 'close' an in-memory position not in Alpaca).
  3. Signal-attribution groups by strategy SOURCE (selector) so PEAD is visible.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# ── 1. Alpaca retry hardening ─────────────────────────────────────────────────
def test_retry_call_succeeds_after_transient_errors():
    from app.integrations.alpaca import _retry_call
    from requests.exceptions import ConnectionError as ReqConnError
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ReqConnError("Connection aborted / RemoteDisconnected")
        return "ok"

    with patch("app.integrations.alpaca.time.sleep", lambda *_: None):
        assert _retry_call(flaky, attempts=3, backoff=0.0) == "ok"
    assert calls["n"] == 3


def test_retry_call_reraises_after_exhausting_attempts():
    from app.integrations.alpaca import _retry_call
    from requests.exceptions import ConnectionError as ReqConnError

    def always_fail():
        raise ReqConnError("down")

    with patch("app.integrations.alpaca.time.sleep", lambda *_: None):
        with pytest.raises(ReqConnError):
            _retry_call(always_fail, attempts=3, backoff=0.0)


def test_retry_call_does_not_retry_non_transient():
    from app.integrations.alpaca import _retry_call
    calls = {"n": 0}

    def value_error():
        calls["n"] += 1
        raise ValueError("not a network error")

    with pytest.raises(ValueError):
        _retry_call(value_error, attempts=3, backoff=0.0)
    assert calls["n"] == 1  # not retried


# ── 2. Intraday force-close phantom guard ─────────────────────────────────────
def _make_trader():
    from app.agents.trader import Trader
    with patch("app.agents.base.BaseAgent.__init__", lambda self, name: None):
        t = Trader.__new__(Trader)
        t.logger = MagicMock()
        t.active_positions = {}
        t.name = "trader"
        return t


@pytest.mark.asyncio
async def test_force_close_drops_phantom_not_in_alpaca():
    t = _make_trader()
    # AAPL is intraday in in-memory state but NOT held in Alpaca (the bug: it was being
    # 'force-closed' every cycle, logging INTRADAY_FORCE_CLOSED + placing spurious sells).
    t.active_positions = {"AAPL": {"trade_type": "intraday", "shares": 1}}
    t._execute_exit = AsyncMock()
    t.log_decision = AsyncMock()

    fake_alpaca = MagicMock()
    fake_alpaca.get_positions.return_value = [{"symbol": "SPY"}]  # no AAPL

    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.all.return_value = []  # no DB intraday rows

    with patch("app.integrations.get_alpaca_client", return_value=fake_alpaca), \
         patch("app.database.session.SessionLocal", return_value=fake_db), \
         patch("app.agents.risk_manager.risk_manager", MagicMock()):
        await t._force_close_intraday()

    # phantom dropped, NO exit order placed, NO force-closed decision logged
    assert "AAPL" not in t.active_positions
    t._execute_exit.assert_not_called()
    t.log_decision.assert_not_called()


@pytest.mark.asyncio
async def test_force_close_still_closes_real_alpaca_position():
    t = _make_trader()
    t.active_positions = {"TSLA": {"trade_type": "intraday", "shares": 10}}
    t._execute_exit = AsyncMock()
    t.log_decision = AsyncMock()

    fake_alpaca = MagicMock()
    fake_alpaca.get_positions.return_value = [{"symbol": "TSLA"}]  # really held
    fake_alpaca.get_latest_price.return_value = 250.0

    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.all.return_value = []

    with patch("app.integrations.get_alpaca_client", return_value=fake_alpaca), \
         patch("app.database.session.SessionLocal", return_value=fake_db), \
         patch("app.agents.risk_manager.risk_manager", MagicMock()):
        await t._force_close_intraday()

    t._execute_exit.assert_awaited_once()
    assert t._execute_exit.call_args[0][0] == "TSLA"


# ── 3. Signal attribution groups by strategy source (selector) ────────────────
class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def all(self):
        return self._rows


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *a, **k):
        return _FakeQuery(self._rows)

    def close(self):
        pass


def test_signal_attribution_surfaces_pead_via_selector():
    from app.analytics.signal_attribution import get_signal_attribution
    rows = [
        SimpleNamespace(selector="pead", signal_type="ML_RANK", pnl=-115.79),
        SimpleNamespace(selector="", signal_type="ML_RANK", pnl=50.0),
        SimpleNamespace(selector=None, signal_type="EMA_CROSSOVER", pnl=20.0),
    ]
    with patch("app.analytics.signal_attribution.get_session", return_value=_FakeDB(rows)):
        result = get_signal_attribution(days=30)
    # PEAD now shows as its own bucket (was previously merged into ML_RANK)
    assert "pead" in result
    assert result["pead"]["trades"] == 1
    # untagged trades fall back to signal_type
    assert "ML_RANK" in result and result["ML_RANK"]["trades"] == 1
    assert "EMA_CROSSOVER" in result
