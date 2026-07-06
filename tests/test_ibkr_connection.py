"""R1.0b — IBKRConnectionManager: dedicated-thread connect/dispatch/disconnect, fail-closed, inert.

Uses a fake IB (monkeypatched into ib_insync) so it runs without a live gateway. The live thread/loop
model is separately validated against the real paper gateway during the build.
"""
import pytest

from app.live_trading.ibkr_connection import IBKRConnectionManager


class _FakeIB:
    def __init__(self):
        self._connected = False

    def isConnected(self):
        return self._connected

    async def connectAsync(self, host, port, clientId, readonly, timeout):
        self._connected = True

    def reqMarketDataType(self, t):
        pass

    def managedAccounts(self):
        return ["TEST123"]

    async def reqCurrentTimeAsync(self):
        return "now"

    def disconnect(self):
        self._connected = False


@pytest.fixture
def fake_ib(monkeypatch):
    import ib_insync
    monkeypatch.setattr(ib_insync, "IB", _FakeIB)


def test_connect_dispatch_disconnect(fake_ib):
    mgr = IBKRConnectionManager(client_id=9)
    try:
        assert mgr.connect() is True
        assert mgr.is_connected() is True
        # sync ib call dispatched to the loop thread
        assert mgr.call(lambda ib: ib.managedAccounts()) == ["TEST123"]
        # async ib call (awaitable) resolved on the loop thread — not a pending Future
        assert mgr.call(lambda ib: ib.reqCurrentTimeAsync()) == "now"
        mgr.disconnect()
        assert mgr.is_connected() is False
    finally:
        mgr.stop()


def test_call_fails_closed_when_not_connected(fake_ib):
    mgr = IBKRConnectionManager(client_id=9)
    try:
        with pytest.raises(ConnectionError):
            mgr.call(lambda ib: ib.managedAccounts())   # never connected -> refuse, don't return empty
    finally:
        mgr.stop()


def test_connect_is_idempotent(fake_ib):
    mgr = IBKRConnectionManager(client_id=9)
    try:
        assert mgr.connect() is True
        assert mgr.connect() is True                    # second connect is a no-op, still True
        assert mgr.is_connected() is True
    finally:
        mgr.stop()


def test_cross_thread_call(fake_ib):
    import threading
    mgr = IBKRConnectionManager(client_id=9)
    try:
        assert mgr.connect() is True
        out = {}
        th = threading.Thread(target=lambda: out.update(r=mgr.call(lambda ib: ib.managedAccounts())))
        th.start()
        th.join(timeout=5)
        assert out.get("r") == ["TEST123"]              # dispatch works from a non-owner thread
    finally:
        mgr.stop()


def test_reentrancy_from_loop_thread_is_forbidden(fake_ib):
    # MAJOR-1: a thunk that re-enters the manager must FAIL LOUD, not deadlock.
    mgr = IBKRConnectionManager(client_id=9)
    try:
        assert mgr.connect() is True
        with pytest.raises(RuntimeError):
            mgr.call(lambda ib: mgr.is_connected())   # re-enters from the loop thread
    finally:
        mgr.stop()


def test_stopped_manager_is_terminal(fake_ib):
    # MINOR-4: after stop() the manager never resurrects a thread.
    mgr = IBKRConnectionManager(client_id=9)
    assert mgr.connect() is True
    mgr.stop()
    assert mgr.connect() is False                     # stopped -> fail-closed, no new thread
    with pytest.raises(ConnectionError):
        mgr.call(lambda ib: ib.managedAccounts())


def test_op_timeout_raises_connectionerror_not_raw_timeout(fake_ib):
    # A hung op must fail-closed with ConnectionError (uniform contract), not a raw TimeoutError.
    import asyncio
    mgr = IBKRConnectionManager(client_id=9)
    try:
        assert mgr.connect() is True

        async def _hang(ib):
            await asyncio.sleep(10)
        with pytest.raises(ConnectionError):
            mgr.call(lambda ib: _hang(ib), timeout=0.2)
    finally:
        mgr.stop()


def test_call_on_loop_fail_closed_when_loop_torn_down(fake_ib):
    # The stop()/dead-loop race must raise ConnectionError, never a raw AttributeError/RuntimeError.
    mgr = IBKRConnectionManager(client_id=9)
    try:
        assert mgr.connect() is True
        mgr._stopped = True                              # simulate a concurrent stop() in progress
        with pytest.raises(ConnectionError):
            mgr._call_on_loop(lambda ib: ib.isConnected(), timeout=1.0)
        mgr._loop = None                                 # or the loop already nulled
        with pytest.raises(ConnectionError):
            mgr._call_on_loop(lambda ib: ib.isConnected(), timeout=1.0)
    finally:
        mgr._stopped = False
        mgr.stop()


def test_r1_0b_no_write_surface():
    # MINOR-6: rename-resistant tripwire — no order/place method on the read-only R1.0b manager.
    # (The real read-only gate is connectAsync(readonly=True) + the gateway's Read-Only API.)
    mgr = IBKRConnectionManager()
    for name in dir(mgr):
        if name.startswith("_"):
            continue
        low = name.lower()
        assert "order" not in low and "place" not in low, f"R1.0b must expose no write method: {name}"
