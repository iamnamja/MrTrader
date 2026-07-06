"""
ibkr_connection.py — Alpha-v10 R1.0b: a robust IBKR connection MANAGER.

`ib_insync`'s `IB()` + its asyncio event loop have **thread/loop affinity** — the object must be
created, connected, and driven on ONE thread's loop, or callbacks (execDetails, orderStatus) fire on
the wrong loop and reads race (the #1 gotcha the R1 design review flagged). The read-only P2.2 adapter
connected synchronously on the caller's thread, which is fine for a one-shot script but unsafe for the
long-running app (the sleeves run on a threadpool).

This manager owns a **single dedicated daemon thread running its own event loop**; `IB()` is created
and all IB operations are dispatched onto that loop from any caller thread via `run_coroutine_threadsafe`.
It is the connection foundation the writable IBKR adapter (R1.0c) runs on. **Inert by default**: it does
NOT auto-connect at import or construction, and it has NO order method — R1.0b is read-capable only.

Read-Only API stays ON at the Gateway for R1.0b (we connect `readonly=True`); flipping it OFF is the
explicit owner step at R1.0c, before any order is ever placed.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
import threading
import time
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


class IBKRConnectionManager:
    """Single dedicated ib_insync thread + loop. Thread-safe connect / dispatch / disconnect.

    Usage (R1.0c adapters):  mgr.ensure_connected(); acct = mgr.call(lambda ib: ib.managedAccounts())
    `call`'s thunk receives the live `IB` and may return a value or a coroutine (awaited on the loop).
    """

    def __init__(self, *, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1,
                 connect_timeout: float = 20.0, op_timeout: float = 30.0):
        self._host = str(host)
        self._port = int(port)
        self._client_id = int(client_id)
        self._connect_timeout = float(connect_timeout)
        self._op_timeout = float(op_timeout)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ib: Any = None
        self._ready = threading.Event()
        self._lock = threading.RLock()          # thread-start + connect single-flight (NOT data ops)
        self._last_reconnect_ts = 0.0
        self._stopped = False                   # terminal: a stopped manager never resurrects a thread

    @classmethod
    def from_config(cls, db) -> "IBKRConnectionManager":
        from app.database.agent_config import get_agent_config as g
        return cls(host=str(g(db, "ibkr.host") or "127.0.0.1"),
                   port=int(g(db, "ibkr.port") or 7497),
                   client_id=int(g(db, "ibkr.client_id") or 1))

    # ── dedicated thread + loop ──────────────────────────────────────────────────
    def _loop_forever(self) -> None:
        # Runs on the dedicated thread: own loop + one IB() bound to it, then serve forever.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            from ib_insync import IB          # lazy — module imports without a gateway/ib_insync
            self._ib = IB()
        except Exception:                      # noqa: BLE001 — surface via _ready; caller sees not-ready
            log.exception("ibkr-conn: failed to create IB() on the dedicated loop")
        finally:
            self._ready.set()                  # unblock start() even on failure (caller checks _ib)
        loop.run_forever()
        # loop stopped (stop()): close it
        try:
            loop.close()
        except Exception:                      # noqa: BLE001
            pass

    def _assert_not_loop_thread(self) -> None:
        # MAJOR-1 guard: a thunk (or anything running on the dedicated loop) must NEVER re-enter the
        # manager — it would block on the RLock (owned by another thread) or on its own loop future →
        # hard deadlock. Fail LOUD instead of hanging. (call()'s thunk contract: non-blocking, no re-entry.)
        t = self._thread
        if t is not None and threading.get_ident() == t.ident:
            raise RuntimeError("ibkr-conn: re-entrant call from the dedicated loop thread is forbidden")

    def _ensure_thread(self) -> None:
        if self._stopped:
            raise RuntimeError("ibkr-conn: manager is stopped (create a new instance)")
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready.clear()
        t = threading.Thread(target=self._loop_forever, name="ibkr-loop", daemon=True)
        self._thread = t
        t.start()
        if not self._ready.wait(timeout=10.0) or self._ib is None or self._loop is None:
            raise RuntimeError("ibkr-conn: dedicated loop/IB failed to start")

    def _call_on_loop(self, thunk: Callable[[Any], Any], timeout: Optional[float]) -> Any:
        """Run `thunk(ib)` on the dedicated loop. If it returns a coroutine, await it there."""
        async def _run() -> Any:
            r = thunk(self._ib)
            if inspect.isawaitable(r):          # coroutine OR Future/Task (ib_insync returns Futures)
                r = await r
            return r
        fut = asyncio.run_coroutine_threadsafe(_run(), self._loop)   # type: ignore[arg-type]
        try:
            return fut.result(timeout if timeout is not None else self._op_timeout)
        except concurrent.futures.TimeoutError:
            # MAJOR-2: cancel the still-running coroutine on the loop, else a flaky-gateway timeout
            # leaves it in flight and a later connect() dispatches a SECOND connectAsync → racing sessions.
            fut.cancel()
            raise

    # ── public API ────────────────────────────────────────────────────────────────
    def call(self, thunk: Callable[[Any], Any], *, timeout: Optional[float] = None) -> Any:
        """Dispatch an IB operation onto the dedicated loop from any thread (fail-closed if down).

        Contract: `thunk(ib)` MUST be non-blocking and MUST NOT re-enter the manager (MAJOR-1)."""
        self._assert_not_loop_thread()
        with self._lock:                          # brief: thread-start single-flight ONLY (no loop op)
            try:
                self._ensure_thread()
            except RuntimeError as e:
                raise ConnectionError(str(e))     # uniform fail-closed contract (MINOR-5)
        # Dispatch WITHOUT the manager lock (MAJOR-3): the loop already serializes ops; holding _lock
        # across a blocking op would stall stop()/is_connected(). Per-op timeout+cancel bounds a hang.
        if not self._is_connected_unlocked():
            raise ConnectionError("ibkr-conn: not connected (call refused, fail-closed)")
        return self._call_on_loop(thunk, timeout)

    def connect(self) -> bool:
        """Idempotent, single-flight connect (read-only). Returns True iff connected. Never raises."""
        self._assert_not_loop_thread()
        with self._lock:                          # single-flight: never two concurrent connectAsync
            try:
                self._ensure_thread()
                if self._is_connected_unlocked():
                    return True
                self._call_on_loop(
                    lambda ib: ib.connectAsync(self._host, self._port, clientId=self._client_id,
                                               readonly=True, timeout=self._connect_timeout),
                    timeout=self._connect_timeout + 5.0)
                try:
                    self._call_on_loop(lambda ib: ib.reqMarketDataType(3), timeout=5.0)  # free delayed
                except Exception:              # noqa: BLE001 — non-fatal
                    log.debug("ibkr-conn: reqMarketDataType(delayed) failed", exc_info=True)
                return self._is_connected_unlocked()
            except Exception as e:             # noqa: BLE001 — fail-closed
                log.warning("ibkr-conn: connect failed: %s", e)
                return False

    def ensure_connected(self, *, min_reconnect_gap: float = 5.0) -> bool:
        """Reconnect if the session dropped, rate-limited so a hard-down gateway isn't hammered."""
        self._assert_not_loop_thread()
        if self._is_connected_unlocked():
            return True
        with self._lock:
            now = time.monotonic()
            if now - self._last_reconnect_ts < min_reconnect_gap:
                return False
            self._last_reconnect_ts = now
        return self.connect()

    def _is_connected_unlocked(self) -> bool:
        try:
            if self._ib is None or self._loop is None or self._stopped:
                return False
            return bool(self._call_on_loop(lambda ib: ib.isConnected(), timeout=5.0))
        except Exception:                      # noqa: BLE001 — treat any error as not-connected
            return False

    def is_connected(self) -> bool:
        self._assert_not_loop_thread()
        return self._is_connected_unlocked()   # no manager lock — never blocks behind a data call()

    def disconnect(self) -> None:
        self._assert_not_loop_thread()
        try:
            if self._ib is not None and self._loop is not None:
                self._call_on_loop(lambda ib: ib.disconnect() if ib.isConnected() else None,
                                   timeout=5.0)
        except Exception:                      # noqa: BLE001
            log.debug("ibkr-conn: disconnect error", exc_info=True)

    def stop(self) -> None:
        """Disconnect + tear down the dedicated thread/loop (TERMINAL — a stopped manager stays down)."""
        self._assert_not_loop_thread()
        self._stopped = True                   # set FIRST (no lock) so no new connect/thread starts
        with self._lock:
            loop, thread = self._loop, self._thread
        try:                                   # best-effort disconnect (short timeout; a hang is cancelled)
            if loop is not None and self._ib is not None:
                self._call_on_loop(lambda ib: ib.disconnect() if ib.isConnected() else None, timeout=5.0)
        except Exception:                      # noqa: BLE001
            pass
        with self._lock:
            self._loop = self._thread = self._ib = None
            self._ready.clear()
        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:                  # noqa: BLE001
                pass
        if thread is not None:
            thread.join(timeout=5.0)
