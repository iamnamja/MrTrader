"""R0.2 Phase 1 — daemon-decouple runtime tests (app/trading_runtime.py + app/tradingd.py).

These pin the two safety-critical invariants of the phase:
  1. DEFAULT (in_process) behavior is unchanged — the web boots the brain, and the
     shutdown helpers are still importable from app.main under their original names
     (so the existing test_shutdown_hardening imports keep working).
  2. MUTUAL EXCLUSION — exactly one process ever boots the brain: the web boots it
     unless mode==subprocess; the daemon REFUSES to start unless mode==subprocess.
"""
from __future__ import annotations

import inspect

import pytest

from app import trading_runtime as tr


# ── daemon mode resolution ────────────────────────────────────────────────────
def test_default_mode_is_in_process(monkeypatch):
    monkeypatch.delenv("MRTRADER_DAEMON_MODE", raising=False)
    assert tr.get_daemon_mode() == tr.DAEMON_MODE_IN_PROCESS
    assert tr.web_boots_brain() is True          # default = web boots the brain (today)


def test_subprocess_mode(monkeypatch):
    monkeypatch.setenv("MRTRADER_DAEMON_MODE", "subprocess")
    assert tr.get_daemon_mode() == tr.DAEMON_MODE_SUBPROCESS
    assert tr.web_boots_brain() is False         # daemon owns the brain → web stays out


def test_invalid_mode_falls_back_to_in_process(monkeypatch):
    # fail-safe: a typo must NEVER leave nothing running the brain → fall back to web-boots
    monkeypatch.setenv("MRTRADER_DAEMON_MODE", "subproc")   # typo
    assert tr.get_daemon_mode() == tr.DAEMON_MODE_IN_PROCESS
    assert tr.web_boots_brain() is True


@pytest.mark.parametrize("mode,web_boots", [
    ("in_process", True), ("subprocess", False), ("", True), ("GARBAGE", True),
    ("SUBPROCESS", False),   # case-insensitive
])
def test_web_boots_brain_mutual_exclusion(monkeypatch, mode, web_boots):
    monkeypatch.setenv("MRTRADER_DAEMON_MODE", mode)
    assert tr.web_boots_brain() is web_boots


# ── daemon entry-point interlock ──────────────────────────────────────────────
def _no_run(coro):
    # close the un-awaited coroutine so no "coroutine was never awaited" warning fires
    coro.close()
    return 0


def test_daemon_refuses_to_start_unless_subprocess(monkeypatch):
    """In the default in_process mode the web already runs the brain — the daemon MUST
    refuse (returns 2) and MUST NOT reach asyncio.run (which would boot a 2nd brain)."""
    from app import tradingd
    monkeypatch.delenv("MRTRADER_DAEMON_MODE", raising=False)
    called = {"run": False}

    def _boom(coro):
        called["run"] = True
        coro.close()
        return 0
    monkeypatch.setattr(tradingd.asyncio, "run", _boom)
    rc = tradingd.main()
    assert rc == 2                       # refusal exit code
    assert called["run"] is False        # never tried to boot the brain


def test_daemon_starts_only_in_subprocess_mode(monkeypatch):
    """In subprocess mode the interlock OPENS — main() proceeds to asyncio.run(_run())."""
    from app import tradingd
    monkeypatch.setenv("MRTRADER_DAEMON_MODE", "subprocess")
    called = {"run": 0}

    def _stub(coro):
        called["run"] += 1
        coro.close()
        return 0
    monkeypatch.setattr(tradingd.asyncio, "run", _stub)
    rc = tradingd.main()
    assert rc == 0
    assert called["run"] == 1            # the brain boot path was entered exactly once


# ── default-mode byte-identical guards (source-level, no live boot) ──────────────
def test_main_lifespan_guards_brain_boot_behind_flag():
    import app.main as m
    src = inspect.getsource(m.lifespan)
    # both the boot AND the orchestrator-stop must be gated on web_boots_brain()
    assert src.count("web_boots_brain()") >= 2
    assert "start_trading_brain" in src
    assert "prepare_trading_state" in src


def test_shutdown_helpers_still_exported_from_main():
    # guards the re-export: test_shutdown_hardening imports these names from app.main
    from app.main import _start_shutdown_watchdog, _kill_worker_pools, SHUTDOWN_WATCHDOG_SECONDS
    assert _start_shutdown_watchdog is tr.start_shutdown_watchdog
    assert _kill_worker_pools is tr.kill_worker_pools
    assert SHUTDOWN_WATCHDOG_SECONDS == tr.SHUTDOWN_WATCHDOG_SECONDS


def test_kill_worker_pools_best_effort_no_raise():
    tr.kill_worker_pools()               # safe even with no loky pool present
