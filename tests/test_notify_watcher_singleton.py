"""Tests for notify_watcher proliferation guards.

Covers two fixes:
  1. app/main.py lifespan never spawns the live notify_watcher under pytest.
  2. scripts/notify_watcher.py enforces a pidfile singleton:
       - live pidfile  → second acquire returns False, no "started" banner
       - stale pidfile → reclaimed, acquire returns True
       - clean exit    → pidfile removed

The singleton lock-acquire is unit-tested directly with a tmp pidfile and a
monkeypatched liveness check; no real processes are spawned.
"""
import os

import scripts.notify_watcher as nw


# ── 1. main.py does not spawn the watcher under pytest ───────────────────────
def test_main_lifespan_skips_watcher_under_pytest(monkeypatch):
    """Under pytest the lifespan must NOT spawn notify_watcher.

    We sentinel the notify_watcher script path: if the lifespan ever tries to
    Popen *that script*, the test fails. Other unrelated subprocess use during
    startup is irrelevant to this contract, so we only flag the watcher script.
    """
    from fastapi.testclient import TestClient
    import app.main as main

    real_popen = main.subprocess.Popen
    spawned = {"watcher": 0}

    def _watching_popen(args, *a, **kw):
        try:
            joined = " ".join(str(x) for x in args)
        except TypeError:
            joined = str(args)
        if "notify_watcher" in joined:
            spawned["watcher"] += 1
            raise AssertionError(
                "notify_watcher must not be spawned under pytest"
            )
        return real_popen(args, *a, **kw)

    monkeypatch.setattr(main.subprocess, "Popen", _watching_popen)

    # PYTEST_CURRENT_TEST is set by pytest during this test; the lifespan should
    # detect it and skip the watcher. Running the full lifespan via TestClient.
    with TestClient(main.app) as client:
        assert client is not None

    assert spawned["watcher"] == 0
    assert getattr(main.app.state, "notify_proc", None) is None


# ── 2. singleton lock-acquire ────────────────────────────────────────────────
def test_acquire_no_pidfile(tmp_path):
    """No pidfile → acquire writes our pid and returns True."""
    pid_path = tmp_path / "nw.pid"
    assert nw.acquire_singleton(pid_path, is_live=lambda p: True) is True
    assert pid_path.read_text().strip() == str(os.getpid())


def test_acquire_live_pidfile_returns_false_quietly(tmp_path, caplog):
    """A live pidfile (other pid) → returns False and does NOT log 'started'."""
    pid_path = tmp_path / "nw.pid"
    pid_path.write_text("999999")  # some other pid
    with caplog.at_level("INFO"):
        result = nw.acquire_singleton(pid_path, is_live=lambda p: True)
    assert result is False
    # pidfile untouched (still names the other process)
    assert pid_path.read_text().strip() == "999999"
    # the duplicate-exit path logs "already running", never the "started" banner
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "already running" in msgs
    assert "started" not in msgs


def test_acquire_stale_pidfile_is_reclaimed(tmp_path):
    """A stale pidfile (dead pid) → reclaimed, acquire returns True."""
    pid_path = tmp_path / "nw.pid"
    pid_path.write_text("999999")
    assert nw.acquire_singleton(pid_path, is_live=lambda p: False) is True
    assert pid_path.read_text().strip() == str(os.getpid())


def test_acquire_garbage_pidfile_is_reclaimed(tmp_path):
    """Unparseable pidfile contents → treated as stale, reclaimed."""
    pid_path = tmp_path / "nw.pid"
    pid_path.write_text("not-a-pid")
    # is_live must not even matter; garbage is always stale.
    assert nw.acquire_singleton(pid_path, is_live=lambda p: True) is True
    assert pid_path.read_text().strip() == str(os.getpid())


def test_acquire_own_pid_is_reclaimed(tmp_path):
    """Pidfile naming our own pid → reclaimed (not a duplicate)."""
    pid_path = tmp_path / "nw.pid"
    pid_path.write_text(str(os.getpid()))
    assert nw.acquire_singleton(pid_path, is_live=lambda p: True) is True
    assert pid_path.read_text().strip() == str(os.getpid())


# ── 3. release removes the pidfile (clean exit) ──────────────────────────────
def test_release_removes_own_pidfile(tmp_path):
    pid_path = tmp_path / "nw.pid"
    nw.acquire_singleton(pid_path, is_live=lambda p: True)
    assert pid_path.exists()
    nw.release_singleton(pid_path)
    assert not pid_path.exists()


def test_release_keeps_other_pidfile(tmp_path):
    """release must NOT delete a pidfile owned by a different process."""
    pid_path = tmp_path / "nw.pid"
    pid_path.write_text("999999")
    nw.release_singleton(pid_path)
    assert pid_path.exists()
    assert pid_path.read_text().strip() == "999999"


def test_release_missing_pidfile_is_noop(tmp_path):
    pid_path = tmp_path / "nw.pid"
    # should not raise
    nw.release_singleton(pid_path)
    assert not pid_path.exists()


# ── 4. liveness check fail-safe (indeterminate → reclaim) ────────────────────
def test_pid_is_live_rejects_nonpositive():
    assert nw._pid_is_live(0) is False
    assert nw._pid_is_live(-1) is False
