"""Alpha-v10 H4/H5 — emergency machinery: out-of-band flatten + durable heartbeat + dead-man watchdog.

Pins: the flatten is DRY-RUN-by-default and only liquidates on execute=True (and never raises); the
heartbeat round-trips + ages correctly + fails safe on missing/corrupt; the watchdog alerts on a
stale/missing heartbeat, re-arms after recovery, and only flattens when --auto-flatten is set.
"""
from __future__ import annotations

import os

import pytest

from app.live_trading import emergency_flatten as ef
from app.live_trading import heartbeat as hb
import scripts.dead_man_watchdog as wd


# ── H4 emergency flatten ──────────────────────────────────────────────────────
class _FakeTradingClient:
    def __init__(self, *, raise_close=False, close_responses=None, on_close=None):
        self.closed = False
        self.cancel_orders_flag = None
        self._raise_close = raise_close
        self._on_close = on_close
        # alpaca-py returns per-symbol responses with an HTTP `status`; default = all-2xx
        self._close_responses = (close_responses if close_responses is not None
                                 else [{"symbol": "SPY", "status": 200}])

    def get_orders(self, filter=None):          # noqa: A002 — matches alpaca-py kw
        return []

    def close_all_positions(self, cancel_orders=False):
        if self._raise_close:
            raise RuntimeError("broker rejected")
        self.closed = True
        self.cancel_orders_flag = cancel_orders
        # a real broker empties the book on a successful all-2xx liquidation; reflect that so the
        # post-flatten verification (Wave 5e) sees a flat book unless a symbol failed
        if self._on_close is not None and all(
                200 <= int(r.get("status", 0)) < 300 for r in self._close_responses):
            self._on_close()
        return self._close_responses


class _FakeAlpaca:
    def __init__(self, *, positions=None, raise_positions=False, raise_close=False,
                 close_responses=None):
        self._positions = positions if positions is not None else [
            {"symbol": "SPY", "qty": "10", "market_value": "1000"}]
        self._raise_positions = raise_positions
        self.trading_client = _FakeTradingClient(
            raise_close=raise_close, close_responses=close_responses,
            on_close=lambda: self._positions.clear())

    def get_positions(self):
        if self._raise_positions:
            raise RuntimeError("positions unavailable")
        return self._positions


def test_flatten_dry_run_reports_and_does_not_liquidate():
    fake = _FakeAlpaca()
    r = ef.flatten_alpaca(execute=False, alpaca=fake)
    assert r["dry_run"] is True and r["ok"] is True
    assert r["positions"] == [{"symbol": "SPY", "qty": "10", "market_value": "1000"}]
    assert fake.trading_client.closed is False          # DRY-RUN never liquidates
    assert any("DRY-RUN" in a for a in r["actions"])


def test_flatten_execute_liquidates_with_cancel_orders():
    fake = _FakeAlpaca()
    r = ef.flatten_alpaca(execute=True, alpaca=fake)
    assert r["dry_run"] is False and r["ok"] is True
    assert fake.trading_client.closed is True and fake.trading_client.cancel_orders_flag is True


def test_flatten_collects_errors_never_raises():
    # get_positions fails AND close fails -> errors collected, no exception escapes
    fake = _FakeAlpaca(raise_positions=True, raise_close=True)
    r = ef.flatten_alpaca(execute=True, alpaca=fake)
    assert r["ok"] is False and len(r["errors"]) >= 1


def test_flatten_partial_failure_is_not_reported_ok():
    # a 207 multi-status where one symbol failed (403) -> ok MUST be False (the misleading-ok bug)
    fake = _FakeAlpaca(close_responses=[{"symbol": "SPY", "status": 200},
                                        {"symbol": "XYZ", "status": 403}])
    r = ef.flatten_alpaca(execute=True, alpaca=fake)
    assert r["ok"] is False
    assert any("XYZ" in e for e in r["errors"])


def test_flatten_cli_defaults_to_dry_run(monkeypatch):
    # the single most safety-critical default: scripts/emergency_flatten.py is dry-run unless --execute
    import sys
    import scripts.emergency_flatten as cli
    seen = {}
    monkeypatch.setattr("app.live_trading.emergency_flatten.flatten_alpaca",
                        lambda **k: seen.update(k) or {"ok": True, "errors": [], "dry_run": True})
    monkeypatch.setattr(sys, "argv", ["emergency_flatten.py"])
    cli.main()
    assert seen.get("execute") is False        # no --execute -> dry-run
    seen.clear()
    monkeypatch.setattr(sys, "argv", ["emergency_flatten.py", "--execute"])
    cli.main()
    assert seen.get("execute") is True


# ── H5 heartbeat ──────────────────────────────────────────────────────────────
def test_heartbeat_write_read_roundtrip(tmp_path):
    p = str(tmp_path / "hb.json")
    assert hb.write_heartbeat(p, now=1000.0) is True
    d = hb.read_heartbeat(p)
    assert d is not None and d["ts"] == 1000.0 and "pid" in d
    assert not os.path.exists(p + ".tmp")                # atomic: tmp cleaned up


def test_heartbeat_age_and_missing(tmp_path):
    p = str(tmp_path / "hb.json")
    hb.write_heartbeat(p, now=1000.0)
    assert hb.heartbeat_age_seconds(p, now=1100.0) == 100.0
    assert hb.heartbeat_age_seconds(str(tmp_path / "nope.json")) is None    # missing -> None


def test_heartbeat_corrupt_file_is_none(tmp_path):
    p = tmp_path / "hb.json"
    p.write_text("{not json")
    assert hb.read_heartbeat(str(p)) is None
    assert hb.heartbeat_age_seconds(str(p)) is None


# ── H5 off-box dead-man's-snitch ──────────────────────────────────────────────
class _InlineThread:
    """Run the target synchronously so the fire-and-forget ping is deterministic in tests."""
    def __init__(self, target=None, daemon=None, name=None):
        self._target = target

    def start(self):
        self._target()


def test_snitch_noop_when_unconfigured(monkeypatch):
    monkeypatch.delenv(hb.SNITCH_URL_ENV, raising=False)
    called = []
    monkeypatch.setattr(hb.urllib.request, "urlopen", lambda *a, **k: called.append(a))
    # unconfigured -> no dispatch, no network call
    assert hb.ping_snitch() is False
    assert called == []


def test_snitch_pings_configured_url(monkeypatch):
    monkeypatch.setattr(hb.threading, "Thread", _InlineThread)
    seen = {}

    class _Resp:
        def close(self):
            seen["closed"] = True

    def _fake_urlopen(u, timeout=None):
        seen["url"] = u
        seen["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(hb.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setenv(hb.SNITCH_URL_ENV, "https://hc-ping.example/abc")
    assert hb.ping_snitch() is True
    assert seen["url"] == "https://hc-ping.example/abc" and seen["closed"] is True


def test_snitch_never_raises_on_network_error(monkeypatch):
    monkeypatch.setattr(hb.threading, "Thread", _InlineThread)

    def _boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(hb.urllib.request, "urlopen", _boom)
    # explicit url arg overrides env; a failing GET must be swallowed (beat loop must never break)
    assert hb.ping_snitch("https://hc-ping.example/abc") is True    # dispatched despite the failure


# ── H5 dead-man watchdog (_check_once) ────────────────────────────────────────
class _RecNotifier:
    def __init__(self):
        self.events = []

    def enqueue(self, event_type, payload, dedup_key=None):
        self.events.append((event_type, payload))


def _patch_watchdog(monkeypatch, age, notifier):
    monkeypatch.setattr("app.live_trading.heartbeat.heartbeat_age_seconds", lambda *a, **k: age)
    monkeypatch.setattr("app.notifications.notifier.enqueue", notifier.enqueue)


def test_watchdog_healthy_no_alert(monkeypatch):
    n = _RecNotifier()
    _patch_watchdog(monkeypatch, age=30.0, notifier=n)
    alerted = wd._check_once(max_stale=600.0, auto_flatten=False, already_alerted=False)
    assert alerted is False and n.events == []


def test_watchdog_stale_alerts_once(monkeypatch):
    n = _RecNotifier()
    _patch_watchdog(monkeypatch, age=900.0, notifier=n)
    a1 = wd._check_once(max_stale=600.0, auto_flatten=False, already_alerted=False)
    assert a1 is True and len(n.events) == 1 and n.events[0][0] == "dead_man_alert"
    # already alerted -> stays quiet (no duplicate while still stale)
    a2 = wd._check_once(max_stale=600.0, auto_flatten=False, already_alerted=True)
    assert a2 is True and len(n.events) == 1


def test_watchdog_missing_heartbeat_alerts(monkeypatch):
    n = _RecNotifier()
    _patch_watchdog(monkeypatch, age=None, notifier=n)          # no heartbeat file at all
    alerted = wd._check_once(max_stale=600.0, auto_flatten=False, already_alerted=False)
    assert alerted is True and len(n.events) == 1


def test_watchdog_recovers_rearms(monkeypatch):
    n = _RecNotifier()
    _patch_watchdog(monkeypatch, age=30.0, notifier=n)
    assert wd._check_once(max_stale=600.0, auto_flatten=False, already_alerted=True) is False


def test_watchdog_auto_flatten_only_when_enabled(monkeypatch):
    n = _RecNotifier()
    _patch_watchdog(monkeypatch, age=900.0, notifier=n)
    called = {"flatten": 0}
    monkeypatch.setattr("app.live_trading.emergency_flatten.flatten_alpaca",
                        lambda **k: called.__setitem__("flatten", called["flatten"] + 1) or
                        {"ok": True, "errors": []})
    # alert-only: no flatten
    wd._check_once(max_stale=600.0, auto_flatten=False, already_alerted=False)
    assert called["flatten"] == 0
    # auto-flatten: flatten invoked (kill-switch best-effort; ignore its result here)
    monkeypatch.setattr("app.live_trading.kill_switch.kill_switch.activate",
                        lambda **k: {"status": "activated"}, raising=False)
    wd._check_once(max_stale=600.0, auto_flatten=True, already_alerted=False)
    assert called["flatten"] == 1


def test_watchdog_start_grace_waits_before_first_check(monkeypatch):
    # serve.ps1 co-launches the watchdog with --start-grace-sec so it doesn't false-alert on the
    # stale heartbeat from the prior run while the brain boots. The grace must sleep BEFORE the
    # first _check_once — otherwise the very first check fires a spurious dead_man_alert on startup.
    slept: list[float] = []
    monkeypatch.setattr(wd.time, "sleep", lambda s: slept.append(s))

    def _stop_the_loop(*a, **k):
        raise KeyboardInterrupt          # BaseException — not swallowed by the loop's `except Exception`

    monkeypatch.setattr(wd, "_check_once", _stop_the_loop)
    monkeypatch.setattr("sys.argv",
                        ["dead_man_watchdog", "--start-grace-sec", "120", "--interval-sec", "5"])

    with pytest.raises(KeyboardInterrupt):
        wd.main()

    assert slept and slept[0] == 120     # grace slept first, before the first check reached


def test_watchdog_no_grace_by_default_checks_immediately(monkeypatch):
    # Default (no serve.ps1 grace, e.g. manual/cron launch): no pre-check sleep — first check is
    # immediate, preserving the existing behavior for the standalone runbook invocation.
    slept: list[float] = []
    monkeypatch.setattr(wd.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(wd, "_check_once", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr("sys.argv", ["dead_man_watchdog", "--interval-sec", "5"])

    with pytest.raises(KeyboardInterrupt):
        wd.main()

    assert slept == []                   # no grace -> nothing slept before the (immediate) first check
