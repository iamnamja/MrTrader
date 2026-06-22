"""Alpha-v10 H8 — alerting/observability: severity tiers + critical-channel hook + gate-error visibility.

Pins: events are classified into severity tiers and the email subject is prefixed for triage;
CATASTROPHIC events route to the (env-gated, no-op-by-default) louder channel; and a whole-book gate
that can't even EVALUATE now emits a `gate_error` alert instead of failing silently.
"""
from __future__ import annotations

from app.notifications import notifier as n
from app.live_trading import whole_book_gate as wbg


# ── severity tiers ────────────────────────────────────────────────────────────
def test_severity_classification():
    assert n.severity("dead_man_alert") == n.CATASTROPHIC
    assert n.severity("kill_switch") == n.CATASTROPHIC
    assert n.severity("gate_error") == n.CATASTROPHIC
    assert n.severity("reconciliation_break") == n.WARNING
    assert n.severity("whole_book_gate_breach") == n.WARNING
    assert n.severity("paper_eod") == n.INFO          # unmapped -> INFO


def test_subject_prefix_by_severity():
    subj_crit, _ = n.render("dead_man_alert", {"age_seconds": 700, "threshold_seconds": 600})
    assert subj_crit.startswith("[CRITICAL] ")
    subj_warn, _ = n.render("reconciliation_break", {"label": "trend", "mode": "shadow",
                                                     "status": "FAIL_CLOSED"})
    assert subj_warn.startswith("[WARN] ")
    subj_info, _ = n.render("paper_eod", {"date": "2026-06-22", "pnl": 0.0})
    assert not subj_info.startswith("[CRITICAL]") and not subj_info.startswith("[WARN]")


def test_gate_error_renders():
    subj, body = n.render("gate_error", {"gate": "whole_book", "label": "trend",
                                         "mode": "shadow", "error": "boom"})
    assert subj.startswith("[CRITICAL] ") and "GATE ERROR" in subj
    assert "whole_book" in body and "boom" in body


# ── critical-channel hook (env-gated, no-op default) ──────────────────────────
def test_route_critical_noop_without_webhook(monkeypatch):
    monkeypatch.delenv("MRTRADER_CRITICAL_WEBHOOK", raising=False)
    import httpx
    calls = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: calls.append((a, k)))
    assert n._route_critical("dead_man_alert", {"age_seconds": 700}) is None
    assert calls == []                                # nothing posted when unconfigured


def test_route_critical_posts_when_configured(monkeypatch):
    monkeypatch.setenv("MRTRADER_CRITICAL_WEBHOOK", "https://example.test/hook")
    import httpx
    calls = []
    monkeypatch.setattr(httpx, "post", lambda url, **k: calls.append((url, k.get("json"))))
    t = n._route_critical("dead_man_alert", {"age_seconds": 700, "threshold_seconds": 600})
    t.join(timeout=3)                                 # fire-and-forget thread; join for determinism
    assert len(calls) == 1
    url, body = calls[0]
    assert url == "https://example.test/hook"
    assert body["event"] == "dead_man_alert" and body["severity"] == n.CATASTROPHIC


def test_route_critical_never_raises(monkeypatch):
    monkeypatch.setenv("MRTRADER_CRITICAL_WEBHOOK", "https://example.test/hook")
    import httpx

    def _boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(httpx, "post", _boom)
    t = n._route_critical("dead_man_alert", {"age_seconds": 1})   # must not raise (error is in-thread)
    if t:
        t.join(timeout=3)                             # the in-thread exception is swallowed


# ── enqueue routes catastrophic events (without touching the real queue DB) ────
class _FakeCur:
    lastrowid = 7


class _FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a):
        return _FakeCur()


def test_enqueue_routes_critical_for_catastrophic_only(monkeypatch):
    monkeypatch.setattr(n, "_conn", lambda: _FakeConn())
    routed = []
    monkeypatch.setattr(n, "_route_critical", lambda et, p: routed.append(et))
    assert n.enqueue("dead_man_alert", {"age_seconds": 700}) == 7
    assert routed == ["dead_man_alert"]               # CATASTROPHIC -> routed
    routed.clear()
    n.enqueue("paper_eod", {"date": "x", "pnl": 0.0})
    assert routed == []                               # INFO -> not routed


class _DedupConn(_FakeConn):
    def execute(self, sql, *a):
        # mimic the dedup SELECT returning an existing unsent row
        class _R:
            def fetchone(self_inner):
                return (99,)
        return _R()


def test_enqueue_does_not_route_on_dedup_hit(monkeypatch):
    monkeypatch.setattr(n, "_conn", lambda: _DedupConn())
    routed = []
    monkeypatch.setattr(n, "_route_critical", lambda et, p: routed.append(et))
    assert n.enqueue("dead_man_alert", {"age_seconds": 700}, dedup_key="x") is None
    assert routed == []                               # deduped -> already alerted -> no route


# ── gate-error visibility: a gate that can't evaluate ALERTS (still fail-safe allow) ──
def test_whole_book_gate_error_emits_alert(monkeypatch):
    class _Notifier:
        def __init__(self):
            self.events = []

        def enqueue(self, et, payload, dedup_key=None):
            self.events.append((et, payload))

    monkeypatch.setattr(wbg, "build_proposed_book",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kaboom")))
    nt = _Notifier()
    v = wbg.shadow_gate_from_intents([], [], {}, 100_000.0, mode="enforce", label="trend",
                                     notifier=nt)
    assert v.allow is True and v.error                # fail-SAFE: proceeds, error captured
    assert any(et == "gate_error" for et, _ in nt.events)


def test_whole_book_gate_error_alert_failure_does_not_break_gate(monkeypatch):
    # even if the notifier itself raises, the gate must still return fail-safe allow=True
    class _BoomNotifier:
        def enqueue(self, *a, **k):
            raise RuntimeError("notifier down")

    monkeypatch.setattr(wbg, "build_proposed_book",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kaboom")))
    v = wbg.shadow_gate_from_intents([], [], {}, 100_000.0, mode="enforce", label="trend",
                                     notifier=_BoomNotifier())
    assert v.allow is True and v.error                # notifier failure swallowed; gate still fail-safe
