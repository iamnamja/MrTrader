"""R1.3 — futures_roll_monitor: urgency tiers, restart catch-up, delivery-risk alerting. Pure logic
(no live positions, nothing traded)."""
from datetime import date, timedelta

from app.live_trading import futures_roll_monitor as mon
from app.live_trading import futures_roll_policy as rp


def test_urgency_tiers_progress_for_cash_market():
    root, cm, lt = "ES", "202609", "20260918"       # fixed_roll < last-trade cap → tiers separate cleanly
    rd = rp.compute_roll_dates(root, contract_month=cm, last_trade=lt)
    ok = mon.assess(root, date(2026, 9, 1), contract_month=cm, last_trade=lt)
    assert ok.urgency == mon.OK and ok.roll_due is False
    appr = mon.assess(root, rp._minus_business_days(rd.recommended, 1), contract_month=cm, last_trade=lt)
    assert appr.urgency == mon.APPROACHING and appr.roll_due is False
    due = mon.assess(root, rd.recommended, contract_month=cm, last_trade=lt)
    assert due.roll_due is True and due.urgency in (mon.ROLL_DUE, mon.CRITICAL)
    crit = mon.assess(root, rd.recommended + timedelta(days=30), contract_month=cm, last_trade=lt)
    assert crit.urgency == mon.CRITICAL


def test_grain_escalates_to_critical_before_fnd():
    # ZS July: FND ≈ 2026-06-30 → CRITICAL fires a couple trading days before it (delivery-risk margin).
    a = mon.assess("ZS", date(2026, 6, 29), contract_month="202607", last_trade="20260714", qty=2)
    assert a.floor == date(2026, 6, 30) and a.urgency == mon.CRITICAL and a.is_alert is True


def test_catch_up_on_restart_returns_due_positions_only():
    held = [
        {"root": "ZS", "contract_month": "202607", "last_trade": "20260714", "qty": 2},   # overdue in Jul
        {"root": "ES", "contract_month": "202612", "last_trade": "20261218", "qty": 1},   # Dec — not due
    ]
    due = mon.catch_up_on_restart(held, date(2026, 7, 7))
    roots = {a.root for a in due}
    assert "ZS" in roots and "ES" not in roots


def test_delivery_risk_alerts_and_notify():
    held = [{"root": "ZS", "contract_month": "202607", "last_trade": "20260714", "qty": 2}]
    alerts = mon.delivery_risk_alerts(held, date(2026, 7, 7))
    assert len(alerts) == 1 and alerts[0].root == "ZS"

    class _FakeNotifier:
        def __init__(self):
            self.calls = []

        def enqueue(self, event, payload, dedup_key=None):
            self.calls.append((event, payload, dedup_key))
            return 1
    n = _FakeNotifier()
    cnt = mon.notify_delivery_risk(alerts, notifier=n)
    assert cnt == 1
    ev, payload, dedup = n.calls[0]
    assert ev == "futures_delivery_risk" and payload["root"] == "ZS" and payload["urgency"] == "critical"
    assert dedup and "ZS" in dedup


def test_physical_with_no_computable_floor_fails_safe_to_critical():
    # MAJOR fail-safe: a physical position with BOTH contract_month and last_trade missing must NOT
    # read OK (silent path to delivery) — it escalates to CRITICAL so it pages.
    a = mon.assess("ZS", date(2026, 7, 7), contract_month=None, last_trade=None, qty=1)
    assert a.floor is None and a.urgency == mon.CRITICAL and a.is_alert is True
    # a CASH market with no floor is fine (no delivery risk) — stays OK.
    c = mon.assess("ES", date(2026, 7, 7), contract_month=None, last_trade=None)
    assert c.urgency == mon.OK
    # an UNKNOWN root defaults to physical → also fails safe.
    u = mon.assess("WHAT", date(2026, 7, 7), contract_month=None, last_trade=None)
    assert u.urgency == mon.CRITICAL


def test_assess_all_skips_malformed_rows():
    out = mon.assess_all(
        [{"nope": 1}, {"root": "ES", "contract_month": "202609", "last_trade": "20260918"}],
        date(2026, 7, 7))
    assert len(out) == 1 and out[0].root == "ES"


def test_delivery_risk_event_is_registered_catastrophic():
    from app.notifications import notifier
    assert notifier.severity("futures_delivery_risk") == notifier.CATASTROPHIC
    subj, html = notifier.render("futures_delivery_risk",
                                 {"root": "ZS", "urgency": "critical", "days_to_floor": -3})
    assert "DELIVERY RISK" in subj and "ZS" in subj
