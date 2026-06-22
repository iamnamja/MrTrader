"""Trend/cash sleeve positions must be managed ONLY by their weekly rebalancers — never by
the PM's 30-min swing-ML review or the trader's PM-message EXIT/EXTEND_TARGET path.

Regression for the bug where the PM re-scored a trend position with the swing model and
force-closed it (Trade #132 QQQ, exit_reason='pm_nis_exit'), fighting the TSMOM rebalancer.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.agents.portfolio_manager import _is_swing_reviewable
from app.agents.trader import _is_rebalancer_managed
from app.startup_reconciler import audit_active_target_stops


def _trade(**kw):
    base = dict(signal_type="", trade_type="swing", selector="")
    base.update(kw)
    return SimpleNamespace(**base)


# ── PM 30-min review work-list (_is_swing_reviewable) ─────────────────────────
def test_swing_position_is_reviewable():
    assert _is_swing_reviewable(_trade(trade_type="swing", selector="")) is True
    assert _is_swing_reviewable(_trade(trade_type="swing", selector="pead")) is True


def test_trend_and_cash_excluded_from_swing_review():
    assert _is_swing_reviewable(_trade(trade_type="trend", selector="trend")) is False
    assert _is_swing_reviewable(_trade(trade_type="cash", selector="cash")) is False
    # excluded if EITHER trade_type OR selector marks it trend/cash
    assert _is_swing_reviewable(_trade(trade_type="swing", selector="trend")) is False
    assert _is_swing_reviewable(_trade(trade_type="trend", selector="")) is False


def test_intraday_excluded_from_swing_review():
    assert _is_swing_reviewable(_trade(signal_type="intraday", trade_type="intraday")) is False


# ── trader sink guard (_is_rebalancer_managed) ───────────────────────────────
def test_rebalancer_managed_detects_trend_cash():
    assert _is_rebalancer_managed({"trade_type": "trend", "selector": "trend"}) is True
    assert _is_rebalancer_managed({"trade_type": "cash"}) is True
    assert _is_rebalancer_managed({"selector": "cash"}) is True
    assert _is_rebalancer_managed({"trade_type": "swing", "selector": "pead"}) is False
    assert _is_rebalancer_managed({}) is False


# ── audit_active_target_stops skips trend/cash (no false "corruption" WARN) ───
class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter_by(self, **kw):
        return self

    def all(self):
        return self._rows


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *_a):
        return _FakeQuery(self._rows)


def test_audit_skips_trend_position_with_large_gain():
    # a long-held trend winner whose target sits >50% above entry must NOT be flagged
    trend = _trade(symbol="IWM", trade_type="trend", selector="trend", direction="BUY",
                   entry_price=296.97, target_price=448.45, stop_price=None, id=131)
    findings = audit_active_target_stops(_FakeDB([trend]))
    assert findings == []


def test_audit_still_flags_corrupt_swing_target():
    # a genuine swing corruption (target ~5x entry) must STILL be flagged
    bad = _trade(symbol="AVGO", trade_type="swing", selector="", direction="BUY",
                 entry_price=413.0, target_price=1993.0, stop_price=None, id=999)
    findings = audit_active_target_stops(_FakeDB([bad]))
    assert len(findings) == 1 and findings[0]["symbol"] == "AVGO"
