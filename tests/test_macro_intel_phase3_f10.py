"""Macro Intel Phase 3 F10 — the SIZE DOWN panel folds the macro sizing factor in.

The fold is honest: the per-symbol NEWS factor (what's numerically applied today) and the
day-level MACRO factor (advisory until unified sizing, F12) are surfaced as TWO distinct
dimensions. This test pins the API contract the frontend relies on: /recent must expose
both `news_sizing_multiplier` and `macro_sizing_factor` per row.
"""
from __future__ import annotations

from datetime import datetime, timezone


class _Row:
    """A stand-in DecisionAudit row carrying every attribute the DTO reads."""
    id = "abc"
    decided_at = datetime(2026, 6, 25, 14, 30, tzinfo=timezone.utc)
    symbol = "AAPL"
    strategy = "swing"
    final_decision = "size_down"
    model_score = 0.62
    size_multiplier = 0.75
    news_sizing_multiplier = 0.75
    macro_sizing_factor = 0.85
    block_reason = None
    news_action_policy = "size_down_light"
    news_materiality = 0.4
    macro_risk_level = "MEDIUM"
    outcome_pnl_pct = None
    outcome_1d_pct = None
    vol_targeting_mult = None
    regime_sizing_mult = None
    regime_label_at_decision = None
    top_features = None
    gate_category = None
    price_at_decision = None


def _patch_one_row(monkeypatch):
    from app.api import nis_routes

    class _Q:
        def order_by(self, *a, **k):
            return self

        def filter(self, *a, **k):
            return self

        def limit(self, *a, **k):
            return self

        def all(self):
            return [_Row()]

    class _DB:
        def query(self, *a, **k):
            return _Q()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(nis_routes, "get_session", lambda: _DB())


def test_recent_exposes_both_sizing_dimensions(monkeypatch):
    from app.api import nis_routes
    _patch_one_row(monkeypatch)
    out = nis_routes.get_recent_decisions(limit=50, strategy=None, final_decision=None, days=0)
    row = out["decisions"][0]
    # both dimensions present and distinct — the frontend reads each separately for the fold
    assert row["news_sizing_multiplier"] == 0.75
    assert row["macro_sizing_factor"] == 0.85
    assert row["size_multiplier"] == 0.75   # unchanged: still the applied (news) scalar


def test_recent_dto_keeps_existing_keys(monkeypatch):
    # guard against an accidental key drop when the two fields were inserted
    from app.api import nis_routes
    _patch_one_row(monkeypatch)
    row = nis_routes.get_recent_decisions(limit=1, strategy=None, final_decision=None, days=0)["decisions"][0]
    for k in ("symbol", "final_decision", "macro_risk_level", "news_action_policy",
              "regime_sizing_mult", "gate_category"):
        assert k in row
