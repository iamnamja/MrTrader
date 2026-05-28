"""Regression tests for the Signal Monitor bug-fix bundle.

Bug 1: MagicMock objects must not leak from tests into agent_decisions.reasoning.
Bug 2: outcome_pnl_pct stored convention is *fraction* (0.0464 = 4.64%) so the
       UI's `value * 100` render is correct.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agents.base import BaseAgent, _sanitize_reasoning
from app.database.models import Base, AgentDecision, DecisionAudit, Trade


def _engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    return eng, sessionmaker(bind=eng)


# ── Bug 1: mock sanitisation ──────────────────────────────────────────────────

class _DummyAgent(BaseAgent):
    async def run(self):  # pragma: no cover
        pass


def test_sanitize_reasoning_scrubs_magicmock():
    leaked = MagicMock()
    leaked.reason = MagicMock()  # nested mock, what RM saw in production
    payload = {"checks": [{"rule": "earnings_gate", "ok": False, "msg": leaked.reason}],
               "failed_rule": "earnings_gate", "nested": {"mock": leaked}}
    cleaned = _sanitize_reasoning(payload)

    # Walk and assert no Mock survived
    import json
    rendered = json.dumps(cleaned, default=str)
    assert "MagicMock" not in rendered
    assert "<scrubbed-mock>" in rendered


@pytest.mark.asyncio
async def test_log_decision_does_not_persist_mocks():
    _, Session = _engine()
    agent = _DummyAgent("test_agent")

    leaked = MagicMock()
    leaked.reason = MagicMock()  # what the earnings_calendar mock looked like
    reasoning = {
        "checks": [{"rule": "earnings_gate", "msg": leaked.reason}],
        "obj": leaked,
    }

    with patch("app.database.session.get_session", side_effect=Session):
        await agent.log_decision("TEST_DECISION", reasoning=reasoning)

    db = Session()
    try:
        rows = db.query(AgentDecision).all()
        assert len(rows) == 1
        # Reasoning is stored as JSON; serialise and check no Mock string leaked.
        import json
        rendered = json.dumps(rows[0].reasoning, default=str)
        assert "MagicMock" not in rendered, rendered
    finally:
        db.close()


# ── Bug 2: outcome_pnl_pct stored as fraction ─────────────────────────────────


def test_backfill_outcomes_stores_pnl_as_fraction():
    """Trade pnl=$464 on $10,000 cost basis = 4.64% = 0.0464 stored as fraction."""
    eng, Session = _engine()

    db = Session()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    audit = DecisionAudit(
        id="abc-123", decided_at=now, symbol="AAPL", strategy="swing",
        final_decision="enter", price_at_decision=100.0,
    )
    db.add(audit)
    db.add(Trade(
        symbol="AAPL", direction="BUY", entry_price=100.0, quantity=100,
        status="CLOSED", pnl=464.0, signal_type="swing", created_at=now,
        closed_at=now + timedelta(hours=4),
    ))
    db.commit()
    db.close()

    with patch("app.database.session.get_session", side_effect=Session):
        from app.database.decision_audit import backfill_outcomes
        n = backfill_outcomes(lookback_days=30)

    assert n == 1
    db = Session()
    try:
        row = db.query(DecisionAudit).first()
        assert row.outcome_pnl_pct is not None
        # 464 / (100 * 100) = 0.0464 — must be fraction, NOT percent (4.64).
        assert abs(row.outcome_pnl_pct - 0.0464) < 1e-4, (
            f"Expected fraction ~0.0464, got {row.outcome_pnl_pct}. "
            "Storing as percent will cause UI to render 100× too large."
        )
    finally:
        db.close()


def test_backfill_outcomes_upper_bound_excludes_later_trades():
    """A trade opened 2 days after the decision must NOT be attributed to it."""
    eng, Session = _engine()

    db = Session()
    decision_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=3)
    audit = DecisionAudit(
        id="bound-1", decided_at=decision_time, symbol="MSFT", strategy="swing",
        final_decision="enter", price_at_decision=300.0,
    )
    db.add(audit)
    # Unrelated later trade on the same symbol — must NOT match.
    db.add(Trade(
        symbol="MSFT", direction="BUY", entry_price=310.0, quantity=10,
        status="CLOSED", pnl=50.0, signal_type="swing",
        created_at=decision_time + timedelta(days=2),
        closed_at=decision_time + timedelta(days=2, hours=4),
    ))
    db.commit()
    db.close()

    with patch("app.database.session.get_session", side_effect=Session):
        from app.database.decision_audit import backfill_outcomes
        n = backfill_outcomes(lookback_days=30)

    assert n == 0, "Outer-bound filter must exclude a trade opened well after the decision"
    db = Session()
    try:
        row = db.query(DecisionAudit).first()
        assert row.outcome_pnl_pct is None
    finally:
        db.close()
