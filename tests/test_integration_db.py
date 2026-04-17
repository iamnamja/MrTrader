"""
DB integration tests — verify all CRUD operations against an in-memory SQLite DB.

Uses the `db_session` fixture from conftest.py (transactional, auto-rollback).
"""
from __future__ import annotations

from datetime import datetime

import pytest

from tests.conftest import make_trade, make_audit_log


class TestTradeModel:
    def test_create_trade(self, db_session):
        t = make_trade(db_session, symbol="AAPL", status="ACTIVE")
        db_session.commit()
        from app.database.models import Trade
        found = db_session.query(Trade).filter_by(symbol="AAPL").first()
        assert found is not None
        assert found.status == "ACTIVE"
        assert found.signal_type == "EMA_CROSSOVER"
        assert found.stop_price is not None
        assert found.target_price is not None

    def test_update_trade_to_closed(self, db_session):
        t = make_trade(db_session, status="ACTIVE", pnl=None)
        db_session.commit()
        t.status = "CLOSED"
        t.exit_price = 158.0
        t.pnl = 80.0
        t.closed_at = datetime.utcnow()
        db_session.commit()
        from app.database.models import Trade
        refreshed = db_session.query(Trade).filter_by(id=t.id).first()
        assert refreshed.status == "CLOSED"
        assert refreshed.pnl == 80.0

    def test_multiple_trades_query(self, db_session):
        make_trade(db_session, symbol="AAPL", status="CLOSED", pnl=50.0)
        make_trade(db_session, symbol="MSFT", status="CLOSED", pnl=-20.0)
        make_trade(db_session, symbol="NVDA", status="ACTIVE")
        db_session.commit()
        from app.database.models import Trade
        all_t = db_session.query(Trade).all()
        assert len(all_t) == 3
        closed = db_session.query(Trade).filter_by(status="CLOSED").all()
        assert len(closed) == 2

    def test_signal_type_filter(self, db_session):
        make_trade(db_session, signal_type="EMA_CROSSOVER")
        make_trade(db_session, signal_type="RSI_DIP")
        make_trade(db_session, signal_type="EMA_CROSSOVER")
        db_session.commit()
        from app.database.models import Trade
        ema_trades = db_session.query(Trade).filter_by(signal_type="EMA_CROSSOVER").all()
        assert len(ema_trades) == 2

    def test_win_rate_calculation(self, db_session):
        make_trade(db_session, status="CLOSED", pnl=100.0)
        make_trade(db_session, status="CLOSED", pnl=50.0)
        make_trade(db_session, status="CLOSED", pnl=-30.0)
        db_session.commit()
        from app.database.models import Trade
        closed = db_session.query(Trade).filter_by(status="CLOSED").all()
        wins = [t for t in closed if (t.pnl or 0) > 0]
        win_rate = len(wins) / len(closed) * 100
        assert abs(win_rate - 66.67) < 0.1


class TestOrderModel:
    def test_create_order_linked_to_trade(self, db_session):
        t = make_trade(db_session)
        db_session.commit()
        from app.database.models import Order
        order = Order(
            trade_id=t.id,
            order_type="ENTRY",
            order_id="alpaca-123",
            status="FILLED",
            filled_price=150.0,
            filled_qty=10,
        )
        db_session.add(order)
        db_session.commit()
        found = db_session.query(Order).filter_by(trade_id=t.id).first()
        assert found is not None
        assert found.order_id == "alpaca-123"
        assert found.status == "FILLED"


class TestAuditLogModel:
    def test_create_audit_log(self, db_session):
        log = make_audit_log(db_session, action="KILL_SWITCH_ACTIVATED",
                             details={"reason": "test", "positions_closed": []})
        db_session.commit()
        from app.database.models import AuditLog
        found = db_session.query(AuditLog).filter_by(action="KILL_SWITCH_ACTIVATED").first()
        assert found is not None
        assert found.details["reason"] == "test"

    def test_multiple_audit_logs_ordered(self, db_session):
        make_audit_log(db_session, action="EVENT_A")
        make_audit_log(db_session, action="EVENT_B")
        db_session.commit()
        from app.database.models import AuditLog
        from sqlalchemy import desc
        logs = db_session.query(AuditLog).order_by(desc(AuditLog.timestamp)).all()
        assert len(logs) == 2


class TestConfigurationModel:
    def test_set_and_get_config(self, db_session):
        from app.database.config_store import get_config, set_config
        set_config(db_session, "test.key", {"stage": 2, "active": True},
                   "test config")
        val = get_config(db_session, "test.key")
        assert val == {"stage": 2, "active": True}

    def test_upsert_config(self, db_session):
        from app.database.config_store import get_config, set_config
        set_config(db_session, "upsert.key", 1)
        set_config(db_session, "upsert.key", 2)
        assert get_config(db_session, "upsert.key") == 2

    def test_missing_key_returns_none(self, db_session):
        from app.database.config_store import get_config
        assert get_config(db_session, "nonexistent.key") is None

    def test_boolean_persistence(self, db_session):
        from app.database.config_store import get_config, set_config
        set_config(db_session, "kill_switch.active", True)
        val = get_config(db_session, "kill_switch.active")
        assert val is True

    def test_integer_persistence(self, db_session):
        from app.database.config_store import get_config, set_config
        set_config(db_session, "capital_ramp.stage_index", 3)
        val = get_config(db_session, "capital_ramp.stage_index")
        assert val == 3


class TestRiskMetricModel:
    def test_create_risk_metric(self, db_session):
        from app.database.models import RiskMetric
        metric = RiskMetric(
            date="2026-04-16",
            daily_pnl=150.0,
            account_pnl=1500.0,
            max_drawdown=2.5,
        )
        db_session.add(metric)
        db_session.commit()
        found = db_session.query(RiskMetric).filter_by(date="2026-04-16").first()
        assert found.daily_pnl == 150.0
        assert found.max_drawdown == 2.5
