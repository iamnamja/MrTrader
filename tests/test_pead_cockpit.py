"""PEAD cockpit-v2 endpoint tests (/api/dashboard/pead/detail).

Read-only join of the three PEAD sources into one payload: daily/summary (chart +
funnel), live_vs_backtest (realized vs +0.546), open positions (drift / unrealized
P&L / days-held), and the per-signal log (pm->rm->trader + reason).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.models import Base, Trade, ProposalLog


def _make_client() -> TestClient:
    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


def _seeded_sessionmaker():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                        poolclass=StaticPool)
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng)
    s = Session()
    now = datetime.utcnow()
    s.add(Trade(symbol="GTLB", direction="BUY", entry_price=30.0, quantity=60,
                status="ACTIVE", selector="pead", stop_price=29.0, target_price=33.0,
                created_at=now - timedelta(days=3)))
    s.add(Trade(symbol="OLD", direction="BUY", entry_price=10.0, quantity=10,
                status="CLOSED", selector="pead", created_at=now))  # closed -> excluded
    s.add(ProposalLog(strategy="swing", selector="pead", symbol="HPE", direction="BUY",
                      ml_score=0.90, confidence=0.90, entry_price=55.5, pm_status="SENT",
                      rm_status="APPROVED", trader_status="QUALITY_REJECTED",
                      trader_reason="Spread 5.6% too wide", proposed_at=now, scan_time=now))
    s.commit(); s.close()
    return Session


def _run(client, monkeypatch_targets, current_price=33.0):
    alpaca = MagicMock()
    alpaca.get_positions.return_value = [
        {"symbol": "GTLB", "current_price": current_price, "unrealized_pl": 180.0}]
    Session = _seeded_sessionmaker()
    with patch("app.live_trading.pead_tracker.read_daily", return_value=[]), \
         patch("app.api.routes._alpaca", return_value=alpaca), \
         patch("app.api.routes.get_session", side_effect=lambda: Session()):
        return client.get("/api/dashboard/pead/detail?days=30&signal_days=5")


class TestPeadDetail:
    def setup_method(self):
        self.client = _make_client()

    def test_shape_and_join(self):
        r = _run(self.client, None)
        assert r.status_code == 200
        d = r.json()
        assert set(d) >= {"daily", "summary", "live_vs_backtest", "positions", "signals"}

    def test_open_position_drift_and_exclusion(self):
        d = _run(self.client, None, current_price=33.0).json()
        # only the ACTIVE pead trade (GTLB); the CLOSED one is excluded
        assert len(d["positions"]) == 1
        p = d["positions"][0]
        assert p["symbol"] == "GTLB"
        assert p["current_price"] == 33.0
        assert abs(p["drift_pct"] - 10.0) < 1e-6   # 33/30 - 1 = +10%
        assert p["unrealized_pl"] == 180.0
        assert p["days_held"] == 3

    def test_signal_log_carries_lifecycle_reason(self):
        d = _run(self.client, None).json()
        assert len(d["signals"]) == 1
        sg = d["signals"][0]
        assert sg["symbol"] == "HPE"
        assert sg["rm_status"] == "APPROVED"
        assert sg["trader_status"] == "QUALITY_REJECTED"
        assert "too wide" in sg["trader_reason"]

    def test_empty_is_not_a_500(self):
        # no positions/signals, empty tracker -> still 200 with empty lists
        eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                            poolclass=StaticPool)
        Base.metadata.create_all(eng)
        empty_sm = sessionmaker(bind=eng)
        with patch("app.live_trading.pead_tracker.read_daily", return_value=[]), \
             patch("app.api.routes._alpaca", return_value=MagicMock(get_positions=MagicMock(return_value=[]))), \
             patch("app.api.routes.get_session", side_effect=lambda: empty_sm()):
            r = self.client.get("/api/dashboard/pead/detail")
        assert r.status_code == 200
        d = r.json()
        assert d["positions"] == [] and d["signals"] == []
