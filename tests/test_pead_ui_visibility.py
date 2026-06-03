"""Tests for PEAD UI visibility.

Covers the three backend surfaces added so the dashboard can distinguish PEAD
(and other directional selectors) from baseline swing/intraday:
  1. /api/dashboard/pead/tracking  — summary math over pead_tracker.read_daily rows
  2. /api/dashboard/proposal-log    — `selector` field + `selector` filter
  3. scripts/migrations/2026_06_proposal_log_selector — batch_id backfill regex
"""
from __future__ import annotations

import importlib.util
import pathlib
from datetime import datetime
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.models import Base, ProposalLog


def _make_client() -> TestClient:
    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


def _daily_row(date: str, **kw):
    """A pead_daily row as read_daily() returns it — every field defaults to None
    (mirrors the partial-upsert NULL semantics) unless overridden."""
    base = dict(
        trade_date=date, n_signals=None, n_entered=None, n_filled=None,
        fill_rate=None, gross_deployed=None, realized_pnl=None, unrealized_pnl=None,
        daily_pnl=None, cumulative_pnl=None, vix_level=None, vix_block_fired=0,
        suppressed_opportunity=0, suppressed_macro=0, suppressed_rm=0,
    )
    base.update(kw)
    return base


# ── /pead/tracking ────────────────────────────────────────────────────────────

class TestPeadTrackingEndpoint:
    def setup_method(self):
        self.client = _make_client()

    def test_empty_returns_zero_summary(self):
        with patch("app.live_trading.pead_tracker.read_daily", return_value=[]):
            r = self.client.get("/api/dashboard/pead/tracking?days=30")
        assert r.status_code == 200
        d = r.json()
        assert d["daily"] == []
        s = d["summary"]
        assert s["n_days"] == 0
        assert s["n_signals"] == 0
        assert s["window_fill_rate"] is None
        assert s["cumulative_pnl"] == 0.0
        assert s["backtest_sharpe"] == 0.546

    def test_summary_funnel_and_pnl(self):
        rows = [
            _daily_row("2026-06-01", n_signals=5, n_entered=3, n_filled=2,
                       daily_pnl=100.0, cumulative_pnl=100.0, vix_level=16.0,
                       suppressed_opportunity=1),
            _daily_row("2026-06-02", n_signals=4, n_entered=0, n_filled=0,
                       daily_pnl=-20.0, cumulative_pnl=80.0, vix_level=31.0,
                       vix_block_fired=1, suppressed_macro=2, suppressed_rm=1),
        ]
        with patch("app.live_trading.pead_tracker.read_daily", return_value=rows):
            r = self.client.get("/api/dashboard/pead/tracking?days=30")
        s = r.json()["summary"]
        assert s["n_days"] == 2
        assert s["n_signals"] == 9
        assert s["n_entered"] == 3
        assert s["n_filled"] == 2
        assert s["window_fill_rate"] == round(2 / 3, 4)
        assert s["cumulative_pnl"] == 80.0          # latest row, not a sum
        assert s["latest_date"] == "2026-06-02"
        assert s["latest_vix"] == 31.0
        assert s["vix_blocks"] == 1
        assert s["suppressed_opportunity"] == 1
        assert s["suppressed_macro"] == 2
        assert s["suppressed_rm"] == 1

    def test_fill_rate_no_div_by_zero(self):
        rows = [_daily_row("2026-06-02", n_signals=4, n_entered=0, n_filled=0)]
        with patch("app.live_trading.pead_tracker.read_daily", return_value=rows):
            r = self.client.get("/api/dashboard/pead/tracking")
        assert r.status_code == 200
        assert r.json()["summary"]["window_fill_rate"] is None

    def test_handles_null_fields(self):
        # partial-upsert row: only signals + vix set, everything else NULL
        rows = [_daily_row("2026-06-03", n_signals=5, vix_level=16.2)]
        with patch("app.live_trading.pead_tracker.read_daily", return_value=rows):
            r = self.client.get("/api/dashboard/pead/tracking")
        assert r.status_code == 200
        s = r.json()["summary"]
        assert s["n_signals"] == 5
        assert s["n_entered"] == 0
        assert s["cumulative_pnl"] == 0.0           # NULL coalesced to 0


# ── proposal-log selector field + filter ────────────────────────────────────────

class TestProposalLogSelector:
    def setup_method(self):
        # StaticPool: one shared connection so every self.Session() sees the SAME
        # in-memory DB (a plain :memory: gives each connection its own empty DB).
        self.engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False},
            poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        now = datetime.utcnow()
        sess = self.Session()
        sess.add_all([
            ProposalLog(strategy="swing", selector="pead", symbol="AAPL",
                        pm_status="SENT", proposed_at=now, scan_time=now),
            ProposalLog(strategy="swing", selector="quality_short", symbol="XYZ",
                        pm_status="SENT", proposed_at=now, scan_time=now),
            ProposalLog(strategy="swing", selector="", symbol="LEG",
                        pm_status="SCORED", proposed_at=now, scan_time=now),
        ])
        sess.commit()
        sess.close()
        self.client = _make_client()

    def test_response_includes_selector(self):
        # fresh session per call — the route closes the session in its finally block
        with patch("app.api.routes.get_session", side_effect=lambda: self.Session()):
            r = self.client.get("/api/dashboard/proposal-log?days=3650")
        assert r.status_code == 200
        got = {row["symbol"]: row["selector"] for row in r.json()}
        assert got == {"AAPL": "pead", "XYZ": "quality_short", "LEG": ""}

    def test_selector_filter_isolates_pead(self):
        with patch("app.api.routes.get_session", side_effect=lambda: self.Session()):
            r = self.client.get("/api/dashboard/proposal-log?days=3650&selector=pead")
        rows = r.json()
        assert len(rows) == 1
        assert rows[0]["symbol"] == "AAPL"

    def test_no_selector_returns_all(self):
        with patch("app.api.routes.get_session", side_effect=lambda: self.Session()):
            r = self.client.get("/api/dashboard/proposal-log?days=3650")
        assert len(r.json()) == 3


# ── migration backfill regex ────────────────────────────────────────────────────

class TestBackfillRegex:
    @staticmethod
    def _pattern():
        p = (pathlib.Path(__file__).resolve().parents[1]
             / "scripts" / "migrations" / "2026_06_proposal_log_selector.py")
        spec = importlib.util.spec_from_file_location("_mig_selector", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._BATCH_SELECTOR_RE

    def test_parses_pead(self):
        m = self._pattern().match("dir_pead_20260603_080400")
        assert m and m.group(1) == "pead"

    def test_parses_underscored_selector(self):
        # greedy .+ must backtrack to leave _YYYYMMDD_HHMMSS at the end
        m = self._pattern().match("dir_quality_short_20260602_120119")
        assert m and m.group(1) == "quality_short"

    def test_rejects_intraday_batch(self):
        assert self._pattern().match("intra_20260602_120119_0945") is None

    def test_rejects_malformed(self):
        assert self._pattern().match("dir_pead_notadate") is None
