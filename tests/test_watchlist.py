"""Tests for watchlist management API."""
from unittest.mock import patch, MagicMock
import pytest


# ── Watchlist routes via shared test_client fixture ───────────────────────────

class TestWatchlistRoutes:

    def test_list_empty(self, test_client):
        r = test_client.get("/api/watchlist")
        assert r.status_code == 200
        assert r.json()["tickers"] == []

    def test_add_ticker(self, test_client):
        r = test_client.post("/api/watchlist", json={"symbol": "TSLA", "sector": "Auto"})
        assert r.status_code == 201
        body = r.json()
        assert body["symbol"] == "TSLA"
        assert body["sector"] == "Auto"
        assert body["active"] is True

    def test_add_then_list(self, test_client):
        test_client.post("/api/watchlist", json={"symbol": "AAPL"})
        r = test_client.get("/api/watchlist")
        symbols = [t["symbol"] for t in r.json()["tickers"]]
        assert "AAPL" in symbols

    def test_add_duplicate_409(self, test_client):
        test_client.post("/api/watchlist", json={"symbol": "MSFT"})
        r = test_client.post("/api/watchlist", json={"symbol": "MSFT"})
        assert r.status_code == 409

    def test_symbol_uppercased(self, test_client):
        r = test_client.post("/api/watchlist", json={"symbol": "nvda"})
        assert r.status_code == 201
        assert r.json()["symbol"] == "NVDA"

    def test_delete_existing(self, test_client):
        test_client.post("/api/watchlist", json={"symbol": "GOOG"})
        r = test_client.delete("/api/watchlist/GOOG")
        assert r.status_code == 200
        assert r.json()["removed"] == "GOOG"

    def test_delete_not_found(self, test_client):
        r = test_client.delete("/api/watchlist/UNKNOWN")
        assert r.status_code == 404

    def test_delete_then_gone(self, test_client):
        test_client.post("/api/watchlist", json={"symbol": "META"})
        test_client.delete("/api/watchlist/META")
        r = test_client.get("/api/watchlist")
        symbols = [t["symbol"] for t in r.json()["tickers"]]
        assert "META" not in symbols

    def test_patch_disable(self, test_client):
        test_client.post("/api/watchlist", json={"symbol": "AMZN"})
        r = test_client.patch("/api/watchlist/AMZN", json={"active": False})
        assert r.status_code == 200
        assert r.json()["active"] is False

    def test_active_only_filter(self, test_client):
        test_client.post("/api/watchlist", json={"symbol": "NFLX"})
        test_client.post("/api/watchlist", json={"symbol": "DIS"})
        test_client.patch("/api/watchlist/DIS", json={"active": False})
        r = test_client.get("/api/watchlist?active_only=true")
        symbols = [t["symbol"] for t in r.json()["tickers"]]
        assert "NFLX" in symbols
        assert "DIS" not in symbols

    def test_patch_notes(self, test_client):
        test_client.post("/api/watchlist", json={"symbol": "JPM"})
        r = test_client.patch("/api/watchlist/JPM", json={"notes": "top pick"})
        assert r.status_code == 200
        assert r.json()["notes"] == "top pick"

    def test_bulk_load_sp100(self, test_client):
        r = test_client.post("/api/watchlist/bulk")
        assert r.status_code == 200
        body = r.json()
        assert body["total_added"] > 0
        assert "AAPL" in body["added"]

    def test_bulk_skips_duplicates(self, test_client):
        test_client.post("/api/watchlist/bulk")
        r2 = test_client.post("/api/watchlist/bulk")
        assert r2.status_code == 200
        assert r2.json()["total_added"] == 0


# ── Portfolio manager universe fallback ───────────────────────────────────────

class TestPortfolioManagerUniverse:
    def _make_pm(self):
        from app.agents.portfolio_manager import PortfolioManager  # noqa
        pm = PortfolioManager.__new__(PortfolioManager)
        pm.logger = MagicMock()
        return pm

    def test_uses_watchlist_when_available(self):
        pm = self._make_pm()

        mock_db = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        row = MagicMock()
        row.symbol = "TSLA"
        mock_query.all.return_value = [row]
        mock_db.query.return_value = mock_query

        with patch("app.database.session.get_session", return_value=mock_db):
            universe = pm._get_universe()

        assert "TSLA" in universe

    def test_falls_back_to_sp100_when_watchlist_empty(self):
        pm = self._make_pm()
        from app.utils.constants import SP_100_TICKERS

        mock_db = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []
        mock_db.query.return_value = mock_query

        with patch("app.database.session.get_session", return_value=mock_db):
            universe = pm._get_universe()

        assert universe == list(SP_100_TICKERS)

    def test_falls_back_to_sp100_on_db_error(self):
        pm = self._make_pm()
        from app.utils.constants import SP_100_TICKERS

        with patch("app.database.session.get_session", side_effect=Exception("db down")):
            universe = pm._get_universe()

        assert universe == list(SP_100_TICKERS)
