"""Fail-closed DB guard: a TEST-MODE process must never connect to the production Postgres.

Regression for the 2026-07-05 incident where a test-mode app subprocess (inheriting
MRTRADER_TEST_MODE but NOT conftest's in-process get_session patch) booted against the live
Postgres and wrote phantom AAPL force-close rows into prod. `session._resolve_database_url`
redirects a test-mode Postgres URL to a SQLite test DB; production is unaffected.
"""
from app.database.session import _resolve_database_url

PROD = "postgresql://mrtrader:mrtrader_password@localhost:5432/mrtrader"


def test_test_mode_redirects_prod_postgres_to_sqlite(monkeypatch):
    monkeypatch.delenv("MRTRADER_TEST_DATABASE_URL", raising=False)
    out = _resolve_database_url(PROD, test_mode=True)
    assert out.startswith("sqlite:///")          # redirected away from Postgres
    assert "postgres" not in out


def test_test_mode_honors_explicit_test_db_url(monkeypatch):
    monkeypatch.setenv("MRTRADER_TEST_DATABASE_URL", "sqlite:///tmp/x/session_guard.db")
    assert _resolve_database_url(PROD, test_mode=True) == "sqlite:///tmp/x/session_guard.db"


def test_production_not_test_mode_uses_prod_url_unchanged():
    # THE important one: real production (not test mode) must be UNAFFECTED.
    assert _resolve_database_url(PROD, test_mode=False) == PROD
    assert _resolve_database_url("postgres://h/db", test_mode=False) == "postgres://h/db"


def test_test_mode_leaves_a_sqlite_url_unchanged(monkeypatch):
    monkeypatch.delenv("MRTRADER_TEST_DATABASE_URL", raising=False)
    url = "sqlite:///:memory:"
    assert _resolve_database_url(url, test_mode=True) == url
