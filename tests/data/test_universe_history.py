"""Unit tests for pit_union and historical_trade_symbols in universe_history."""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

from app.data.universe_history import pit_union, invalidate_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    invalidate_cache()
    yield
    invalidate_cache()


def test_pit_union_combines_start_and_end():
    start = date(2022, 1, 1)
    end = date(2022, 6, 30)
    stub = {start: ["AAPL", "MSFT"], end: ["MSFT", "GOOG"]}
    with patch("app.data.universe_history.members_at", side_effect=lambda i, d: stub.get(d, [])):
        result = pit_union("russell1000", start, end)
    assert set(result) == {"AAPL", "MSFT", "GOOG"}


def test_pit_union_includes_extra_symbols():
    start = date(2022, 1, 1)
    end = date(2022, 6, 30)
    with patch("app.data.universe_history.members_at", return_value=["AAPL"]):
        result = pit_union("russell1000", start, end, extra_symbols=["DELISTED1", "DELISTED2"])
    assert "DELISTED1" in result
    assert "DELISTED2" in result
    assert "AAPL" in result


def test_pit_union_filters_falsy_extras():
    start = date(2022, 1, 1)
    end = date(2022, 6, 30)
    with patch("app.data.universe_history.members_at", return_value=[]):
        result = pit_union("russell1000", start, end, extra_symbols=["", "AAPL"])
    assert "" not in result
    assert "AAPL" in result


def test_pit_union_no_extra_symbols():
    start = date(2022, 1, 1)
    end = date(2022, 6, 30)
    with patch("app.data.universe_history.members_at", return_value=["AAPL"]):
        result = pit_union("russell1000", start, end, extra_symbols=None)
    assert result == ["AAPL"]


def test_pit_union_returns_sorted_list():
    start = date(2022, 1, 1)
    end = date(2022, 6, 30)
    with patch("app.data.universe_history.members_at", side_effect=lambda i, d: ["TSLA", "AAPL", "MSFT"]):
        result = pit_union("russell1000", start, end)
    assert result == sorted(result)


def test_pit_union_deduplicates():
    start = date(2022, 1, 1)
    end = date(2022, 6, 30)
    with patch("app.data.universe_history.members_at", return_value=["AAPL", "MSFT"]):
        result = pit_union("russell1000", start, end, extra_symbols=["AAPL"])
    assert result.count("AAPL") == 1


def test_historical_trade_symbols_returns_list_on_infra_error():
    """When DB/feature-store is unreachable, returns empty list — never raises."""
    from app.data.universe_history import historical_trade_symbols
    result = historical_trade_symbols(date(2020, 1, 1), date(2020, 12, 31), trade_type="swing")
    assert isinstance(result, list)
