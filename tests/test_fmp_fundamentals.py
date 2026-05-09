"""Tests for app.data.fmp_fundamentals (Phase 93)."""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from app.data import fmp_fundamentals as fmp


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_parquet(tmp_path, monkeypatch):
    """Redirect FMP_PATH to a tmp parquet for the duration of the test."""
    p = tmp_path / "fmp_fundamentals_history.parquet"
    monkeypatch.setattr(fmp, "FMP_PATH", p)
    fmp._LOAD_CACHE.clear()
    return p


def _sample_rows():
    """Three quarterly rows for AAPL spanning 2 fiscal years."""
    return [
        {
            "symbol": "AAPL", "as_of_date": "2023-05-15", "period_end": "2023-03-31",
            "period": "Q1", "fiscal_year": 2023,
            "revenue": 100.0, "net_income": 20.0, "profit_margin": 0.20,
            "revenue_growth_yoy": None, "gross_margin": 0.40, "operating_margin": 0.25,
            "fcf_margin": 0.18, "debt_to_equity": 1.2, "book_value_per_share": 4.0,
            "eps_diluted": 1.50, "shares_outstanding": 1.6e10,
            "operating_cash_flow": 25.0, "capex": -3.0, "data_source": "fmp_quarterly",
        },
        {
            "symbol": "AAPL", "as_of_date": "2023-08-10", "period_end": "2023-06-30",
            "period": "Q2", "fiscal_year": 2023,
            "revenue": 110.0, "net_income": 22.0, "profit_margin": 0.20,
            "revenue_growth_yoy": None, "gross_margin": 0.41, "operating_margin": 0.26,
            "fcf_margin": 0.19, "debt_to_equity": 1.1, "book_value_per_share": 4.2,
            "eps_diluted": 1.60, "shares_outstanding": 1.6e10,
            "operating_cash_flow": 27.0, "capex": -3.0, "data_source": "fmp_quarterly",
        },
        {
            "symbol": "AAPL", "as_of_date": "2024-05-15", "period_end": "2024-03-31",
            "period": "Q1", "fiscal_year": 2024,
            "revenue": 120.0, "net_income": 26.0, "profit_margin": 0.2167,
            "revenue_growth_yoy": None, "gross_margin": 0.42, "operating_margin": 0.27,
            "fcf_margin": 0.20, "debt_to_equity": 1.0, "book_value_per_share": 4.5,
            "eps_diluted": 1.70, "shares_outstanding": 1.6e10,
            "operating_cash_flow": 30.0, "capex": -4.0, "data_source": "fmp_quarterly",
        },
    ]


def _write_sample(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(_sample_rows(), columns=fmp._SCHEMA_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


# ── Schema ───────────────────────────────────────────────────────────────────

def test_schema_columns_correct(tmp_parquet):
    _write_sample(tmp_parquet)
    df = fmp.load_fmp_fundamentals(force_reload=True)
    assert list(df.columns) == fmp._SCHEMA_COLUMNS
    # Required columns explicitly present
    for col in ("symbol", "as_of_date", "period_end", "period", "fiscal_year",
                "revenue", "net_income", "profit_margin", "revenue_growth_yoy",
                "gross_margin", "operating_margin", "fcf_margin",
                "debt_to_equity", "book_value_per_share", "eps_diluted",
                "shares_outstanding", "operating_cash_flow", "capex",
                "data_source"):
        assert col in df.columns


# ── PIT semantics ────────────────────────────────────────────────────────────

def test_pit_safety_returns_none_before_first_filing(tmp_parquet):
    _write_sample(tmp_parquet)
    out = fmp.get_fundamentals_as_of("AAPL", date(2023, 1, 1))
    assert out is None


def test_pit_returns_correct_row(tmp_parquet):
    _write_sample(tmp_parquet)
    out = fmp.get_fundamentals_as_of("AAPL", date(2023, 9, 1), latest_close=200.0)
    # Should pick the 2023-08-10 filing (Q2 2023), not the May filing
    assert out is not None
    # Q2 2023 eps_diluted = 1.60 → PE = 200 / (1.60 * 4) = 31.25
    assert out["pe_ratio"] == pytest.approx(31.25, rel=1e-3)


def test_pit_takes_latest_when_as_of_after_all(tmp_parquet):
    _write_sample(tmp_parquet)
    out = fmp.get_fundamentals_as_of("AAPL", date(2025, 12, 31), latest_close=180.0)
    assert out is not None
    # Should pick 2024-05-15 row (Q1 2024) — bvps=4.5
    assert out["pb_ratio"] == pytest.approx(180.0 / 4.5, rel=1e-3)


# ── PE / PB computation ──────────────────────────────────────────────────────

def test_pe_computed_correctly(tmp_parquet):
    _write_sample(tmp_parquet)
    out = fmp.get_fundamentals_as_of("AAPL", date(2024, 6, 1), latest_close=170.0)
    # Q1 2024 eps_diluted=1.70 → PE = 170 / (1.70 * 4) = 25.0
    assert out["pe_ratio"] == pytest.approx(25.0, rel=1e-3)


def test_pb_computed_correctly(tmp_parquet):
    _write_sample(tmp_parquet)
    out = fmp.get_fundamentals_as_of("AAPL", date(2024, 6, 1), latest_close=180.0)
    # bvps=4.5 → PB = 180 / 4.5 = 40
    assert out["pb_ratio"] == pytest.approx(40.0, rel=1e-3)


def test_pe_pb_zero_without_price(tmp_parquet):
    _write_sample(tmp_parquet)
    out = fmp.get_fundamentals_as_of("AAPL", date(2024, 6, 1), latest_close=None)
    assert out["pe_ratio"] == 0.0
    assert out["pb_ratio"] == 0.0


# ── Revenue growth YoY ───────────────────────────────────────────────────────

def test_revenue_growth_yoy(tmp_parquet):
    rows_no_growth = [
        {**r, "revenue_growth_yoy": None} for r in _sample_rows()
    ]
    out = fmp._compute_yoy_growth(rows_no_growth)
    # Q1 2024 vs Q1 2023: (120-100)/100 = 0.20
    q1_2024 = next(r for r in out if r["fiscal_year"] == 2024 and r["period"] == "Q1")
    assert q1_2024["revenue_growth_yoy"] == pytest.approx(0.20, rel=1e-3)
    # Q1 2023 has no prior — stays None
    q1_2023 = next(r for r in out if r["fiscal_year"] == 2023 and r["period"] == "Q1")
    assert q1_2023["revenue_growth_yoy"] is None


# ── Incremental dedupe ───────────────────────────────────────────────────────

def test_incremental_deduplicates(tmp_parquet, monkeypatch):
    _write_sample(tmp_parquet)
    # Mock backfill to "re-fetch" the same rows (simulates an incremental run)
    def fake_backfill(symbols, **kwargs):
        return pd.DataFrame(_sample_rows(), columns=fmp._SCHEMA_COLUMNS)
    monkeypatch.setattr(fmp, "backfill_fmp_fundamentals", fake_backfill)
    # Force AAPL to be considered stale
    out = fmp.update_fmp_fundamentals_incremental(["AAPL"], stale_days=0)
    # Should still be exactly 3 rows (deduped on symbol/as_of_date/period_end)
    assert len(out) == 3
    fmp._LOAD_CACHE.clear()
    re_read = fmp.load_fmp_fundamentals(force_reload=True)
    assert len(re_read) == 3


# ── Graceful missing symbol ──────────────────────────────────────────────────

def test_graceful_on_missing_symbol(tmp_parquet):
    _write_sample(tmp_parquet)
    out = fmp.get_fundamentals_as_of("DOES_NOT_EXIST", date(2024, 1, 1))
    assert out is None


def test_graceful_on_missing_parquet(tmp_parquet):
    # Don't write — file should not exist
    assert not tmp_parquet.exists()
    out = fmp.get_fundamentals_as_of("AAPL", date(2024, 1, 1))
    assert out is None


# ── Rate limit ───────────────────────────────────────────────────────────────

def test_fmp_api_rate_limit_respected(monkeypatch):
    """backfill_fmp_fundamentals must enforce request_delay between calls."""
    sleeps: list[float] = []

    def fake_sleep(s):
        sleeps.append(s)

    def fake_fetch_symbol(sym, lookback_quarters=100):
        return [{
            "symbol": sym, "as_of_date": "2024-01-01", "period_end": "2023-12-31",
            "period": "Q4", "fiscal_year": 2023,
            "revenue": 100.0, "net_income": 10.0, "profit_margin": 0.1,
            "revenue_growth_yoy": None, "gross_margin": 0.3, "operating_margin": 0.2,
            "fcf_margin": 0.15, "debt_to_equity": 1.0, "book_value_per_share": 5.0,
            "eps_diluted": 2.0, "shares_outstanding": 1e9,
            "operating_cash_flow": 12.0, "capex": -2.0, "data_source": "fmp_quarterly",
        }]

    monkeypatch.setattr(fmp.time, "sleep", fake_sleep)
    monkeypatch.setattr(fmp, "_fetch_symbol_quarterly", fake_fetch_symbol)

    fmp.backfill_fmp_fundamentals(
        ["AAPL", "MSFT", "GOOG"], workers=1, dry_run=True, request_delay=0.15,
    )
    # One sleep per symbol
    assert len(sleeps) == 3
    assert all(s == 0.15 for s in sleeps)


# ── lookup_pit_from_index (worker fast path) ─────────────────────────────────

def test_lookup_pit_from_index_pe_pb(tmp_parquet):
    _write_sample(tmp_parquet)
    df = fmp.load_fmp_fundamentals(force_reload=True)
    idx = fmp.build_fmp_lookup_index(df)
    aapl_idx = idx["AAPL"]
    out = fmp.lookup_pit_from_index(aapl_idx, date(2024, 6, 1), latest_close=170.0)
    assert out is not None
    assert out["pe_ratio"] == pytest.approx(25.0, rel=1e-3)
    assert out["pb_ratio"] == pytest.approx(170.0 / 4.5, rel=1e-3)


def test_lookup_pit_from_index_returns_none_before_first(tmp_parquet):
    _write_sample(tmp_parquet)
    df = fmp.load_fmp_fundamentals(force_reload=True)
    idx = fmp.build_fmp_lookup_index(df)
    out = fmp.lookup_pit_from_index(idx["AAPL"], date(2020, 1, 1), latest_close=100.0)
    assert out is None
