"""
Tests for Phase 89a and 89b backfill scripts.

Uses dry-run mode so no files are written and no real API calls are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest


# ── Phase 89a: fundamentals backfill ─────────────────────────────────────────

class TestFundamentalsBackfill:
    def test_dry_run_no_crash(self, tmp_path, monkeypatch):
        """dry_run=True should log sample rows and return without writing files."""
        monkeypatch.chdir(tmp_path)

        mock_facts = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {"form": "10-K", "end": "2022-12-31", "start": "2022-01-01", "val": 1_000_000_000},
                                {"form": "10-K", "end": "2023-12-31", "start": "2023-01-01", "val": 1_100_000_000},
                            ]
                        }
                    },
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {"form": "10-K", "end": "2022-12-31", "start": "2022-01-01", "val": 100_000_000},
                                {"form": "10-K", "end": "2023-12-31", "start": "2023-01-01", "val": 110_000_000},
                            ]
                        }
                    },
                    "StockholdersEquity": {
                        "units": {
                            "USD": [
                                {"form": "10-K", "end": "2023-12-31", "val": 500_000_000},
                            ]
                        }
                    },
                    "LongTermDebt": {
                        "units": {
                            "USD": [
                                {"form": "10-K", "end": "2023-12-31", "val": 250_000_000},
                            ]
                        }
                    },
                }
            }
        }

        with patch("app.ml.fundamental_fetcher._get_cik", return_value=12345), \
             patch("app.ml.fundamental_fetcher._get_company_facts", return_value=mock_facts), \
             patch("app.utils.constants.SP_500_TICKERS", ["AAPL"]):

            from scripts.backfill_fundamentals_history import run
            run(workers=1, dry_run=True)

        # No parquet file written in dry_run
        assert not (tmp_path / "data" / "fundamentals" / "fundamentals_history.parquet").exists()

    def test_fetch_symbol_history_correct_values(self):
        """_fetch_symbol_history extracts correct profit_margin and revenue_growth."""
        mock_facts = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {"form": "10-K", "end": "2022-12-31", "start": "2022-01-01", "val": 1_000_000_000},
                                {"form": "10-K", "end": "2023-12-31", "start": "2023-01-01", "val": 1_200_000_000},
                            ]
                        }
                    },
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {"form": "10-K", "end": "2023-12-31", "start": "2023-01-01", "val": 240_000_000},
                            ]
                        }
                    },
                    "StockholdersEquity": {"units": {"USD": []}},
                    "LongTermDebt": {"units": {"USD": []}},
                }
            }
        }

        with patch("app.ml.fundamental_fetcher._get_cik", return_value=99999), \
             patch("app.ml.fundamental_fetcher._get_company_facts", return_value=mock_facts):

            from scripts.backfill_fundamentals_history import _fetch_symbol_history
            rows = _fetch_symbol_history("TEST")

        assert len(rows) == 2
        yr2023 = next(r for r in rows if r["as_of_date"] == "2023-12-31")
        assert yr2023["revenue_growth"] == pytest.approx(0.2, abs=1e-5)
        assert yr2023["profit_margin"] == pytest.approx(0.2, abs=1e-5)
        assert yr2023["pe_ratio"] == 0.0   # placeholder until bar close available
        assert yr2023["pb_ratio"] == 0.0

    def test_no_cik_returns_empty(self):
        with patch("app.ml.fundamental_fetcher._get_cik", return_value=None):
            from scripts.backfill_fundamentals_history import _fetch_symbol_history
            assert _fetch_symbol_history("XXXX") == []


# ── Phase 89b: sector ETF backfill ────────────────────────────────────────────

class TestSectorEtfBackfill:
    def _make_mock_client(self, bars_df):
        client = MagicMock()
        client.get_bars.return_value = bars_df
        return client

    def test_dry_run_no_crash(self, tmp_path, monkeypatch):
        """dry_run=True logs output and returns without writing files."""
        monkeypatch.chdir(tmp_path)

        dates = pd.date_range("2023-01-03", periods=30, freq="B")
        df = pd.DataFrame({
            "open": [100.0] * 30,
            "high": [101.0] * 30,
            "low":  [99.0] * 30,
            "close": [100.5] * 30,
            "volume": [1_000_000] * 30,
        }, index=dates)

        mock_client = self._make_mock_client(df)

        with patch("app.integrations.get_alpaca_client", return_value=mock_client):
            from scripts.backfill_sector_etf_history import run
            run(days=90, dry_run=True)

        assert not (tmp_path / "data" / "sector_etf" / "sector_etf_history.parquet").exists()

    def test_writes_parquet_with_correct_columns(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        dates = pd.date_range("2023-01-03", periods=30, freq="B")
        df = pd.DataFrame({
            "open": [100.0] * 30,
            "high": [101.0] * 30,
            "low":  [99.0] * 30,
            "close": [100.5] * 30,
            "volume": [1_000_000] * 30,
        }, index=dates)

        mock_client = self._make_mock_client(df)

        with patch("app.integrations.get_alpaca_client", return_value=mock_client):
            from scripts.backfill_sector_etf_history import run, SECTOR_ETFS
            run(days=90, dry_run=False)

        out = tmp_path / "data" / "sector_etf" / "sector_etf_history.parquet"
        assert out.exists()
        result = pd.read_parquet(out)
        assert set(result.columns) >= {"etf", "date", "open", "high", "low", "close", "volume"}
        assert set(result["etf"].unique()) == set(SECTOR_ETFS)
