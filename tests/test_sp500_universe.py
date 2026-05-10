"""
Tests for trading universe consistency.

Verifies:
- SECTOR_MAP covers every symbol in SP_500_TICKERS (no silent unknowns)
- SP_100_TICKERS is a subset of SP_500_TICKERS (backward compatibility)
- PM fallback uses RUSSELL_1000_TICKERS (training universe)
- Training default uses RUSSELL_1000_TICKERS
- Watchlist bulk seed uses RUSSELL_1000_TICKERS
"""
import pytest
from unittest.mock import MagicMock, patch


class TestSectorMapCoverage:
    def test_all_sp500_symbols_in_sector_map(self):
        from app.utils.constants import SP_500_TICKERS, SECTOR_MAP
        missing = [sym for sym in SP_500_TICKERS if sym not in SECTOR_MAP]
        assert missing == [], f"Missing from SECTOR_MAP: {missing}"

    def test_no_unknown_sector_values(self):
        from app.utils.constants import SECTOR_MAP, SECTOR_LIST
        # All values in SECTOR_MAP must be one of the recognized sectors
        valid = set(SECTOR_LIST)
        bad = {sym: sec for sym, sec in SECTOR_MAP.items() if sec not in valid}
        assert bad == {}, f"Invalid sector values: {bad}"

    def test_sector_list_derived_from_map(self):
        from app.utils.constants import SECTOR_MAP, SECTOR_LIST
        assert sorted(set(SECTOR_MAP.values())) == SECTOR_LIST

    def test_sp100_subset_of_sp500(self):
        from app.utils.constants import SP_100_TICKERS, SP_500_TICKERS
        sp500_set = set(SP_500_TICKERS)
        not_in_sp500 = [s for s in SP_100_TICKERS if s not in sp500_set]
        assert not_in_sp500 == [], f"SP-100 symbols not in SP-500: {not_in_sp500}"

    def test_sp500_larger_than_sp100(self):
        from app.utils.constants import SP_100_TICKERS, SP_500_TICKERS
        assert len(SP_500_TICKERS) > len(SP_100_TICKERS)

    def test_sp500_has_all_gics_sectors(self):
        from app.utils.constants import SECTOR_MAP
        sectors = set(SECTOR_MAP.values())
        expected = {
            "Technology", "Communication Services", "Consumer Discretionary",
            "Consumer Staples", "Financial Services", "Health Care",
            "Industrials", "Energy", "Materials", "Real Estate", "Utilities",
        }
        assert expected.issubset(sectors), f"Missing sectors: {expected - sectors}"

    def test_no_duplicates_in_sp500(self):
        from app.utils.constants import SP_500_TICKERS
        assert len(SP_500_TICKERS) == len(set(SP_500_TICKERS))

    def test_r1k_larger_than_sp500(self):
        from app.utils.constants import SP_500_TICKERS, RUSSELL_1000_TICKERS
        assert len(RUSSELL_1000_TICKERS) > len(SP_500_TICKERS)

    def test_sp500_subset_of_r1k(self):
        """SP_500 should be mostly contained in R1K (some overlap expected)."""
        from app.utils.constants import SP_500_TICKERS, RUSSELL_1000_TICKERS
        r1k_set = set(RUSSELL_1000_TICKERS)
        # At least 90% of SP500 should be in R1K
        in_r1k = [s for s in SP_500_TICKERS if s in r1k_set]
        assert len(in_r1k) / len(SP_500_TICKERS) >= 0.90


class TestTrainingDefaultUniverse:
    def test_training_defaults_to_r1k(self):
        """ModelTrainer.train() should use RUSSELL_1000_TICKERS when symbols=None."""
        from app.utils.constants import RUSSELL_1000_TICKERS
        import app.ml.training as training_mod
        assert training_mod.RUSSELL_1000_TICKERS is RUSSELL_1000_TICKERS

    def test_r1k_referenced_in_training(self):
        import inspect
        import app.ml.training as training_mod
        src = inspect.getsource(training_mod)
        assert "RUSSELL_1000_TICKERS" in src


class TestPortfolioManagerFallback:
    def test_pm_imports_r1k(self):
        import inspect
        import app.agents.portfolio_manager as pm_mod
        src = inspect.getsource(pm_mod)
        assert "RUSSELL_1000_TICKERS" in src
        assert "SP_500_TICKERS" not in src


class TestWatchlistSeed:
    def test_bulk_seed_uses_r1k(self):
        import inspect
        import app.api.watchlist_routes as wr_mod
        src = inspect.getsource(wr_mod)
        assert "RUSSELL_1000_TICKERS" in src
        assert "SP_500_TICKERS" not in src

    def test_bulk_seed_endpoint(self):
        from unittest.mock import MagicMock
        from app.api.watchlist_routes import bulk_load_r1k
        from app.utils.constants import RUSSELL_1000_TICKERS

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        result = bulk_load_r1k(db=mock_db)
        assert result["total_added"] == len(RUSSELL_1000_TICKERS)
        assert mock_db.commit.called
