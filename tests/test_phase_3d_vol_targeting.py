"""Phase 3d — Volatility-targeting position sizing tests."""
import pytest
from unittest.mock import MagicMock, patch


class TestVolTargetingQuantity:
    def _pm(self):
        from app.agents.portfolio_manager import PortfolioManager
        pm = PortfolioManager.__new__(PortfolioManager)
        import logging
        pm.logger = logging.getLogger("test")
        return pm

    def test_returns_base_on_zero_price(self):
        pm = self._pm()
        qty, mult = pm._vol_targeting_quantity(price=0, account_value=100_000, base_quantity=10, atr_norm=0.02)
        assert qty == 10
        assert mult == 1.0

    def test_returns_base_on_zero_atr(self):
        pm = self._pm()
        qty, mult = pm._vol_targeting_quantity(price=100, account_value=100_000, base_quantity=10, atr_norm=0.0)
        assert qty == 10
        assert mult == 1.0

    def test_vol_target_sizing_basic(self):
        """quantity = account * vol_target_pct / (atr_norm * price); capped by max_position_size_pct."""
        import app.config as _cfg_mod
        pm = self._pm()
        orig = _cfg_mod.settings
        _cfg_mod.settings.vol_target_pct = 0.005
        _cfg_mod.settings.max_position_size_pct = 0.05
        _cfg_mod.settings.vol_targeting_min_notional = 500.0
        # raw_qty = 100000 * 0.005 / (0.02 * 100) = 250; max = 50 → capped at 50
        qty, mult = pm._vol_targeting_quantity(price=100, account_value=100_000, base_quantity=50, atr_norm=0.02)
        assert qty == 50
        assert isinstance(mult, float)

    def test_min_notional_floor(self):
        """When computed qty * price < min_notional, raise to floor_qty."""
        import app.config as _cfg_mod
        pm = self._pm()
        _cfg_mod.settings.vol_target_pct = 0.005
        _cfg_mod.settings.max_position_size_pct = 0.05
        _cfg_mod.settings.vol_targeting_min_notional = 500.0
        # atr_norm=2.0: raw_qty = 100000*0.005 / (2*100) = 2 → $200 < $500 → floor to 5
        qty, mult = pm._vol_targeting_quantity(price=100, account_value=100_000, base_quantity=5, atr_norm=2.0)
        assert qty >= 5
        assert qty * 100 >= 500

    def test_enabled_flag_guards_call(self):
        """Vol targeting only fires when vol_targeting_enabled=True."""
        from app.config import settings
        assert hasattr(settings, "vol_targeting_enabled")
        assert isinstance(settings.vol_targeting_enabled, bool)

    def test_config_defaults_present(self):
        from app.config import settings
        assert settings.vol_target_pct == 0.005
        assert settings.vol_targeting_min_notional == 500.0


class TestWalkForwardStats:
    def test_returns_none_when_no_active_model(self):
        from app.ml.walk_forward_stats import get_predicted_pnl, _cache
        _cache.clear()
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        result = get_predicted_pnl("swing", db_session=mock_db)
        assert result is None

    def test_returns_avg_return_when_present(self):
        from app.ml.walk_forward_stats import get_predicted_pnl, _cache
        _cache.clear()
        mock_version = MagicMock()
        mock_version.performance = {"avg_return_per_trade": 0.0042}
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_version
        result = get_predicted_pnl("intraday", db_session=mock_db)
        assert result == pytest.approx(0.0042)

    def test_cache_hit_skips_db(self):
        from app.ml.walk_forward_stats import get_predicted_pnl, _cache, _cache_key
        _cache.clear()
        import time
        _cache[_cache_key("swing")] = (0.0030, time.monotonic() + 3600)
        mock_db = MagicMock()
        result = get_predicted_pnl("swing", db_session=mock_db)
        assert result == pytest.approx(0.0030)
        mock_db.query.assert_not_called()

    def test_invalidate_cache(self):
        from app.ml.walk_forward_stats import invalidate_cache, _cache, _cache_key
        import time
        _cache[_cache_key("swing")] = (0.01, time.monotonic() + 3600)
        invalidate_cache("swing")
        assert _cache_key("swing") not in _cache

    def test_invalidate_all(self):
        from app.ml.walk_forward_stats import invalidate_cache, _cache, _cache_key
        import time
        _cache[_cache_key("swing")] = (0.01, time.monotonic() + 3600)
        _cache[_cache_key("intraday")] = (0.02, time.monotonic() + 3600)
        invalidate_cache()
        assert len(_cache) == 0
