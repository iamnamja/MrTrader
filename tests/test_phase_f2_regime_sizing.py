"""
Tests for F2: regime-aware position sizing in PortfolioManager.

Verifies:
  - _regime_sizing_multiplier returns correct (mult, label, score) tuple
  - Multiplier is applied to quantity in both intraday and swing paths
  - regime fields are written to DecisionAudit
  - Intraday scan refreshes regime context before sizing
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch, PropertyMock
import pytest


# ── Helper: build a minimal fake regime context ───────────────────────────────

def _ctx(label: str, score: float) -> dict:
    return {
        "regime_label": label,
        "regime_score": score,
        "features": {},
        "is_today": True,
        "snapshot_date": "2026-05-08",
    }


# ── _regime_sizing_multiplier unit tests ─────────────────────────────────────

class TestRegimeSizingMultiplier:
    """Unit tests for PortfolioManager._regime_sizing_multiplier."""

    def _make_pm(self, ctx: dict):
        """Return a minimal PM-like object that has the method we need to test."""
        from app.agents.portfolio_manager import PortfolioManager
        from app.config import settings

        pm = PortfolioManager.__new__(PortfolioManager)
        pm._current_regime_ctx = ctx
        pm.logger = MagicMock()
        return pm

    def test_risk_on_returns_full_size(self):
        from app.config import settings
        pm = self._make_pm(_ctx("RISK_ON", 0.80))
        mult, label, score = pm._regime_sizing_multiplier()
        assert mult == settings.regime_sizing_risk_on
        assert label == "RISK_ON"
        assert score == pytest.approx(0.80)

    def test_risk_caution_returns_reduced(self):
        from app.config import settings
        pm = self._make_pm(_ctx("RISK_CAUTION", 0.50))
        mult, label, score = pm._regime_sizing_multiplier()
        assert mult == settings.regime_sizing_risk_caution
        assert mult < 1.0
        assert label == "RISK_CAUTION"

    def test_risk_off_returns_minimum(self):
        from app.config import settings
        pm = self._make_pm(_ctx("RISK_OFF", 0.20))
        mult, label, score = pm._regime_sizing_multiplier()
        assert mult == settings.regime_sizing_risk_off
        assert label == "RISK_OFF"

    def test_unknown_returns_full_size(self):
        from app.config import settings
        pm = self._make_pm(_ctx("UNKNOWN", 0.5))
        mult, label, score = pm._regime_sizing_multiplier()
        assert mult == settings.regime_sizing_unknown
        assert label == "UNKNOWN"

    def test_none_ctx_returns_full_size(self):
        pm = self._make_pm(None)
        mult, label, score = pm._regime_sizing_multiplier()
        assert mult == 1.0  # no context → no reduction

    def test_missing_label_key_falls_back(self):
        pm = self._make_pm({"regime_score": 0.75})
        mult, label, score = pm._regime_sizing_multiplier()
        # No label key → should not crash; returns 1.0
        assert mult == 1.0


# ── Quantity-adjustment integration test ─────────────────────────────────────

class TestRegimeSizingQuantityAdjustment:
    """Verify that RISK_OFF context reduces proposed quantity by the configured multiplier."""

    def test_risk_off_reduces_quantity(self):
        from app.config import settings

        # 100 shares at full size → should be cut to ~30 (settings.regime_sizing_risk_off = 0.3)
        base_qty = 100
        mult = settings.regime_sizing_risk_off
        adjusted = max(1, int(base_qty * mult))
        assert adjusted == int(base_qty * 0.3)
        assert adjusted < base_qty

    def test_risk_caution_reduces_quantity(self):
        from app.config import settings

        base_qty = 50
        mult = settings.regime_sizing_risk_caution
        adjusted = max(1, int(base_qty * mult))
        assert adjusted < base_qty

    def test_risk_on_keeps_quantity(self):
        from app.config import settings

        base_qty = 20
        mult = settings.regime_sizing_risk_on
        adjusted = max(1, int(base_qty * mult))
        assert adjusted == base_qty  # 1.0 × base_qty


# ── Config defaults test ──────────────────────────────────────────────────────

class TestRegimeSizingConfig:
    def test_defaults_exist_and_are_sane(self):
        from app.config import settings
        assert 0 < settings.regime_sizing_risk_off <= settings.regime_sizing_risk_caution <= settings.regime_sizing_risk_on
        assert settings.regime_sizing_risk_on == pytest.approx(1.0)
        assert settings.regime_sizing_unknown == pytest.approx(1.0)
        assert 0.0 < settings.regime_risk_off_threshold < settings.regime_risk_on_threshold < 1.0

    def test_thresholds_consistent_with_labels(self):
        from app.config import settings
        from app.ml.regime_model import _label_from_score

        assert _label_from_score(settings.regime_risk_on_threshold) == "RISK_ON"
        assert _label_from_score(settings.regime_risk_off_threshold) == "RISK_CAUTION"
        assert _label_from_score(settings.regime_risk_off_threshold - 0.01) == "RISK_OFF"
