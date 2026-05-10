"""Tests for app/strategy/benign_gate.py — BenignGate inference guard."""
import pytest
from unittest.mock import patch, MagicMock


SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]


def _gate(threshold=0.5, score=0.8, components=None):
    """Build a BenignGate with mocked regime score."""
    if components is None:
        components = {"spy_above_ma50": 1.0, "spy_above_ma200": 1.0,
                      "vix_term_ratio": 1.0, "breadth_20d_change": 1.0,
                      "credit_20d_change": 1.0}
    from app.strategy.benign_gate import BenignGate
    g = BenignGate(threshold=threshold)
    g.current_score = MagicMock(return_value=(score, components))
    return g


class TestGatePassFail:
    def test_favorable_regime_passes_all_symbols(self):
        g = _gate(threshold=0.5, score=0.8)
        result = g.gate(SYMBOLS, reason="swing_ml")
        assert result == SYMBOLS

    def test_adverse_regime_blocks_all_symbols(self):
        g = _gate(threshold=0.5, score=0.3)
        result = g.gate(SYMBOLS, reason="swing_ml")
        assert result == []

    def test_exactly_at_threshold_is_favorable(self):
        g = _gate(threshold=0.5, score=0.5)
        result = g.gate(SYMBOLS, reason="swing_ml")
        assert result == SYMBOLS

    def test_just_below_threshold_blocks(self):
        g = _gate(threshold=0.5, score=0.49)
        result = g.gate(SYMBOLS, reason="swing_ml")
        assert result == []

    def test_empty_symbol_list(self):
        g = _gate(threshold=0.5, score=0.8)
        result = g.gate([], reason="swing_ml")
        assert result == []

    def test_adverse_logs_warning(self, caplog):
        import logging
        g = _gate(threshold=0.5, score=0.2)
        with caplog.at_level(logging.WARNING, logger="app.strategy.benign_gate"):
            g.gate(SYMBOLS, reason="swing_ml")
        assert "BenignGate FIRED" in caplog.text

    def test_favorable_no_warning(self, caplog):
        import logging
        g = _gate(threshold=0.5, score=0.9)
        with caplog.at_level(logging.WARNING, logger="app.strategy.benign_gate"):
            g.gate(SYMBOLS, reason="swing_ml")
        assert "BenignGate FIRED" not in caplog.text


class TestIsFavorable:
    def test_is_favorable_true_above_threshold(self):
        g = _gate(score=0.8)
        assert g.is_favorable() is True

    def test_is_favorable_false_below_threshold(self):
        g = _gate(score=0.3)
        assert g.is_favorable() is False


class TestRegimeFlip:
    def test_flip_from_favorable_to_adverse_calls_tighten(self):
        from app.strategy.benign_gate import BenignGate
        g = BenignGate(threshold=0.5)
        g.current_score = MagicMock(return_value=(0.2, {}))
        g._tighten_open_swing_stops = MagicMock(return_value=3)
        n = g.handle_regime_flip(prior_score=0.8)
        g._tighten_open_swing_stops.assert_called_once()
        assert n == 3

    def test_no_flip_already_adverse_skips_tighten(self):
        from app.strategy.benign_gate import BenignGate
        g = BenignGate(threshold=0.5)
        g.current_score = MagicMock(return_value=(0.2, {}))
        g._tighten_open_swing_stops = MagicMock(return_value=0)
        n = g.handle_regime_flip(prior_score=0.3)  # already adverse
        g._tighten_open_swing_stops.assert_not_called()
        assert n == 0

    def test_favorable_regime_no_action(self):
        from app.strategy.benign_gate import BenignGate
        g = BenignGate(threshold=0.5)
        g.current_score = MagicMock(return_value=(0.8, {}))
        g._tighten_open_swing_stops = MagicMock()
        n = g.handle_regime_flip(prior_score=0.9)
        g._tighten_open_swing_stops.assert_not_called()
        assert n == 0


class TestScoreCache:
    def test_score_cached_per_day(self):
        """current_score() only calls the underlying function once per day."""
        from app.strategy.benign_gate import _SCORE_CACHE, BenignGate
        from datetime import date
        _SCORE_CACHE.clear()

        call_count = {"n": 0}

        def _fake_get():
            call_count["n"] += 1
            return (0.7, {})

        with patch("app.ml.regime_score_pit.get_current_regime_score", _fake_get):
            g = BenignGate(threshold=0.5)
            _SCORE_CACHE.pop(date.today(), None)
            with patch("app.strategy.benign_gate.BenignGate._persist_daily_score"):
                g.current_score()
                g.current_score()
                g.current_score()

        assert call_count["n"] == 1, "Score should only be computed once per day"


class TestLKGHelpers:
    def test_set_and_get_lkg_version(self):
        from app.strategy.benign_gate import get_lkg_version, set_lkg_version

        mock_row = None

        def mock_get_db():
            db = MagicMock()
            query = MagicMock()
            db.query.return_value = query
            filter_result = MagicMock()
            query.filter_by.return_value = filter_result
            filter_result.first.return_value = mock_row
            return db

        with patch("app.strategy.benign_gate._get_db", mock_get_db):
            # Should not raise even if DB is mocked
            set_lkg_version("swing", 182)

    def test_get_lkg_returns_none_when_db_unavailable(self):
        from app.strategy.benign_gate import get_lkg_version
        with patch("app.strategy.benign_gate._get_db", return_value=None):
            result = get_lkg_version("swing")
        assert result is None
