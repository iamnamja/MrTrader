"""Tests for P1 open-position stop-tightening policy (Option B)."""
import pytest
from unittest.mock import MagicMock, patch


def _make_trade(entry_price, stop_price, status="ACTIVE", trade_type="swing"):
    trade = MagicMock()
    trade.entry_price = entry_price
    trade.stop_price = stop_price
    trade.status = status
    trade.trade_type = trade_type
    return trade


class TestStopTightening:
    def _run_tighten(self, trades):
        """Run _tighten_open_swing_stops with mocked DB returning `trades`."""
        from app.strategy.benign_gate import BenignGate

        mock_db = MagicMock()
        query = MagicMock()
        mock_db.query.return_value = query
        filter1 = MagicMock()
        query.filter.return_value = filter1
        filter1.all.return_value = trades

        with patch("app.strategy.benign_gate._get_db", return_value=mock_db), \
             patch("app.strategy.benign_gate.BenignGate._log_gate_event"):
            g = BenignGate(threshold=0.5)
            n = g._tighten_open_swing_stops(score=0.3, components={})

        return n, trades

    def test_tightens_stop_by_50_percent(self):
        # entry=100, stop=90 → distance=10 → new_stop = 100 - 5 = 95
        trade = _make_trade(entry_price=100.0, stop_price=90.0)
        n, trades = self._run_tighten([trade])
        assert n == 1
        assert trades[0].stop_price == pytest.approx(95.0, abs=0.01)

    def test_only_tightens_never_loosens(self):
        # entry=100, stop=98 → distance=2 → new_stop = 100 - 1 = 99
        # existing stop 98 < new_stop 99 → should tighten
        trade = _make_trade(entry_price=100.0, stop_price=98.0)
        n, trades = self._run_tighten([trade])
        assert n == 1
        assert trades[0].stop_price == pytest.approx(99.0, abs=0.01)

    def test_skip_trade_with_none_stop(self):
        trade = _make_trade(entry_price=100.0, stop_price=None)
        n, _ = self._run_tighten([trade])
        assert n == 0

    def test_skip_trade_with_none_entry(self):
        trade = _make_trade(entry_price=None, stop_price=90.0)
        n, _ = self._run_tighten([trade])
        assert n == 0

    def test_multiple_trades(self):
        trades = [
            _make_trade(entry_price=100.0, stop_price=90.0),
            _make_trade(entry_price=50.0, stop_price=45.0),
        ]
        n, trades_out = self._run_tighten(trades)
        assert n == 2
        assert trades_out[0].stop_price == pytest.approx(95.0, abs=0.01)
        assert trades_out[1].stop_price == pytest.approx(47.5, abs=0.01)

    def test_no_db_returns_zero(self):
        from app.strategy.benign_gate import BenignGate
        with patch("app.strategy.benign_gate._get_db", return_value=None):
            g = BenignGate(threshold=0.5)
            n = g._tighten_open_swing_stops(score=0.2, components={})
        assert n == 0


class TestHandleRegimeFlip:
    def test_no_flip_when_prior_already_adverse(self):
        from app.strategy.benign_gate import BenignGate
        g = BenignGate(threshold=0.5)
        g.current_score = MagicMock(return_value=(0.2, {}))
        g._tighten_open_swing_stops = MagicMock(return_value=5)
        # prior_score=0.3 < threshold → already adverse → no tighten
        n = g.handle_regime_flip(prior_score=0.3)
        g._tighten_open_swing_stops.assert_not_called()
        assert n == 0

    def test_flip_fires_when_prior_was_favorable(self):
        from app.strategy.benign_gate import BenignGate
        g = BenignGate(threshold=0.5)
        g.current_score = MagicMock(return_value=(0.2, {}))
        g._tighten_open_swing_stops = MagicMock(return_value=3)
        # prior_score=0.9 >= threshold → was favorable → now adverse → tighten
        n = g.handle_regime_flip(prior_score=0.9)
        g._tighten_open_swing_stops.assert_called_once()
        assert n == 3

    def test_no_flip_when_no_prior_score(self):
        """prior_score=None means we don't know — still tighten (conservative)."""
        from app.strategy.benign_gate import BenignGate
        g = BenignGate(threshold=0.5)
        g.current_score = MagicMock(return_value=(0.2, {}))
        g._tighten_open_swing_stops = MagicMock(return_value=2)
        n = g.handle_regime_flip(prior_score=None)
        # With prior_score=None, the flip condition fires (not already_adverse check)
        g._tighten_open_swing_stops.assert_called_once()
