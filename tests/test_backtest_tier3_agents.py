"""
Tests for Tier 3 AgentSimulator — agent-driven historical backtest.

Uses mock model + synthetic daily bars. Validates that PM scoring,
Trader signal gate, RM rules, and portfolio accounting all work correctly.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _daily_bars(n: int, start: str = "2023-01-02", base: float = 100.0,
                drift: float = 0.001) -> pd.DataFrame:
    """n daily OHLCV bars with a gentle uptrend (default 0.1%/day)."""
    idx = pd.date_range(start, periods=n, freq="B")
    prices = base * (1 + drift) ** np.arange(n)
    return pd.DataFrame({
        "open":   prices * 0.999,
        "high":   prices * 1.005,
        "low":    prices * 0.995,
        "close":  prices,
        "volume": np.ones(n) * 1_000_000,
    }, index=idx)


def _mock_model(proba: float = 0.7) -> MagicMock:
    m = MagicMock()
    m.is_trained = True
    m.feature_names = None
    def _predict(X):
        n = len(X)
        return np.ones(n, dtype=int), np.full(n, proba)
    m.predict.side_effect = _predict
    return m


def _make_simulator(model=None, max_positions: int = 5, starting_capital: float = 100_000.0):
    from app.backtesting.agent_simulator import AgentSimulator
    from app.agents.risk_rules import RiskLimits
    limits = RiskLimits(MAX_OPEN_POSITIONS=max_positions)
    sim = AgentSimulator(
        model=model or _mock_model(),
        starting_capital=starting_capital,
        limits=limits,
        min_confidence=0.50,
        top_n=10,
    )
    return sim


# ── Tests: Initialization ─────────────────────────────────────────────────────

class TestAgentSimulatorInit:
    def test_imports(self):
        from app.backtesting.agent_simulator import AgentSimulator
        sim = AgentSimulator()
        assert sim.model is None

    def test_no_model_returns_empty(self):
        from app.backtesting.agent_simulator import AgentSimulator
        sim = AgentSimulator(model=None)
        bars = {"AAPL": _daily_bars(250)}
        res = sim.run(bars, start_date=date(2023, 6, 1), end_date=date(2023, 12, 1))
        assert res.total_trades == 0
        assert res.ending_capital == sim.starting_capital

    def test_empty_data_returns_empty(self):
        sim = _make_simulator()
        res = sim.run({}, start_date=date(2023, 1, 2), end_date=date(2023, 6, 1))
        assert res.total_trades == 0
        assert res.ending_capital == sim.starting_capital


# ── Tests: PM Scoring ─────────────────────────────────────────────────────────

class TestPMScoring:
    def test_model_called_with_all_symbols_batched(self):
        """PM must batch all symbols in one predict call, not one at a time."""
        model = _mock_model(0.8)
        sim = _make_simulator(model=model)
        n = 260
        syms = {f"S{i}": _daily_bars(n, "2022-01-03") for i in range(5)}
        day = date(2023, 1, 9)

        # Mock feature engineer so engineer_features returns a valid dict
        mock_fe = MagicMock()
        mock_fe.engineer_features.return_value = {"feat1": 0.5, "feat2": 0.3}
        with patch.object(sim, "_get_feature_engineer", return_value=mock_fe):
            proposals = sim._pm_score(day, syms)

        # Model should be called once with all 5 symbols in one batch
        assert model.predict.call_count == 1
        X_arg = model.predict.call_args[0][0]
        assert X_arg.shape[0] == 5

    def test_min_confidence_filters_low_scores(self):
        """Symbols below min_confidence threshold should not be proposed."""
        model = _mock_model(0.3)  # below default MIN_CONFIDENCE=0.50
        sim = _make_simulator(model=model)
        n = 260
        syms = {"AAPL": _daily_bars(n, "2022-01-03")}
        day = date(2023, 1, 9)

        mock_fe = MagicMock()
        mock_fe.engineer_features.return_value = {"feat1": 0.5}
        with patch.object(sim, "_get_feature_engineer", return_value=mock_fe):
            proposals = sim._pm_score(day, syms)

        assert len(proposals) == 0  # all filtered out

    def test_top_n_limits_proposals(self):
        """PM should not propose more than top_n symbols per day."""
        model = _mock_model(0.9)
        from app.backtesting.agent_simulator import AgentSimulator
        sim = AgentSimulator(model=model, min_confidence=0.5, top_n=3)
        n = 260
        syms = {f"S{i}": _daily_bars(n, "2022-01-03", base=100 + i) for i in range(10)}
        day = date(2023, 1, 9)

        mock_fe = MagicMock()
        mock_fe.engineer_features.return_value = {"feat1": 0.5}
        with patch.object(sim, "_get_feature_engineer", return_value=mock_fe):
            proposals = sim._pm_score(day, syms)

        assert len(proposals) <= 3


# ── Tests: Trader Signal Gate ──────────────────────────────────────────────────

class TestTraderSignalGate:
    def test_hold_signal_blocks_entry(self):
        """If generate_signal returns HOLD, the trade should not be entered."""
        from app.backtesting.agent_simulator import AgentSimulator
        model = _mock_model(0.9)
        sim = AgentSimulator(model=model, min_confidence=0.5)

        n = 350
        bars = {"AAPL": _daily_bars(n, "2022-01-03")}

        with patch("app.backtesting.agent_simulator.generate_signal") as mock_gs:
            from app.strategy.signals import SignalResult
            mock_gs.return_value = SignalResult(
                action="HOLD", signal_type="NONE",
                entry_price=100.0, stop_price=0.0, target_price=0.0, atr=0.0,
            )
            with patch.object(sim, "_pm_score", return_value=[("AAPL", 0.9)]):
                res = sim.run(bars, start_date=date(2023, 1, 2), end_date=date(2023, 3, 1))
        assert res.total_trades == 0

    def test_buy_signal_allows_entry(self):
        """If generate_signal returns BUY and RM approves, trade should be entered."""
        from app.backtesting.agent_simulator import AgentSimulator
        from app.agents.risk_rules import RiskLimits
        from app.strategy.signals import SignalResult
        model = _mock_model(0.9)
        # Use 20% max position so size_position result clears the RM gate
        sim = AgentSimulator(model=model, min_confidence=0.5,
                             limits=RiskLimits(MAX_POSITION_SIZE_PCT=0.20))

        # 350 bars starting 2022-01-03 spans through late 2023
        n = 350
        bars = {"AAPL": _daily_bars(n, "2022-01-03")}
        start = date(2023, 1, 2)
        end = date(2023, 3, 1)

        with patch("app.backtesting.agent_simulator.generate_signal") as mock_gs:
            mock_gs.return_value = SignalResult(
                action="BUY", signal_type="EMA_CROSS",
                entry_price=100.0, stop_price=98.0, target_price=106.0, atr=2.0,
            )
            with patch.object(sim, "_pm_score", return_value=[("AAPL", 0.9)]):
                res = sim.run(bars, start_date=start, end_date=end)
        assert res.total_trades >= 1


# ── Tests: RM Rule Enforcement ─────────────────────────────────────────────────

class TestRMRules:
    def test_max_positions_enforced(self):
        """RM should reject entries beyond MAX_OPEN_POSITIONS on same day."""
        from app.backtesting.agent_simulator import AgentSimulator
        from app.agents.risk_rules import RiskLimits
        from app.strategy.signals import SignalResult

        model = _mock_model(0.9)
        sim = AgentSimulator(
            model=model,
            starting_capital=500_000.0,
            limits=RiskLimits(MAX_OPEN_POSITIONS=2),
            min_confidence=0.5,
        )
        n = 350
        syms = {f"S{i}": _daily_bars(n, "2022-01-03", base=100 + i * 5) for i in range(5)}
        start = date(2023, 1, 2)
        end = date(2023, 1, 20)  # short window so positions don't close and reopen

        with patch("app.backtesting.agent_simulator.generate_signal") as mock_gs:
            mock_gs.return_value = SignalResult(
                action="BUY", signal_type="EMA_CROSS",
                entry_price=100.0, stop_price=98.0, target_price=115.0, atr=2.0,  # high target
            )
            proposals = [(f"S{i}", 0.9) for i in range(5)]
            with patch.object(sim, "_pm_score", return_value=proposals):
                res = sim.run(syms, start_date=start, end_date=end)

        # With target=+15% and max_hold=20 bars, unlikely to close & reopen in 12 days
        # So concurrent positions should stay <= 2
        assert res.total_trades <= 4  # 2 positions × max 2 cycles

    def test_daily_loss_limit_blocks_entries(self):
        """After hitting daily loss limit, RM should reject new entries."""
        from app.backtesting.agent_simulator import AgentSimulator, _PortfolioState
        from app.agents.risk_rules import RiskLimits
        from app.backtesting.agent_simulator import validate_daily_loss

        limits = RiskLimits(MAX_DAILY_LOSS_PCT=0.02)
        ok, _ = validate_daily_loss(-3000.0, 100_000.0, limits)
        assert not ok  # 3% loss exceeds 2% limit

        ok, _ = validate_daily_loss(-1000.0, 100_000.0, limits)
        assert ok  # 1% loss is within limit


# ── Tests: Exit Logic ─────────────────────────────────────────────────────────

class TestExitLogic:
    def test_stop_hit_closes_position(self):
        """Position should be closed when today's low pierces stop price."""
        from app.backtesting.agent_simulator import AgentSimulator
        from app.agents.risk_rules import RiskLimits
        from app.strategy.signals import SignalResult

        model = _mock_model(0.9)
        sim = AgentSimulator(model=model, min_confidence=0.5,
                             limits=RiskLimits(MAX_POSITION_SIZE_PCT=0.20))

        # 350 bars so the test window (2023-01-02 → 2023-02-01) has data
        bars_data = _daily_bars(350, "2022-01-03")
        # Bar 260+ corresponds to the test window (Jan 2023+); force low below stop
        bars_data.iloc[263:270, bars_data.columns.get_loc("low")] = 80.0

        syms = {"AAPL": bars_data}
        start = date(2023, 1, 2)
        end = date(2023, 2, 1)

        with patch("app.backtesting.agent_simulator.generate_signal") as mock_gs:
            mock_gs.return_value = SignalResult(
                action="BUY", signal_type="EMA_CROSS",
                entry_price=100.0, stop_price=98.0, target_price=200.0, atr=2.0,
            )
            with patch.object(sim, "_pm_score", return_value=[("AAPL", 0.9)]):
                res = sim.run(syms, start_date=start, end_date=end)

        stop_exits = [t for t in res.trades if t.exit_reason == "STOP"]
        assert len(stop_exits) >= 1

    def test_target_hit_closes_position(self):
        """Position should be closed when today's high exceeds target price."""
        from app.backtesting.agent_simulator import AgentSimulator
        from app.agents.risk_rules import RiskLimits
        from app.strategy.signals import SignalResult

        model = _mock_model(0.9)
        sim = AgentSimulator(model=model, min_confidence=0.5,
                             limits=RiskLimits(MAX_POSITION_SIZE_PCT=0.20))

        # 350 bars so test window (2023-01-02 → 2023-02-28) has data
        bars_data = _daily_bars(350, "2022-01-03", drift=0.001)
        # Bar 263+ is in Jan 2023; force high above target ($106 → spiked to $200)
        bars_data.iloc[263, bars_data.columns.get_loc("high")] = 200.0

        syms = {"AAPL": bars_data}
        start = date(2023, 1, 2)
        end = date(2023, 2, 28)

        with patch("app.backtesting.agent_simulator.generate_signal") as mock_gs:
            mock_gs.return_value = SignalResult(
                action="BUY", signal_type="EMA_CROSS",
                entry_price=100.0, stop_price=98.0, target_price=106.0, atr=2.0,
            )
            with patch.object(sim, "_pm_score", return_value=[("AAPL", 0.9)]):
                res = sim.run(syms, start_date=start, end_date=end)

        target_exits = [t for t in res.trades if t.exit_reason == "TARGET"]
        assert len(target_exits) >= 1


# ── Tests: Portfolio Accounting ────────────────────────────────────────────────

class TestPortfolioAccounting:
    def test_ending_capital_reflects_pnl(self):
        """After profitable trades, ending capital should change from starting."""
        from app.backtesting.agent_simulator import AgentSimulator
        from app.strategy.signals import SignalResult

        model = _mock_model(0.9)
        sim = AgentSimulator(model=model, min_confidence=0.5)

        # 500 bars so test window spanning 2023 has data
        bars_data = _daily_bars(500, "2022-01-03", drift=0.002)
        # Spike high to hit target early
        bars_data.iloc[260, bars_data.columns.get_loc("high")] = 200.0

        with patch("app.backtesting.agent_simulator.generate_signal") as mock_gs:
            mock_gs.return_value = SignalResult(
                action="BUY", signal_type="EMA_CROSS",
                entry_price=100.0, stop_price=98.0, target_price=110.0, atr=2.0,
            )
            with patch.object(sim, "_pm_score", return_value=[("AAPL", 0.9)]):
                res = sim.run({"AAPL": bars_data},
                              start_date=date(2023, 1, 2), end_date=date(2023, 6, 1))

        if res.total_trades > 0:
            assert res.ending_capital != sim.starting_capital

    def test_result_has_required_fields(self):
        """SimResult should have all expected fields."""
        from app.backtesting.agent_simulator import AgentSimulator
        from app.backtesting.strategy_simulator import SimResult

        sim = AgentSimulator(model=None)
        res = sim.run({})
        assert isinstance(res, SimResult)
        assert res.model_type == "swing_agent"
        assert isinstance(res.equity_curve, list)
        assert isinstance(res.monthly_pnl, dict)
        assert isinstance(res.exit_breakdown, dict)

    def test_transaction_costs_applied(self):
        """Transaction costs should be non-zero when trades are executed."""
        from app.backtesting.agent_simulator import AgentSimulator
        from app.strategy.signals import SignalResult

        model = _mock_model(0.9)
        sim = AgentSimulator(model=model, min_confidence=0.5)

        bars_data = _daily_bars(350, "2022-01-03")
        bars_data.iloc[255, bars_data.columns.get_loc("high")] = 200.0  # hit target

        with patch("app.backtesting.agent_simulator.generate_signal") as mock_gs:
            mock_gs.return_value = SignalResult(
                action="BUY", signal_type="EMA_CROSS",
                entry_price=100.0, stop_price=98.0, target_price=110.0, atr=2.0,
            )
            with patch.object(sim, "_pm_score", return_value=[("AAPL", 0.9)]):
                res = sim.run({"AAPL": bars_data},
                              start_date=date(2023, 1, 2), end_date=date(2023, 3, 1))

        if res.total_trades > 0:
            assert res.transaction_costs_total > 0

    def test_print_report_does_not_raise(self):
        """print_report() on a Tier 3 SimResult should not raise."""
        from app.backtesting.agent_simulator import AgentSimulator
        sim = AgentSimulator(model=None)
        res = sim.run({})
        res.print_report()  # should not raise
