"""Tests for Tier 2 StrategySimulator — portfolio-level backtest."""
import pytest
from datetime import date
from app.backtesting.metrics import BacktestResult, Trade
from app.backtesting.strategy_simulator import (
    StrategySimulator, SimResult, STARTING_CAPITAL, MAX_POSITIONS, TRANSACTION_COST
)


def _trade(symbol: str, pnl_pct: float, entry: date, exit_: date,
           hold: int = 5, exit_reason: str = "TARGET") -> Trade:
    ep = 100.0
    xp = ep * (1 + pnl_pct)
    qty = 10
    return Trade(
        symbol=symbol, entry_date=entry, exit_date=exit_,
        entry_price=ep, exit_price=round(xp, 4),
        quantity=qty, pnl=round((xp - ep) * qty, 4),
        pnl_pct=pnl_pct, hold_bars=hold,
        exit_reason=exit_reason, trade_type="swing",
    )


def _result(trades):
    return BacktestResult.from_trades(trades, model_type="swing")


D = date(2024, 1, 2)
D2 = date(2024, 1, 12)
D3 = date(2024, 2, 1)


class TestSimulatorBasics:
    def test_empty_trades_returns_starting_capital(self):
        sim = StrategySimulator()
        res = sim.run(_result([]), start_date=D, end_date=D2)
        assert res.ending_capital == STARTING_CAPITAL
        assert res.total_trades == 0

    def test_winning_trades_increase_capital(self):
        trades = [_trade("AAPL", 0.05, D, D2), _trade("MSFT", 0.03, D, D2)]
        sim = StrategySimulator()
        res = sim.run(_result(trades), start_date=D, end_date=D2)
        assert res.ending_capital > STARTING_CAPITAL
        assert res.total_return_pct > 0

    def test_losing_trades_decrease_capital(self):
        trades = [_trade("AAPL", -0.02, D, D2, exit_reason="STOP")]
        sim = StrategySimulator()
        res = sim.run(_result(trades), start_date=D, end_date=D2)
        assert res.ending_capital < STARTING_CAPITAL

    def test_transaction_costs_applied(self):
        trades = [_trade("AAPL", 0.0, D, D2)]  # flat trade — only cost is TX
        sim = StrategySimulator(transaction_cost_pct=TRANSACTION_COST)
        res = sim.run(_result(trades), start_date=D, end_date=D2)
        assert res.transaction_costs_total > 0
        assert res.ending_capital < STARTING_CAPITAL


class TestPositionLimit:
    def test_max_positions_enforced(self):
        """On same entry date, only MAX_POSITIONS trades should be accepted."""
        trades = [_trade(f"SYM{i}", 0.01, D, D2) for i in range(10)]
        sim = StrategySimulator(max_positions=3)
        res = sim.run(_result(trades), start_date=D, end_date=D2)
        assert res.total_trades <= 3

    def test_positions_freed_after_exit(self):
        """Trades on different days can fill position slots freed by earlier exits."""
        # Day 1: fill 3 slots (exit D2), Day 2: 3 new slots available
        trades_d1 = [_trade(f"A{i}", 0.01, D, D2) for i in range(3)]
        trades_d3 = [_trade(f"B{i}", 0.01, D3, date(2024, 2, 10)) for i in range(3)]
        all_trades = trades_d1 + trades_d3
        sim = StrategySimulator(max_positions=3)
        res = sim.run(_result(all_trades), start_date=D, end_date=date(2024, 2, 10))
        assert res.total_trades == 6  # all 6 accepted (slots freed between D2 and D3)


class TestMetrics:
    def _sim_with(self, pnls):
        trades = [_trade(f"S{i}", p, D, D2) for i, p in enumerate(pnls)]
        return StrategySimulator(max_positions=len(pnls)).run(
            _result(trades), start_date=D, end_date=D2
        )

    def test_win_rate_correct(self):
        res = self._sim_with([0.05, 0.03, -0.02, -0.01])
        assert res.win_rate == pytest.approx(0.5, abs=0.01)

    def test_profit_factor_above_1_when_profitable(self):
        res = self._sim_with([0.05, 0.05, -0.01])
        assert res.profit_factor > 1.0

    def test_sharpe_positive_when_mostly_winning(self):
        res = self._sim_with([0.03] * 8 + [-0.005])
        assert res.sharpe_ratio > 0

    def test_max_drawdown_non_negative(self):
        res = self._sim_with([0.05, -0.10, 0.03])
        assert res.max_drawdown_pct >= 0

    def test_annualized_return_scales_with_period(self):
        trades = [_trade("X", 0.01, D, D2)]
        sim = StrategySimulator(max_positions=5)
        # Longer period → same P&L → lower annualized return
        res_short = sim.run(_result(trades), start_date=D, end_date=date(2024, 1, 20))
        res_long  = sim.run(_result(trades), start_date=D, end_date=date(2025, 1, 2))
        assert res_short.annualized_return_pct >= res_long.annualized_return_pct


class TestMonthlyPnL:
    def test_monthly_pnl_keys_are_yyyy_mm(self):
        trades = [
            _trade("A", 0.02, date(2024, 1, 5),  date(2024, 1, 15)),
            _trade("B", 0.03, date(2024, 2, 5),  date(2024, 2, 15)),
        ]
        sim = StrategySimulator(max_positions=2)
        res = sim.run(_result(trades), start_date=date(2024, 1, 1), end_date=date(2024, 3, 1))
        assert "2024-01" in res.monthly_pnl
        assert "2024-02" in res.monthly_pnl

    def test_monthly_pnl_positive_in_winning_month(self):
        trades = [_trade(f"X{i}", 0.03, date(2024, 1, 2), date(2024, 1, 12)) for i in range(3)]
        sim = StrategySimulator(max_positions=3)
        res = sim.run(_result(trades), start_date=date(2024, 1, 1), end_date=date(2024, 2, 1))
        assert res.monthly_pnl.get("2024-01", 0) > 0


class TestSimResultFields:
    def test_result_has_all_required_fields(self):
        trades = [_trade("A", 0.02, D, D2)]
        sim = StrategySimulator()
        res = sim.run(_result(trades), start_date=D, end_date=D2)
        assert isinstance(res, SimResult)
        assert res.model_type == "swing"
        assert res.starting_capital > 0
        assert isinstance(res.exit_breakdown, dict)
        assert isinstance(res.equity_curve, list)
        assert isinstance(res.monthly_pnl, dict)

    def test_print_report_does_not_raise(self):
        trades = [_trade("A", 0.02, D, D2), _trade("B", -0.01, D, D2, exit_reason="STOP")]
        sim = StrategySimulator()
        res = sim.run(_result(trades), start_date=D, end_date=D2)
        res.print_report()  # should not raise
