import backtrader as bt
import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


class MrTraderStrategy(bt.Strategy):
    """
    MrTrader trend-following strategy with ATR-based stops.

    Entry logic (long only, in-uptrend filter):
      - Trend filter : price > EMA(200)  — only trade with the long-term trend
      - Signal A     : EMA(20) crosses above EMA(50)  AND RSI > 50  (momentum breakout)
      - Signal B     : RSI dips below rsi_dip_entry (45) then recovers above it  (pullback buy)
                       while price is above EMA(20)

    Exit logic:
      - Hard stop    : entry - atr_stop_mult * ATR(14)
      - Profit target: entry + atr_target_mult * ATR(14)
      - Trailing stop: ratchets up as price moves in favour (activated after profit >= trail_activation)

    Position sizing: risk 2% of account per trade using the ATR stop distance.
    """

    params = dict(
        rsi_period=14,
        rsi_dip_entry=45,       # RSI level that triggers a pullback-buy on recovery
        ema_fast=20,
        ema_slow=50,
        ema_trend=200,          # long-term trend filter
        atr_period=14,
        atr_stop_mult=2.5,      # stop = entry - 2.5×ATR (wider → fewer whipsaws)
        atr_target_mult=4.0,    # target = entry + 4×ATR  (R:R = 1.6:1)
        trail_activation=0.04,  # only trail once up 4% (let winners breathe)
        trail_pct=0.03,         # trail at 3% below highest close (loose trail)
        min_hold_bars=3,        # must hold at least 3 bars before stop checks
        risk_per_trade=0.02,    # risk 2% of account per trade
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow)
        self.ema_trend = bt.indicators.EMA(self.data.close, period=self.p.ema_trend)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.highest_since_entry = None
        self.bars_in_trade = 0
        self.trades_executed = 0
        self.winning_trades = 0
        self.trade_returns = []  # per-trade return %

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f"{dt.isoformat()} {txt}")

    def next(self):
        if self.position:
            self._check_exit_signals()
        else:
            self._check_entry_signals()

    def _in_uptrend(self):
        """
        Multi-timeframe trend check:
          - Price above EMA(200)   (long-term uptrend)
          - 3-month momentum positive: close > close 63 bars ago
        Both must be true to avoid entering stocks in broad downtrends.
        """
        if len(self.data) < 64:
            return False
        trend_ok = self.data.close[0] > self.ema_trend[0]
        momentum_ok = self.data.close[0] > self.data.close[-63]
        return trend_ok and momentum_ok

    def _check_entry_signals(self):
        if not self._in_uptrend():
            return

        # Signal A: EMA(20) crosses above EMA(50) with RSI between 50-70
        #           (momentum breakout, but not already overbought)
        ema_crossover = (
            self.ema_fast[0] > self.ema_slow[0] and
            self.ema_fast[-1] <= self.ema_slow[-1] and
            50 < self.rsi[0] < 70
        )

        # Signal B: RSI dip-and-recovery (pullback buy in uptrend)
        rsi_recovery = (
            self.rsi[-1] < self.p.rsi_dip_entry and
            self.rsi[0] >= self.p.rsi_dip_entry and
            self.data.close[0] > self.ema_fast[0]
        )

        if not (ema_crossover or rsi_recovery):
            return

        price = self.data.close[0]
        atr = self.atr[0]
        if atr <= 0:
            return

        stop_distance = self.p.atr_stop_mult * atr
        stop = price - stop_distance
        target = price + self.p.atr_target_mult * atr

        # Position size: risk 2% of account on this stop distance
        account_value = self.broker.getvalue()
        risk_amount = account_value * self.p.risk_per_trade
        risk_based_size = int(risk_amount / stop_distance)

        # Never use more than 90% of available cash
        max_affordable = int(self.broker.getcash() * 0.9 / price)
        position_size = min(risk_based_size, max_affordable)

        if position_size <= 0:
            return

        signal = "EMA-X" if ema_crossover else "RSI-DIP"
        self.buy(size=position_size)
        self.entry_price = price
        self.stop_price = stop
        self.target_price = target
        self.highest_since_entry = price
        self.bars_in_trade = 0
        self.trades_executed += 1
        self.log(
            f"BUY [{signal}] {position_size} @ ${price:.2f}  "
            f"stop=${stop:.2f}  target=${target:.2f}  "
            f"RSI={self.rsi[0]:.1f}  ATR={atr:.2f}"
        )

    def _check_exit_signals(self):
        if self.entry_price is None or not self.position:
            return

        self.bars_in_trade += 1
        price = self.data.close[0]

        # Update trailing stop once trade is profitable enough
        if price > self.highest_since_entry:
            self.highest_since_entry = price

        pnl_pct = (price - self.entry_price) / self.entry_price
        if pnl_pct >= self.p.trail_activation:
            trail_stop = self.highest_since_entry * (1 - self.p.trail_pct)
            if trail_stop > self.stop_price:
                self.stop_price = trail_stop

        # Don't check stop during minimum hold period (let trade settle)
        if self.bars_in_trade < self.p.min_hold_bars:
            return

        # Check exits
        if price >= self.target_price:
            self._exit(price, "PROFIT")
        elif price <= self.stop_price:
            self._exit(price, "STOP")

    def _exit(self, price, reason):
        pnl_pct = (price - self.entry_price) / self.entry_price * 100
        self.close()
        self.log(f"SELL ({reason}) @ ${price:.2f}  P&L={pnl_pct:+.1f}%")
        self.trade_returns.append(pnl_pct)
        if pnl_pct > 0:
            self.winning_trades += 1
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.highest_since_entry = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"Trade closed: Gross=${trade.pnl:.2f}  Net=${trade.pnlcomm:.2f}")


class BacktestRunner:
    """Run backtests"""

    def __init__(self, initial_cash: float = 20000):
        self.initial_cash = initial_cash

    def run_backtest(
        self,
        symbol: str,
        data: pd.DataFrame,
        strategy_class=MrTraderStrategy,
        **strategy_params,
    ) -> Dict[str, Any]:
        """Run backtest for a single symbol. Returns performance metrics."""
        logger.info(f"Running backtest for {symbol}...")

        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy_class, **strategy_params)

        data_feed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(data_feed)

        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=0.001)  # 0.1%

        # Attach an analyser to track the equity curve
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

        results = cerebro.run()
        strategy = results[0]

        final_value = cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash

        # Strategy Sharpe: only from non-zero return days (excludes flat cash days)
        eq_returns = pd.Series(strategy.analyzers.time_return.get_analysis())
        active_returns = eq_returns[eq_returns != 0]
        sharpe = calculate_sharpe_ratio(active_returns) if len(active_returns) > 5 else 0.0

        # Max drawdown from analyser
        dd_info = strategy.analyzers.drawdown.get_analysis()
        max_dd = -(dd_info.get("max", {}).get("drawdown", 0) / 100)

        close = data["close"]

        n_closed = len(strategy.trade_returns)
        win_rate = (strategy.winning_trades / n_closed * 100) if n_closed > 0 else 0.0
        avg_trade = (sum(strategy.trade_returns) / n_closed) if n_closed > 0 else 0.0

        metrics = {
            "symbol": symbol,
            "initial_cash": self.initial_cash,
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return * 100, 2),
            "trades_executed": strategy.trades_executed,
            "closed_trades": n_closed,
            "win_rate_pct": round(win_rate, 1),
            "avg_trade_return_pct": round(avg_trade, 2),
            "benchmark_return_pct": round(
                ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100, 2
            ),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
        }

        logger.info(f"Backtest complete for {symbol}. Return: {metrics['total_return_pct']:.2f}%")
        return metrics

    def run_backtest_portfolio(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        **strategy_params,
    ) -> Dict[str, Any]:
        """Run backtest for a portfolio of symbols. Returns summary metrics."""
        results = {}

        for symbol, data in symbols_data.items():
            try:
                result = self.run_backtest(symbol, data, **strategy_params)
                results[symbol] = result
            except Exception as e:
                logger.error(f"Error backtesting {symbol}: {e}")

        if not results:
            return {"symbols": 0, "results": {}}

        avg_return = sum(r["total_return_pct"] for r in results.values()) / len(results)
        total_trades = sum(r["trades_executed"] for r in results.values())

        # Aggregate win rate and avg trade from all closed trades
        all_wins = sum(r["win_rate_pct"] * r["closed_trades"] / 100 for r in results.values())
        all_closed = sum(r["closed_trades"] for r in results.values())
        agg_win_rate = (all_wins / all_closed * 100) if all_closed > 0 else 0.0
        agg_avg_trade = (
            sum(r["avg_trade_return_pct"] * r["closed_trades"] for r in results.values())
            / all_closed
        ) if all_closed > 0 else 0.0

        # Sharpe from stocks with enough trades to be meaningful (≥ 3 closed trades)
        valid_sharpes = [r["sharpe_ratio"] for r in results.values() if r["closed_trades"] >= 3]
        avg_sharpe = sum(valid_sharpes) / len(valid_sharpes) if valid_sharpes else 0.0

        symbols_positive = sum(1 for r in results.values() if r["total_return_pct"] > 0)

        summary = {
            "symbols": len(results),
            "symbols_positive": symbols_positive,
            "average_return_pct": round(avg_return, 2),
            "aggregate_win_rate_pct": round(agg_win_rate, 1),
            "aggregate_avg_trade_pct": round(agg_avg_trade, 2),
            "average_sharpe_ratio": round(avg_sharpe, 3),
            "total_trades": total_trades,
            "total_closed_trades": all_closed,
            "results": results,
        }

        return summary


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate annualised Sharpe ratio"""
    if returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    return float((excess_returns.mean() / excess_returns.std()) * (252 ** 0.5))


def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown (negative value)"""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return float(drawdown.min())
