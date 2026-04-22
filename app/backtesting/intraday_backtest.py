"""
Intraday model backtester.

For each (symbol, trading_day):
  1. Use the trained intraday model to score the symbol at 09:45 (after 15 min of data)
  2. If score >= threshold, simulate a trade:
     - Entry: close of the 3rd bar (09:45 price)
     - Exit: whichever comes first:
         a. Price hits profit_target (+TARGET_PCT)
         b. Price hits stop_loss (-STOP_PCT)
         c. 3:45 PM force-close
  3. Collect all trades → compute metrics
"""

import logging
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.backtesting.metrics import BacktestResult, Trade
from app.ml.intraday_features import compute_intraday_features
from app.ml.model import PortfolioSelectorModel

logger = logging.getLogger(__name__)

TARGET_PCT = 0.01      # 1% profit target (realistic for intraday)
STOP_PCT = 0.005       # 0.5% stop loss
FEATURE_BARS = 12      # 1 hour of 5-min bars (matches MIN_BARS in intraday_features)
FORCE_CLOSE_BAR = 72   # 6 hours × 12 bars/hr = ~3:45 PM equivalent
MIN_CONFIDENCE = 0.55
POSITION_SIZE = 1_000  # fixed $1,000 per trade


class IntradayBacktester:
    """
    Walk-forward backtest for the intraday ML model on 5-minute bars.
    """

    def __init__(
        self,
        model: Optional[PortfolioSelectorModel] = None,
        min_confidence: float = MIN_CONFIDENCE,
    ):
        self.model = model
        self.min_confidence = min_confidence

    def run(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        spy_data: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Run the intraday backtest on historical 5-min data.

        Args:
            symbols_data: dict of symbol → 5-min OHLCV DataFrame (multi-day)
            spy_data:     optional SPY 5-min data for benchmark features

        Returns:
            BacktestResult with per-trade details and aggregate metrics.
        """
        if self.model is None or not self.model.is_trained:
            logger.warning("No trained intraday model — returning empty result")
            return BacktestResult(model_type="intraday")

        # Collect all trading days
        all_days: set = set()
        for df in symbols_data.values():
            if df is not None and len(df) > 0:
                idx = pd.DatetimeIndex(df.index)
                for d in idx.normalize().unique():
                    all_days.add(d.date())

        sorted_days = sorted(all_days)
        trades: List[Trade] = []

        for sym, df in symbols_data.items():
            if df is None or len(df) == 0:
                continue
            df_idx = pd.DatetimeIndex(df.index)

            for day in sorted_days:
                day_mask = df_idx.normalize().date == day
                day_bars = df.loc[day_mask]

                if len(day_bars) < FEATURE_BARS + 2:
                    continue

                # Feature window: first FEATURE_BARS bars (≈ 09:45)
                feat_bars = day_bars.iloc[:FEATURE_BARS]
                spy_day = self._get_spy_day(spy_data, day)
                prior_close, prior_day_high, prior_day_low = self._prior_day_ohlc(
                    df, df_idx, day
                )

                feats = compute_intraday_features(
                    feat_bars, spy_day, prior_close,
                    prior_day_high=prior_day_high,
                    prior_day_low=prior_day_low,
                )
                if feats is None:
                    continue

                X = np.array([list(feats.values())])
                try:
                    _, proba = self.model.predict(X)
                    score = float(proba[0])
                except Exception:
                    continue

                if score < self.min_confidence:
                    continue

                entry_price = float(feat_bars["close"].iloc[-1])
                target = entry_price * (1 + TARGET_PCT)
                stop = entry_price * (1 - STOP_PCT)

                future_bars = day_bars.iloc[FEATURE_BARS:]
                trade = self._simulate_intraday_trade(
                    sym, day, entry_price, target, stop, future_bars, score,
                )
                if trade:
                    trades.append(trade)

        logger.info("Intraday backtest: %d trades simulated", len(trades))
        return BacktestResult.from_trades(trades, model_type="intraday")

    def _simulate_intraday_trade(
        self,
        symbol: str,
        day: date,
        entry_price: float,
        target: float,
        stop: float,
        future_bars: pd.DataFrame,
        confidence: float,
    ) -> Optional[Trade]:
        quantity = max(1, int(POSITION_SIZE / entry_price))
        exit_price = entry_price
        exit_reason = "FORCE_CLOSE"
        hold_bars = 0

        for bar_offset, (_, bar) in enumerate(future_bars.iterrows()):
            if bar_offset >= FORCE_CLOSE_BAR:
                exit_reason = "FORCE_CLOSE"
                break

            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
            hold_bars = bar_offset + 1

            if low <= stop:
                exit_price = stop
                exit_reason = "STOP"
                break
            if high >= target:
                exit_price = target
                exit_reason = "TARGET"
                break
            exit_price = close

        pnl = (exit_price - entry_price) * quantity
        pnl_pct = (exit_price - entry_price) / entry_price

        return Trade(
            symbol=symbol,
            entry_date=day,
            exit_date=day,
            entry_price=round(entry_price, 4),
            exit_price=round(exit_price, 4),
            quantity=quantity,
            pnl=round(pnl, 4),
            pnl_pct=round(pnl_pct, 6),
            hold_bars=hold_bars,
            exit_reason=exit_reason,
            trade_type="intraday",
        )

    def _get_spy_day(
        self, spy_data: Optional[pd.DataFrame], day: date
    ) -> Optional[pd.DataFrame]:
        if spy_data is None or len(spy_data) == 0:
            return None
        spy_idx = pd.DatetimeIndex(spy_data.index)
        mask = spy_idx.normalize().date == day
        result = spy_data.loc[mask]
        return result if len(result) > 0 else None

    def _prior_day_ohlc(
        self, df: pd.DataFrame, df_idx: pd.DatetimeIndex, day: date
    ) -> tuple:
        """Return (prior_close, prior_day_high, prior_day_low) for the session before `day`."""
        prior_mask = df_idx.normalize().date < day
        if not prior_mask.any():
            return None, None, None
        prior_bars = df.loc[prior_mask]
        last_day = prior_bars.index.normalize().date[-1]
        last_day_bars = prior_bars.loc[prior_bars.index.normalize().date == last_day]
        if len(last_day_bars) == 0:
            return None, None, None
        return (
            float(last_day_bars["close"].iloc[-1]),
            float(last_day_bars["high"].max()),
            float(last_day_bars["low"].min()),
        )
