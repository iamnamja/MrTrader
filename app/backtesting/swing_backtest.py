"""
Swing model backtester.

For each (symbol, window) in historical data:
  1. Use the trained swing model to score the symbol at the window start
  2. If score >= threshold, simulate a trade:
     - Entry: close price at window_end
     - Exit: whichever comes first over the next HOLD_DAYS bars:
         a. Price hits profit_target (+TARGET_PCT)
         b. Price hits stop_loss (-STOP_PCT)
         c. MAX_HOLD_DAYS reached → close at market
  3. Collect all trades → compute metrics

This is a walk-forward simulation: the model is scored on each window using
the same rolling-window features it was trained on, then the simulated trades
play out in the subsequent HOLD_DAYS period.
"""

import logging
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.backtesting.metrics import BacktestResult, Trade
from app.ml.features import FeatureEngineer
from app.ml.model import PortfolioSelectorModel
from app.ml.training import WINDOW_DAYS, FORWARD_DAYS, STEP_DAYS
from app.utils.constants import SECTOR_MAP

logger = logging.getLogger(__name__)

TARGET_PCT = 0.05      # 5% profit target
STOP_PCT = 0.02        # 2% stop loss
MAX_HOLD_DAYS = 10     # maximum 10 trading days (~2 weeks)
MIN_CONFIDENCE = 0.55  # only trade when model >= this threshold
POSITION_SIZE = 1_000  # fixed $1,000 per trade for P&L calculation


class SwingBacktester:
    """
    Walk-forward backtest for the swing ML model on daily bars.
    """

    def __init__(
        self,
        model: Optional[PortfolioSelectorModel] = None,
        min_confidence: float = MIN_CONFIDENCE,
    ):
        self.model = model
        self.feature_engineer = FeatureEngineer()
        self.min_confidence = min_confidence

    def run(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        fetch_fundamentals: bool = False,
    ) -> BacktestResult:
        """
        Run the swing backtest on historical daily data.

        Args:
            symbols_data: dict of symbol → daily OHLCV DataFrame
            fetch_fundamentals: pass to feature engineer (False for speed)

        Returns:
            BacktestResult with per-trade details and aggregate metrics.
        """
        if self.model is None or not self.model.is_trained:
            logger.warning("No trained swing model — returning empty result")
            return BacktestResult(model_type="swing")

        all_dates = sorted(set.intersection(
            *[set(df.index.date) for df in symbols_data.values()]
        ))
        if len(all_dates) < WINDOW_DAYS + FORWARD_DAYS + MAX_HOLD_DAYS:
            logger.warning("Insufficient data for swing backtest")
            return BacktestResult(model_type="swing")

        window_starts = list(range(
            0,
            len(all_dates) - WINDOW_DAYS - FORWARD_DAYS - MAX_HOLD_DAYS,
            STEP_DAYS,
        ))

        trades: List[Trade] = []

        for w_start_idx in window_starts:
            w_end_idx = w_start_idx + WINDOW_DAYS
            w_end_date = all_dates[w_end_idx]

            # Score all symbols at this window — batch predict for LambdaRank
            symbol_feats: Dict[str, list] = {}
            for symbol, df in symbols_data.items():
                idx = df.index.date
                window_df = df.loc[(idx >= all_dates[w_start_idx]) & (idx <= w_end_date)]
                if len(window_df) < FeatureEngineer.MIN_BARS:
                    continue
                sector = SECTOR_MAP.get(symbol)
                try:
                    feats = self.feature_engineer.engineer_features(
                        symbol, window_df, sector=sector,
                        fetch_fundamentals=fetch_fundamentals,
                        as_of_date=w_end_date,
                    )
                except Exception:
                    continue
                if feats is not None:
                    symbol_feats[symbol] = feats  # keep as dict for named alignment

            scores: Dict[str, float] = {}
            if symbol_feats:
                sym_list = list(symbol_feats.keys())
                model_feat_names = getattr(self.model, "feature_names", None)
                try:
                    if model_feat_names:
                        X_batch = np.array([
                            [symbol_feats[s].get(f, 0.0) for f in model_feat_names]
                            for s in sym_list
                        ])
                    else:
                        X_batch = np.array([list(symbol_feats[s].values()) for s in sym_list])
                    _, probas = self.model.predict(X_batch)
                    for sym, proba in zip(sym_list, probas):
                        scores[sym] = float(proba)
                except Exception:
                    pass

            # Simulate trades for symbols above threshold
            for symbol, score in scores.items():
                if score < self.min_confidence:
                    continue

                df = symbols_data[symbol]
                idx = df.index.date

                entry_rows = df.loc[idx == w_end_date]
                if len(entry_rows) == 0:
                    continue
                entry_price = float(entry_rows["close"].iloc[0])

                target = entry_price * (1 + TARGET_PCT)
                stop = entry_price * (1 - STOP_PCT)

                # Simulate forward HOLD_DAYS
                trade = self._simulate_swing_trade(
                    symbol, df, w_end_date, all_dates, w_end_idx,
                    entry_price, target, stop, score,
                )
                if trade:
                    trades.append(trade)

        logger.info("Swing backtest: %d trades simulated", len(trades))
        return BacktestResult.from_trades(trades, model_type="swing")

    def _simulate_swing_trade(
        self,
        symbol: str,
        df: pd.DataFrame,
        entry_date: date,
        all_dates: list,
        entry_idx: int,
        entry_price: float,
        target: float,
        stop: float,
        confidence: float,
    ) -> Optional[Trade]:
        idx = df.index.date
        quantity = max(1, int(POSITION_SIZE / entry_price))
        exit_price = entry_price
        exit_reason = "MAX_HOLD"
        hold_bars = 0

        for bar_offset in range(1, MAX_HOLD_DAYS + 1):
            future_idx = entry_idx + bar_offset
            if future_idx >= len(all_dates):
                break
            future_date = all_dates[future_idx]
            bar_rows = df.loc[idx == future_date]
            if len(bar_rows) == 0:
                continue

            high = float(bar_rows["high"].iloc[0])
            low = float(bar_rows["low"].iloc[0])
            close = float(bar_rows["close"].iloc[0])
            hold_bars = bar_offset

            if low <= stop:
                exit_price = stop
                exit_reason = "STOP"
                break
            if high >= target:
                exit_price = target
                exit_reason = "TARGET"
                break
            exit_price = close  # update in case we run out

        pnl = (exit_price - entry_price) * quantity
        pnl_pct = (exit_price - entry_price) / entry_price

        return Trade(
            symbol=symbol,
            entry_date=entry_date,
            exit_date=all_dates[min(entry_idx + hold_bars, len(all_dates) - 1)],
            entry_price=round(entry_price, 4),
            exit_price=round(exit_price, 4),
            quantity=quantity,
            pnl=round(pnl, 4),
            pnl_pct=round(pnl_pct, 6),
            hold_bars=hold_bars,
            exit_reason=exit_reason,
            trade_type="swing",
        )
