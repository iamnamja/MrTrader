"""
Intraday model backtester — label-aligned simulation.

Matches the training label scheme:
  - Model trained on: cross-sectional Sharpe-adjusted 24-bar (2h) best return
  - Backtester: score all symbols together each day (cross-sectional, matching
    how PM actually works), pick top-N, enter at bar 12, hold 24 bars with
    stop/target guards, exit at 2h time stop if neither hit.

Exit priority per trade:
  1. Stop loss  (-STOP_PCT)
  2. Profit target (+TARGET_PCT)
  3. 24-bar time exit (2h — matches HOLD_BARS in training)
"""

import logging
from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.backtesting.metrics import BacktestResult, Trade
from app.ml.intraday_features import compute_intraday_features
from app.ml.model import PortfolioSelectorModel

logger = logging.getLogger(__name__)

# ── Constants — must match app/ml/intraday_training.py ───────────────────────
HOLD_BARS    = 24      # 2h of 5-min bars — outer time exit (matches training)
FEATURE_BARS = 12      # 1h of bars used to build features before entry
TARGET_PCT   = 0.005   # 0.5% profit target (matches intraday_training.STOP_PCT/TARGET_PCT)
STOP_PCT     = 0.003   # 0.3% stop loss
TOP_N        = 5       # pick top-N per day by model score (matches PM TOP_N_INTRADAY)
POSITION_SIZE = 1_000  # fixed $1,000 per trade for P&L reporting


class IntradayBacktester:
    """
    Walk-forward backtest for the intraday ML model on 5-minute bars.

    Scores all available symbols together each day (cross-sectional batch),
    picks the top-N by model probability, and simulates a 2-hour trade for
    each. This mirrors how PM selects and how the model was trained.
    """

    def __init__(
        self,
        model: Optional[PortfolioSelectorModel] = None,
        top_n: int = TOP_N,
        min_confidence: float = 0.0,  # legacy param — ignored; use top_n instead
    ):
        self.model = model
        self.top_n = top_n

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

        # Collect all trading days across all symbols
        all_days: set = set()
        for df in symbols_data.values():
            if df is not None and len(df) > 0:
                for d in pd.DatetimeIndex(df.index).normalize().unique():
                    all_days.add(d.date())

        trades: List[Trade] = []

        for day in sorted(all_days):
            day_trades = self._process_day(day, symbols_data, spy_data)
            trades.extend(day_trades)

        logger.info("Intraday backtest: %d trades simulated across %d days",
                    len(trades), len(all_days))
        return BacktestResult.from_trades(trades, model_type="intraday")

    def _process_day(
        self,
        day: date,
        symbols_data: Dict[str, pd.DataFrame],
        spy_data: Optional[pd.DataFrame],
    ) -> List[Trade]:
        """Score all symbols for a given day in one batch, pick top-N, simulate trades."""
        spy_day = self._get_spy_day(spy_data, day)

        # ── Step 1: build feature vectors for all eligible symbols ────────────
        sym_feats: Dict[str, dict] = {}
        sym_entry: Dict[str, float] = {}
        sym_future: Dict[str, pd.DataFrame] = {}

        for sym, df in symbols_data.items():
            df_idx = pd.DatetimeIndex(df.index)
            day_mask = df_idx.normalize().date == day
            day_bars = df.loc[day_mask]

            if len(day_bars) < FEATURE_BARS + HOLD_BARS:
                continue  # need enough bars for features + full hold window

            feat_bars = day_bars.iloc[:FEATURE_BARS]
            prior_close, prior_high, prior_low = self._prior_day_ohlc(df, df_idx, day)

            feats = compute_intraday_features(
                feat_bars, spy_day, prior_close,
                prior_day_high=prior_high,
                prior_day_low=prior_low,
            )
            if feats is None:
                continue

            sym_feats[sym] = feats
            sym_entry[sym] = float(feat_bars["close"].iloc[-1])
            sym_future[sym] = day_bars.iloc[FEATURE_BARS: FEATURE_BARS + HOLD_BARS]

        if not sym_feats:
            return []

        # ── Step 2: batch predict (cross-sectional — matches training) ─────────
        sym_list = list(sym_feats.keys())
        model_feat_names = getattr(self.model, "feature_names", None)
        try:
            if model_feat_names:
                X = np.array([
                    [sym_feats[s].get(f, 0.0) for f in model_feat_names]
                    for s in sym_list
                ])
            else:
                X = np.array([list(sym_feats[s].values()) for s in sym_list])
            _, probas = self.model.predict(X)
        except Exception as exc:
            logger.debug("Predict failed on %s: %s", day, exc)
            return []

        # ── Step 3: pick top-N by score ───────────────────────────────────────
        scored = sorted(zip(sym_list, probas), key=lambda x: x[1], reverse=True)
        selected = scored[: self.top_n]

        # ── Step 4: simulate each trade ───────────────────────────────────────
        trades: List[Trade] = []
        for sym, score in selected:
            entry = sym_entry[sym]
            target = entry * (1 + TARGET_PCT)
            stop   = entry * (1 - STOP_PCT)
            trade = self._simulate_trade(sym, day, entry, target, stop,
                                         sym_future[sym], float(score))
            if trade:
                trades.append(trade)

        return trades

    def _simulate_trade(
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
        exit_price = float(future_bars["close"].iloc[-1])  # default: 2h close
        exit_reason = "TIME_EXIT"
        hold_bars = len(future_bars)

        for bar_offset, (_, bar) in enumerate(future_bars.iterrows()):
            high  = float(bar["high"])
            low   = float(bar["low"])
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
            exit_price = close  # update running exit to close of each bar

        pnl     = (exit_price - entry_price) * quantity
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
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return (prior_close, prior_high, prior_low) for the session before `day`."""
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
