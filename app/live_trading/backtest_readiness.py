"""
Backtest-based go/no-go criteria for enabling paper trading.

Evaluates the trained ML models against historical data and returns
structured pass/fail results that feed into the paper trading decision.

Thresholds (conservative for first paper trading run):
  - Sharpe ratio  >= 0.5   (better than random, some edge)
  - Win rate      >= 0.45  (not necessarily > 50%; profitable with good R:R)
  - Max drawdown  <= 15%   (acceptable simulated drawdown)
  - Profit factor >= 1.0   (gross wins > gross losses)
  - Min trades    >= 30    (enough samples to be statistically meaningful)
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Go/no-go thresholds
MIN_SHARPE = 0.5
MIN_WIN_RATE = 0.45
MAX_DRAWDOWN = 0.15
MIN_PROFIT_FACTOR = 1.0
MIN_TRADES = 30


class BacktestCheckResult:
    def __init__(self, name: str, passed: bool, value: Any, detail: str, is_warning: bool = False):
        self.name = name
        self.passed = passed
        self.value = value
        self.detail = detail
        self.is_warning = is_warning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.name,
            "passed": self.passed,
            "value": self.value,
            "detail": self.detail,
            "is_warning": self.is_warning,
        }


class BacktestReadinessChecker:
    """
    Run backtests for swing and/or intraday models and evaluate
    whether results meet the minimum bar for paper trading.
    """

    def __init__(
        self,
        min_sharpe: float = MIN_SHARPE,
        min_win_rate: float = MIN_WIN_RATE,
        max_drawdown: float = MAX_DRAWDOWN,
        min_profit_factor: float = MIN_PROFIT_FACTOR,
        min_trades: int = MIN_TRADES,
    ):
        self.min_sharpe = min_sharpe
        self.min_win_rate = min_win_rate
        self.max_drawdown = max_drawdown
        self.min_profit_factor = min_profit_factor
        self.min_trades = min_trades

    def evaluate(
        self,
        swing_result=None,     # BacktestResult or None
        intraday_result=None,  # BacktestResult or None
    ) -> Dict[str, Any]:
        """
        Evaluate backtest results against go/no-go criteria.

        Returns a dict with:
          - ready: bool
          - blockers: list of failed non-warning checks
          - warnings: list of warning-level issues
          - passed: list of passed checks
          - model_results: summary dict per model
        """
        checks: List[BacktestCheckResult] = []

        if swing_result is not None:
            checks.extend(self._evaluate_model(swing_result, "swing"))

        if intraday_result is not None:
            checks.extend(self._evaluate_model(intraday_result, "intraday"))

        if not checks:
            checks.append(BacktestCheckResult(
                "backtest_data_available", False, None,
                "No backtest results provided — run scripts/backtest_ml_models.py first",
            ))

        blockers = [c for c in checks if not c.passed and not c.is_warning]
        warnings = [c for c in checks if not c.passed and c.is_warning]
        passed = [c for c in checks if c.passed]

        model_results = {}
        if swing_result is not None:
            model_results["swing"] = swing_result.summary()
        if intraday_result is not None:
            model_results["intraday"] = intraday_result.summary()

        return {
            "ready": len(blockers) == 0,
            "blockers": [c.to_dict() for c in blockers],
            "warnings": [c.to_dict() for c in warnings],
            "passed": [c.to_dict() for c in passed],
            "all_checks": [c.to_dict() for c in checks],
            "model_results": model_results,
        }

    def _evaluate_model(self, result, model_name: str) -> List[BacktestCheckResult]:
        checks = []
        pfx = model_name

        # 1. Minimum trade count (statistical significance)
        n = result.total_trades
        passed = n >= self.min_trades
        checks.append(BacktestCheckResult(
            f"{pfx}_min_trades", passed, n,
            f"{model_name}: {n} trades ({'≥' if passed else '<'} {self.min_trades} required)",
            is_warning=not passed,  # not enough trades is a warning, not a hard block
        ))

        if n == 0:
            # Can't evaluate any other metrics without trades
            return checks

        # 2. Sharpe ratio
        s = result.sharpe_ratio
        passed = s >= self.min_sharpe
        checks.append(BacktestCheckResult(
            f"{pfx}_sharpe", passed, round(s, 3),
            f"{model_name}: Sharpe {s:.3f} ({'≥' if passed else '<'} {self.min_sharpe})",
        ))

        # 3. Win rate
        wr = result.win_rate
        passed = wr >= self.min_win_rate
        checks.append(BacktestCheckResult(
            f"{pfx}_win_rate", passed, round(wr, 4),
            f"{model_name}: Win rate {wr:.1%} ({'≥' if passed else '<'} {self.min_win_rate:.0%})",
        ))

        # 4. Max drawdown
        dd = result.max_drawdown_pct
        passed = dd <= self.max_drawdown
        checks.append(BacktestCheckResult(
            f"{pfx}_max_drawdown", passed, round(dd, 4),
            f"{model_name}: Max drawdown {dd:.1%} ({'≤' if passed else '>'} "
            f"{self.max_drawdown:.0%})",
        ))

        # 5. Profit factor
        pf = result.profit_factor
        passed = pf >= self.min_profit_factor
        checks.append(BacktestCheckResult(
            f"{pfx}_profit_factor", passed, round(pf, 3),
            f"{model_name}: Profit factor {pf:.3f} ({'≥' if passed else '<'} "
            f"{self.min_profit_factor})",
        ))

        return checks


def run_quick_backtest(
    model_name: str,
    symbols: Optional[List[str]] = None,
    years: int = 2,
    days: int = 55,
):
    """
    Run a quick backtest for one model type and return BacktestResult.
    Used by the review script without needing to download data separately.
    """
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta

    if symbols is None:
        from app.utils.constants import SP_100_TICKERS
        symbols = SP_100_TICKERS[:20]

    model = _load_model(model_name)
    if model is None:
        logger.warning("No trained %s model found", model_name)
        return None

    if model_name == "swing":
        from app.backtesting.swing_backtest import SwingBacktester
        end = datetime.now()
        start = end - timedelta(days=365 * years + 100)
        try:
            raw = yf.download(
                symbols, start=start.date().isoformat(), end=end.date().isoformat(),
                interval="1d", progress=False, auto_adjust=True, group_by="ticker",
            )
        except Exception as exc:
            logger.error("Download failed: %s", exc)
            return None

        symbols_data = {}
        for sym in symbols:
            try:
                df = raw[sym].copy() if len(symbols) > 1 else raw.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                df = df.dropna(subset=["close"])
                if len(df) >= 50:
                    symbols_data[sym] = df
            except Exception:
                pass

        bt = SwingBacktester(model=model)
        return bt.run(symbols_data, fetch_fundamentals=False)

    elif model_name == "intraday":
        from app.backtesting.intraday_backtest import IntradayBacktester
        end = datetime.now()
        start = end - timedelta(days=days + 5)

        symbols_data = {}
        for sym in symbols[:15]:  # fewer symbols for 5-min data
            try:
                df = yf.download(
                    sym, start=start.isoformat(), end=end.isoformat(),
                    interval="5m", progress=False, auto_adjust=True,
                )
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                if len(df) >= 12:
                    symbols_data[sym] = df
            except Exception:
                pass

        bt = IntradayBacktester(model=model)
        return bt.run(symbols_data)

    return None


def _load_model(model_name: str):
    import os
    try:
        from app.database.models import ModelVersion
        from app.database.session import get_session
        from app.ml.model import PortfolioSelectorModel

        db = get_session()
        try:
            latest = (
                db.query(ModelVersion)
                .filter_by(model_name=model_name, status="ACTIVE")
                .order_by(ModelVersion.version.desc())
                .first()
            )
            if latest and latest.model_path:
                m = PortfolioSelectorModel()
                directory = os.path.dirname(latest.model_path)
                m.load(directory, latest.version, model_name=model_name)
                return m
        finally:
            db.close()
    except Exception as exc:
        logger.debug("Could not load %s model: %s", model_name, exc)
    return None
