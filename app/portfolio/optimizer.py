"""
Portfolio weight optimizer using PyPortfolioOpt.

Used by the portfolio manager to size positions optimally rather than
using fixed equal-weight allocation.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def optimize_weights(
    symbols: List[str],
    prices_df: pd.DataFrame,
    method: str = "max_sharpe",
    risk_free_rate: float = 0.05,
    weight_bounds: tuple = (0.0, 0.15),
) -> Dict[str, float]:
    """
    Compute optimal portfolio weights.

    Args:
        symbols:        Ticker list (must match prices_df columns).
        prices_df:      Daily close prices, columns = symbols, index = dates.
        method:         "max_sharpe" | "min_volatility" | "equal_weight" | "hrp"
        risk_free_rate: Annual rate (e.g. 0.05 for 5%).
        weight_bounds:  (min, max) weight per asset.

    Returns:
        Dict {symbol: weight}, weights sum to 1.0.
    """
    if method == "equal_weight" or len(symbols) < 2:
        w = 1.0 / len(symbols)
        return {s: round(w, 6) for s in symbols}

    try:
        from pypfopt import EfficientFrontier, HRPOpt, expected_returns, risk_models

        # Require at least 30 days of data
        if len(prices_df) < 30:
            logger.warning("Insufficient price history (%d days), using equal weight", len(prices_df))
            return {s: round(1.0 / len(symbols), 6) for s in symbols}

        if method == "hrp":
            returns_df = prices_df.pct_change().dropna()
            hrp = HRPOpt(returns_df)
            hrp.optimize()
            weights = hrp.clean_weights()
            return dict(weights)

        mu = expected_returns.mean_historical_return(prices_df)
        S = risk_models.sample_cov(prices_df)
        ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)

        if method == "max_sharpe":
            ef.max_sharpe(risk_free_rate=risk_free_rate)
        elif method == "min_volatility":
            ef.min_volatility()
        else:
            ef.max_sharpe(risk_free_rate=risk_free_rate)

        weights = ef.clean_weights()
        return dict(weights)

    except Exception as exc:
        logger.warning("PyPortfolioOpt failed (%s), falling back to equal weight", exc)
        return {s: round(1.0 / len(symbols), 6) for s in symbols}


def portfolio_metrics(
    weights: Dict[str, float],
    prices_df: pd.DataFrame,
    risk_free_rate: float = 0.05,
) -> Dict[str, Optional[float]]:
    """Expected annual return, volatility and Sharpe for a given weight dict."""
    try:
        from pypfopt import expected_returns, risk_models

        mu = expected_returns.mean_historical_return(prices_df)
        S = risk_models.sample_cov(prices_df)

        w = np.array([weights.get(c, 0.0) for c in prices_df.columns])
        exp_ret = float(mu.values @ w)
        variance = float(w @ S.values @ w)
        vol = float(np.sqrt(max(variance, 0.0)))
        sharpe = (exp_ret - risk_free_rate) / vol if vol > 0 else 0.0

        return {
            "expected_annual_return_pct": round(exp_ret * 100, 3),
            "annual_volatility_pct": round(vol * 100, 3),
            "sharpe_ratio": round(sharpe, 4),
        }
    except Exception as exc:
        logger.warning("portfolio_metrics failed: %s", exc)
        return {"expected_annual_return_pct": None, "annual_volatility_pct": None, "sharpe_ratio": None}


def fetch_prices_for_symbols(symbols: List[str], period: str = "1y") -> pd.DataFrame:
    """Download daily close prices for a list of symbols via yfinance."""
    import yfinance as yf

    df = yf.download(symbols, period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(name=symbols[0])
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    return df
