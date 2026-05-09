"""Phase 2d — Walk-forward predicted P&L helper.

Fetches the most recent ACTIVE ModelVersion for swing/intraday and extracts
average per-trade P&L from the walk-forward fold stats stored in
ModelVersion.performance.  Result is cached in memory for TTL seconds so
the PM does not hit the DB on every trade decision.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# --- cache -----------------------------------------------------------------
_cache: dict[str, tuple[float, float]] = {}   # key -> (predicted_pnl, expires_at)
_TTL = 3600.0   # 1 hour — model versions change infrequently


def _cache_key(trade_type: str) -> str:
    return f"wf_predicted_pnl:{trade_type}"


def _model_name(trade_type: str) -> str:
    return "intraday_selector" if trade_type == "intraday" else "portfolio_selector"


def get_predicted_pnl(trade_type: str, db_session=None) -> Optional[float]:
    """Return the walk-forward average per-trade P&L (in $ per trade, as a %)
    for the active model of *trade_type* ("swing" | "intraday").

    Returns None when no ACTIVE model is found or the performance blob lacks
    the expected fields.  Callers should treat None as "data unavailable" and
    store NULL in Trade.sim_predicted_pnl.
    """
    key = _cache_key(trade_type)
    now = time.monotonic()
    if key in _cache:
        value, expires = _cache[key]
        if now < expires:
            return value

    try:
        predicted = _fetch_from_db(trade_type, db_session)
    except Exception as exc:
        logger.debug("get_predicted_pnl(%s) DB error (non-fatal): %s", trade_type, exc)
        predicted = None

    # Cache even None so we don't hammer DB on every trade when model is missing
    _cache[key] = (predicted, now + _TTL)
    return predicted


def _fetch_from_db(trade_type: str, db_session=None) -> Optional[float]:
    """Query DB for the latest ACTIVE ModelVersion and extract avg per-trade return."""
    from app.database.models import ModelVersion

    close_session = False
    if db_session is None:
        from app.database.session import SessionLocal
        db_session = SessionLocal()
        close_session = True

    try:
        row = (
            db_session.query(ModelVersion)
            .filter(
                ModelVersion.model_name == _model_name(trade_type),
                ModelVersion.status == "ACTIVE",
            )
            .order_by(ModelVersion.version.desc())
            .first()
        )
        if row is None or not row.performance:
            return None

        perf = row.performance  # JSON dict
        # Walk-forward fold stats may be stored under different key names;
        # try several conventions written by RetainCron / ModelTrainer.
        for key in ("avg_return_per_trade", "avg_trade_return", "mean_return"):
            val = perf.get(key)
            if val is not None:
                return float(val)

        # Fall back: derive from avg_sharpe * avg_vol / sqrt(avg_trades_per_fold)
        # This is an approximation only used when per-trade return isn't stored.
        sharpe = perf.get("avg_sharpe") or perf.get("sharpe")
        n_trades = perf.get("avg_trades") or perf.get("n_trades") or perf.get("total_trades")
        if sharpe is not None and n_trades and float(n_trades) > 0:
            # Annualised Sharpe → daily Sharpe → per-trade %; rough heuristic
            daily_sharpe = float(sharpe) / (252 ** 0.5)
            return round(daily_sharpe / float(n_trades) ** 0.5, 6)

        return None
    finally:
        if close_session:
            db_session.close()


def invalidate_cache(trade_type: Optional[str] = None) -> None:
    """Invalidate the cache, e.g. after a model retrain completes."""
    if trade_type is None:
        _cache.clear()
    else:
        _cache.pop(_cache_key(trade_type), None)
