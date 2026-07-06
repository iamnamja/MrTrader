import logging
import threading
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from app.config import settings

logger = logging.getLogger(__name__)


class _TokenBucket:
    """
    Token-bucket rate limiter.

    Alpaca allows 200 requests/minute.  We cap at 180/min (10% headroom)
    to avoid racing the exact boundary.
    """

    def __init__(self, rate: float = 100, per: float = 60.0):
        self._capacity = rate
        self._tokens = rate
        self._rate = rate / per          # tokens per second
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        """Block until a token is available."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            if self._tokens < 1:
                sleep_for = (1 - self._tokens) / self._rate
            else:
                sleep_for = 0
            self._tokens -= 1

        if sleep_for > 0:
            logger.debug("Rate limit: sleeping %.2fs", sleep_for)
            time.sleep(sleep_for)


_rate_limiter = _TokenBucket()

try:
    from requests.exceptions import ConnectionError as _ReqConnError, Timeout as _ReqTimeout
    _TRANSIENT_ERRORS: tuple = (_ReqConnError, _ReqTimeout)
except Exception:  # pragma: no cover
    _TRANSIENT_ERRORS = ()


def _retry_call(fn, *args, attempts: int = 3, backoff: float = 0.4, **kwargs):
    """Call an idempotent Alpaca READ and retry on TRANSIENT connection errors.

    Alpaca (like any pooled-HTTP REST API) periodically closes idle keep-alive sockets;
    the next request on a stale socket raises requests ConnectionError (wrapping
    RemoteDisconnected). A fresh connection on immediate retry almost always succeeds, so
    these blips should not surface as errors or as fail-closed sleeve skips. Re-raises the
    last error after `attempts` so a genuine outage still fails (and the caller's own
    except/fail-closed path runs). Only for idempotent reads (account/positions/clock/price)."""
    last = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except _TRANSIENT_ERRORS as e:
            last = e
            if i < attempts - 1:
                logger.warning("Alpaca %s transient network error (attempt %d/%d): %s — retrying",
                               getattr(fn, "__name__", "call"), i + 1, attempts, e)
                time.sleep(backoff * (i + 1))
                continue
            raise
    raise last  # pragma: no cover


def _notify_circuit_breaker():
    """Inform the circuit breaker of a network error (best-effort, no circular import)."""
    try:
        from app.agents.circuit_breaker import circuit_breaker
        circuit_breaker.record_network_error()
    except Exception:
        pass


def _err_text(err) -> str:
    """Error message text, NEVER raising. Alpaca's APIError.message can be a property that parses the
    JSON body and raises on a non-JSON body — getattr only swallows AttributeError, so a 5xx HTML
    body would otherwise crash the classifier (and the except handler, bypassing circuit-breaker
    notify in the order path)."""
    try:
        return str(getattr(err, "message", "") or err)
    except Exception:
        try:
            return str(err)
        except Exception:
            return ""


def _err_code(err):
    """Error code/status, NEVER raising (same property-raises-on-non-JSON hazard as _err_text)."""
    try:
        c = getattr(err, "status_code", None)
        if c is None:
            c = getattr(err, "code", None)
        return c
    except Exception:
        return None


def _is_duplicate_client_order_id(err) -> bool:
    """True if an Alpaca error indicates a DUPLICATE client_order_id — i.e. the order we're (re)trying
    to place ALREADY exists (a prior attempt placed it). Conservative: only treat clear duplicate
    messages as such (anything ambiguous falls through to a normal error -> raise)."""
    msg = _err_text(err).lower()
    return "client_order_id" in msg and ("exist" in msg or "unique" in msg or "duplicate" in msg)


class OrderSanityError(Exception):
    """Raised by the H3 pre-trade guard when a single order exceeds the absolute fat-finger caps.

    Distinct from APIError, so it propagates to the caller (which logs + skips the order) and never
    trips the circuit breaker (it's our own refusal to place, not a broker fault)."""


def _assert_order_within_caps(symbol: str, quantity, side: str, price=None) -> None:
    """H3 pre-trade fat-finger backstop — FAIL-CLOSED. Reject (raise OrderSanityError) a single order
    whose share count exceeds H3_MAX_ORDER_SHARES, or whose notional (qty*price, when a price is
    available) exceeds H3_MAX_ORDER_NOTIONAL_USD. A malformed/non-positive qty is itself rejected (a
    malformed order is suspect). Pure arithmetic — no broker/DB/network I/O, so it cannot fail-open
    on an outage. Disabled only via the explicit H3_PRETRADE_CAP_ENABLED flag."""
    from app.ml.retrain_config import (
        H3_PRETRADE_CAP_ENABLED, H3_MAX_ORDER_NOTIONAL_USD, H3_MAX_ORDER_SHARES,
    )
    if not H3_PRETRADE_CAP_ENABLED:
        return
    try:
        qty = int(quantity)
    except (TypeError, ValueError):
        raise OrderSanityError(f"H3: {symbol} {side} qty is not an integer: {quantity!r}")
    if qty <= 0:
        raise OrderSanityError(f"H3: {symbol} {side} non-positive qty {qty}")
    if qty > H3_MAX_ORDER_SHARES:
        raise OrderSanityError(
            f"H3: {symbol} {side} {qty} shares exceeds max-order-size cap {H3_MAX_ORDER_SHARES}")
    try:
        px = float(price) if price is not None else None
    except (TypeError, ValueError):
        px = None
    if px is not None and px > 0:
        notional = qty * px
        if notional > H3_MAX_ORDER_NOTIONAL_USD:
            raise OrderSanityError(
                f"H3: {symbol} {side} notional ${notional:,.0f} ({qty} @ ${px:,.2f}) exceeds "
                f"per-order cap ${H3_MAX_ORDER_NOTIONAL_USD:,.0f}")
    # px None/<=0 (e.g. a market order placed with no est_price): notional not checkable here — the
    # shares cap above still applies as the coarse backstop.


def _is_position_not_found(err) -> bool:
    """True if an Alpaca error means the position is CONFIRMED not to exist (the account is flat in
    that symbol), as opposed to an indeterminate read failure (network/5xx/auth). Only a confirmed
    not-found may be reported as None; an indeterminate failure must NOT be conflated with 'flat'
    (that is the fail-OPEN that lets the entry guard double-buy). Conservative: anything that isn't a
    clear 404/not-found is treated as indeterminate."""
    code = _err_code(err)
    if code in (404, "404", 40410000):  # alpaca "position does not exist" code
        return True
    msg = _err_text(err).lower()
    return "position does not exist" in msg or "position not found" in msg


class AlpacaClient:
    """Alpaca API client wrapper for trading and market data"""

    def __init__(self):
        is_paper = settings.trading_mode.lower() != "live"
        self.trading_client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=is_paper,
        )
        self.data_client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
        logger.info(f"Alpaca client initialized (Mode: {settings.trading_mode})")

    def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            _rate_limiter.acquire()
            account = _retry_call(self.trading_client.get_account)
            return {
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity) if account.last_equity is not None else None,
                "long_market_value": float(account.long_market_value) if account.long_market_value is not None else 0.0,
                "short_market_value": float(account.short_market_value) if account.short_market_value is not None else 0.0,
                "daytrade_count": int(account.daytrade_count) if account.daytrade_count is not None else 0,
                "account_blocked": account.account_blocked,
                "status": account.status,
            }
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            raise

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        try:
            _rate_limiter.acquire()
            positions = _retry_call(self.trading_client.get_all_positions)
            result = []
            for pos in positions:
                # Per-row guard: a single malformed/fractional position must not abort the whole
                # fetch (which would break reconciliation/account snapshots). int(float(...)) handles
                # Alpaca's decimal-string qty (e.g. fractional shares from a corporate action).
                try:
                    result.append({
                        "symbol": pos.symbol,
                        "qty": int(float(pos.qty)),
                        "avg_entry_price": float(pos.avg_entry_price),
                        "market_value": float(pos.market_value),
                        "unrealized_pl": float(pos.unrealized_pl),
                        "unrealized_plpc": float(pos.unrealized_plpc),
                        "current_price": float(pos.current_price),
                    })
                except (TypeError, ValueError) as row_exc:
                    logger.error("Skipping malformed position row %s: %s",
                                 getattr(pos, "symbol", "?"), row_exc)
            return result
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            raise

    def get_position(self, symbol: str, *, raise_on_error: bool = False) -> Optional[Dict[str, Any]]:
        """Get position for a specific symbol.

        Returns None ONLY when the position is CONFIRMED not to exist (flat). On an indeterminate
        read failure (network/5xx/auth), if `raise_on_error` is True the error is re-raised so the
        caller can fail-CLOSED — the anti-duplicate entry guard must NEVER treat 'could not
        determine' as 'flat' (that conflation is the double-buy BLOCKER). Legacy callers
        (raise_on_error=False, default) log and return None to preserve prior behavior.
        """
        try:
            _rate_limiter.acquire()
            position = _retry_call(self.trading_client.get_open_position, symbol)
            return {
                "symbol": position.symbol,
                "qty": int(float(position.qty)),
                "avg_entry_price": float(position.avg_entry_price),
                "market_value": float(position.market_value),
                "unrealized_pl": float(position.unrealized_pl),
                "unrealized_plpc": float(position.unrealized_plpc),
                "current_price": float(position.current_price),
            }
        except Exception as e:
            if _is_position_not_found(e):
                logger.debug(f"Position not found for {symbol}: {e}")
                return None
            # Indeterminate failure — NOT a confirmed flat.
            logger.warning(f"get_position({symbol}) indeterminate read error: {e}")
            if raise_on_error:
                raise
            return None

    def get_portfolio_history(
        self, period: str = "1D", timeframe: str = "5Min"
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch portfolio equity history from Alpaca.

        Returns a dict with `timestamp`, `equity`, `profit_loss`,
        `profit_loss_pct`, `base_value`, `timeframe`, or ``None`` if
        the endpoint is unavailable.
        """
        try:
            from alpaca.trading.requests import GetPortfolioHistoryRequest
            _rate_limiter.acquire()
            req = GetPortfolioHistoryRequest(period=period, timeframe=timeframe)
            h = self.trading_client.get_portfolio_history(req)
            return {
                "timestamp": list(getattr(h, "timestamp", []) or []),
                "equity": [float(x) for x in (getattr(h, "equity", []) or []) if x is not None],
                "profit_loss": [float(x) for x in (getattr(h, "profit_loss", []) or []) if x is not None],
                "profit_loss_pct": [float(x) for x in (getattr(h, "profit_loss_pct", []) or []) if x is not None],
                "base_value": float(getattr(h, "base_value", 0.0) or 0.0),
                "timeframe": getattr(h, "timeframe", timeframe),
            }
        except Exception as e:
            logger.debug(f"Portfolio history unavailable: {e}")
            return None

    def place_market_order(
        self, symbol: str, quantity: int, side: str, client_order_id: str | None = None,
        est_price: float | None = None,
    ) -> Dict[str, Any]:
        """
        Place a market order

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: 'buy' or 'sell'
            client_order_id: Optional identifier we control (proposal UUID); queryable via Alpaca
            est_price: Optional est. fill price so the H3 pre-trade guard can check notional (a
                market order carries no price; callers pass the quote/last they already have).
        """
        # H3 pre-trade fat-finger backstop (fail-closed) — raises OrderSanityError BEFORE any submit;
        # propagates to the caller (logs + skips), never trips the circuit breaker.
        _assert_order_within_caps(symbol, quantity, side, est_price)
        try:
            _rate_limiter.acquire()
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            kwargs = dict(symbol=symbol, qty=quantity, side=order_side, time_in_force=TimeInForce.DAY)
            if client_order_id:
                kwargs["client_order_id"] = client_order_id
            order = self.trading_client.submit_order(MarketOrderRequest(**kwargs))
            logger.info(f"Market order placed: {symbol} {side} {quantity} shares")
            return {
                "order_id": str(order.id),
                "symbol": order.symbol,
                "qty": int(float(order.qty)),
                "side": str(order.side),
                "status": order.status,
                "created_at": order.created_at.isoformat() if order.created_at else None,
                "idempotent_reuse": False,
            }
        except APIError as e:
            # H6 idempotency: a retry of an order we ALREADY placed (same client_order_id) is rejected
            # by Alpaca as a duplicate — that's SUCCESS, not failure. Fetch the existing order and
            # return it, so a crash-retry never double-fills AND never logs a spurious order_error or
            # orphans the position. Not a network fault -> do NOT trip the circuit breaker.
            if client_order_id and _is_duplicate_client_order_id(e):
                try:
                    existing = self.trading_client.get_order_by_client_id(client_order_id)
                    # Verify the existing order MATCHES what we intended before treating it as
                    # success. A client_order_id collision with a different symbol/side would
                    # otherwise record a PHANTOM fill for the wrong order — fail CLOSED instead.
                    if (existing.symbol != symbol
                            or side.lower() not in str(existing.side).lower()):
                        logger.error("Idempotent reuse MISMATCH: existing %s %s vs requested %s %s "
                                     "(coid=%s) — failing closed", existing.symbol, existing.side,
                                     symbol, side, client_order_id)
                        # fall through to the generic error path below (which notifies the circuit
                        # breaker + re-raises) — fail-closed without double-notifying.
                        raise RuntimeError(f"client_order_id {client_order_id} maps to a different "
                                           f"order ({existing.symbol} {existing.side})")
                    # Only a LIVE/filled order counts as "already placed". A coid that maps to a
                    # DEAD order (canceled/expired/rejected — e.g. a re-quote that cancelled then
                    # re-used the same generation key) must NOT be returned as if it were resting,
                    # else the caller books a phantom placement on a dead order. Fail closed.
                    _st = str(getattr(existing, "status", "")).lower()
                    if any(d in _st for d in ("cancel", "expired", "rejected")):
                        logger.error("Idempotent reuse: coid %s maps to a DEAD order (status=%s) — "
                                     "not resting; failing closed", client_order_id, _st)
                        raise RuntimeError(f"client_order_id {client_order_id} maps to a {_st} order")
                    logger.info("Idempotent order reuse: %s %s already placed (client_order_id=%s)",
                                side, symbol, client_order_id)
                    return {
                        "order_id": str(existing.id),
                        "symbol": existing.symbol,
                        "qty": int(float(existing.qty)),
                        "side": str(existing.side),
                        "status": existing.status,
                        "created_at": (existing.created_at.isoformat()
                                       if existing.created_at else None),
                        "idempotent_reuse": True,
                    }
                except Exception as le:
                    logger.error("dup client_order_id %s detected but lookup failed: %s",
                                 client_order_id, le)
            logger.error(f"Error placing market order: {e}")
            _notify_circuit_breaker()
            raise
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            _notify_circuit_breaker()
            raise

    def place_limit_order(
        self, symbol: str, quantity: int, side: str, limit_price: float,
        client_order_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        Place a limit order

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: 'buy' or 'sell'
            limit_price: Limit price
            client_order_id: Optional identifier we control (proposal UUID); queryable via Alpaca
        """
        # H3 pre-trade fat-finger backstop (fail-closed) — limit orders carry their own price.
        _assert_order_within_caps(symbol, quantity, side, limit_price)
        try:
            _rate_limiter.acquire()
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            kwargs = dict(
                symbol=symbol, qty=quantity, side=order_side,
                limit_price=limit_price, time_in_force=TimeInForce.DAY,
            )
            if client_order_id:
                kwargs["client_order_id"] = client_order_id
            order = self.trading_client.submit_order(LimitOrderRequest(**kwargs))
            logger.info(f"Limit order placed: {symbol} {side} {quantity} @ ${limit_price}")
            return {
                "order_id": str(order.id),
                "symbol": order.symbol,
                "qty": int(float(order.qty)),
                "side": str(order.side),
                "limit_price": float(order.limit_price),
                "status": order.status,
                "created_at": order.created_at.isoformat() if order.created_at else None,
                "idempotent_reuse": False,
            }
        except APIError as e:
            # H6 idempotency (mirror place_market_order): a retry of a limit order we ALREADY placed
            # (same client_order_id) is rejected as a duplicate — that is SUCCESS, not failure. Fetch
            # and return the existing order so a lost-response retry never orphans a live limit order
            # (booked as FAILED while resting at the broker). Not a network fault -> no circuit breaker.
            if client_order_id and _is_duplicate_client_order_id(e):
                try:
                    existing = self.trading_client.get_order_by_client_id(client_order_id)
                    if (existing.symbol != symbol
                            or side.lower() not in str(existing.side).lower()):
                        logger.error("Idempotent limit reuse MISMATCH: existing %s %s vs requested "
                                     "%s %s (coid=%s) — failing closed", existing.symbol,
                                     existing.side, symbol, side, client_order_id)
                        # fall through to the generic error path (notifies CB + re-raises) once.
                        raise RuntimeError(f"client_order_id {client_order_id} maps to a different "
                                           f"order ({existing.symbol} {existing.side})")
                    # DEAD-order guard (see place_market_order): a coid mapping to a canceled/expired/
                    # rejected order is NOT a resting order — fail closed rather than book a phantom.
                    _st = str(getattr(existing, "status", "")).lower()
                    if any(d in _st for d in ("cancel", "expired", "rejected")):
                        logger.error("Idempotent limit reuse: coid %s maps to a DEAD order "
                                     "(status=%s) — not resting; failing closed", client_order_id, _st)
                        raise RuntimeError(f"client_order_id {client_order_id} maps to a {_st} order")
                    logger.info("Idempotent limit-order reuse: %s %s already placed (client_order_id=%s)",
                                side, symbol, client_order_id)
                    return {
                        "order_id": str(existing.id),
                        "symbol": existing.symbol,
                        "qty": int(float(existing.qty)),
                        "side": str(existing.side),
                        "limit_price": float(existing.limit_price) if existing.limit_price else limit_price,
                        "status": existing.status,
                        "created_at": (existing.created_at.isoformat()
                                       if existing.created_at else None),
                        "idempotent_reuse": True,
                    }
                except Exception as le:
                    logger.error("dup client_order_id %s on limit order but lookup failed: %s",
                                 client_order_id, le)
            logger.error(f"Error placing limit order: {e}")
            _notify_circuit_breaker()
            raise
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            _notify_circuit_breaker()
            raise

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status"""
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return {
                "order_id": order.id,
                "symbol": order.symbol,
                "qty": int(float(order.qty)),
                "filled_qty": int(float(order.filled_qty)) if order.filled_qty else 0,
                "side": str(order.side),
                "status": order.status,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            }
        except Exception as e:
            logger.error(f"Error fetching order status: {e}")
            return None

    def get_orders(self, limit: int = 100, status: str = "all") -> List[Dict[str, Any]]:
        """Recent orders (EXECUTIONS) from Alpaca — the source of truth for what actually traded,
        including weekly rebalance resizings that never create a DB Trade row. Newest-first.

        The DB `orders` table + the /trades (positions) view do NOT capture rebalance buys/sells;
        this reads the broker directly so the dashboard can show an execution-level blotter."""
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            _status = {"all": QueryOrderStatus.ALL, "open": QueryOrderStatus.OPEN,
                       "closed": QueryOrderStatus.CLOSED}.get(str(status).lower(), QueryOrderStatus.ALL)
            req = GetOrdersRequest(status=_status, limit=min(max(int(limit), 1), 500), direction="desc")
            orders = self.trading_client.get_orders(filter=req) or []
            out: List[Dict[str, Any]] = []
            for o in orders:
                out.append({
                    "order_id": str(o.id),
                    "symbol": o.symbol,
                    "side": str(o.side).rsplit(".", 1)[-1].lower(),          # OrderSide.BUY -> buy
                    "qty": float(o.qty) if o.qty else 0,                     # keep fractional (blotter = truth)
                    "filled_qty": float(o.filled_qty) if o.filled_qty else 0,
                    "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                    "status": str(o.status).rsplit(".", 1)[-1].lower(),      # OrderStatus.FILLED -> filled
                    "order_type": str(getattr(o, "order_type", "") or "").rsplit(".", 1)[-1].lower() or None,
                    "submitted_at": o.submitted_at.isoformat() if getattr(o, "submitted_at", None) else None,
                    "filled_at": o.filled_at.isoformat() if getattr(o, "filled_at", None) else None,
                })
            return out
        except Exception as e:  # noqa: BLE001 — dashboard read must degrade gracefully
            logger.error(f"Error fetching orders: {e}")
            return []

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "5Min",
        limit: int = 100,
        start=None,
        end=None,
    ) -> pd.DataFrame:
        """
        Get historical bars (OHLCV data)

        Args:
            symbol: Stock symbol
            timeframe: '1Min', '5Min', '15Min', '1H', '1D', '1Day'
            limit: Number of bars to fetch (ignored for daily if start/end provided)
            start: Optional start datetime/date
            end: Optional end datetime/date
        """
        try:
            timeframe_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "1H": TimeFrame.Hour,
                "1D": TimeFrame.Day,
                "1Day": TimeFrame.Day,
            }

            tf = timeframe_map.get(timeframe, TimeFrame(5, TimeFrameUnit.Minute))

            is_daily = timeframe in ("1D", "1Day")
            # IEX feed works on free tier for both intraday and daily with date ranges.
            # Only omit feed for the limit-based (recent) daily path where default works.
            if start is not None or end is not None:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start,
                    end=end,
                    feed="iex",
                )
            elif is_daily:
                _start = datetime.utcnow() - timedelta(days=int(limit * 1.5))
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=_start,
                )
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    limit=limit,
                    feed="iex",
                )

            _rate_limiter.acquire()
            bars = _retry_call(self.data_client.get_stock_bars, request)

            try:
                symbol_bars = bars[symbol]
                if symbol_bars:
                    records = [
                        {
                            "open": b.open, "high": b.high, "low": b.low,
                            "close": b.close, "volume": b.volume,
                            "vwap": getattr(b, "vwap", None),
                            "trade_count": getattr(b, "trade_count", None),
                        }
                        for b in symbol_bars
                    ]
                    timestamps = [b.timestamp for b in symbol_bars]
                    df = pd.DataFrame(records, index=pd.DatetimeIndex(timestamps, name="timestamp"))
                    # Keep only the last `limit` rows
                    return df.tail(limit)
            except (KeyError, TypeError):
                pass
            logger.debug(f"No data found for {symbol}")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            raise

    def get_bars_batch(
        self,
        symbols: List[str],
        timeframe: str = "5Min",
        limit: int = 100,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch bars for many symbols in a single API request.

        Returns dict {symbol: DataFrame} — symbols with no data are omitted.
        Uses one rate-limiter token for the entire batch (vs. one per symbol in get_bars).
        Splits into chunks of 200 to stay within Alpaca URL length limits.
        """
        timeframe_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1H": TimeFrame.Hour,
            "1D": TimeFrame.Day,
            "1Day": TimeFrame.Day,
        }
        tf = timeframe_map.get(timeframe, TimeFrame(5, TimeFrameUnit.Minute))
        is_daily = timeframe in ("1D", "1Day")
        result: Dict[str, pd.DataFrame] = {}
        # 5Min bars: 50 symbols per chunk keeps response under Alpaca's page limit
        # (~50 × 78 bars = 3,900 rows — safely under the ~10k row page cap).
        # Daily bars: 200 per chunk is fine (25 rows per symbol = 5,000 rows).
        chunk_size = 50 if not is_daily else 200
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i: i + chunk_size]
            try:
                if is_daily:
                    _start = datetime.utcnow() - timedelta(days=int(limit * 1.5))
                    request = StockBarsRequest(
                        symbol_or_symbols=chunk,
                        timeframe=tf,
                        start=_start,
                    )
                else:
                    # Do not specify feed — lets Alpaca use SIP on paper accounts
                    # (IEX covers only a subset of symbols in multi-symbol batch mode)
                    request = StockBarsRequest(
                        symbol_or_symbols=chunk,
                        timeframe=tf,
                        limit=limit,
                    )
                _rate_limiter.acquire()
                bars_resp = _retry_call(self.data_client.get_stock_bars, request)
                for sym in chunk:
                    try:
                        sym_bars = bars_resp[sym]
                        if not sym_bars:
                            continue
                        records = [
                            {
                                "open": b.open, "high": b.high, "low": b.low,
                                "close": b.close, "volume": b.volume,
                                "vwap": getattr(b, "vwap", None),
                                "trade_count": getattr(b, "trade_count", None),
                            }
                            for b in sym_bars
                        ]
                        timestamps = [b.timestamp for b in sym_bars]
                        df = pd.DataFrame(records, index=pd.DatetimeIndex(timestamps, name="timestamp"))
                        result[sym] = df.tail(limit)
                    except (KeyError, TypeError):
                        pass
            except Exception as exc:
                logger.warning("get_bars_batch chunk %d failed: %s", i // chunk_size, exc)
        return result

    def get_quote(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Return latest NBBO quote for symbol.

        Returns dict with keys: bid, ask, mid, spread_pct
        or None if the quote cannot be fetched.
        """
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            _rate_limiter.acquire()
            quotes = _retry_call(self.data_client.get_stock_latest_quote, request)
            quote = quotes.get(symbol)
            if quote is None:
                return None
            bid = float(quote.bid_price)
            ask = float(quote.ask_price)
            if bid <= 0 or ask <= 0:
                return None
            mid = (bid + ask) / 2
            spread_pct = (ask - bid) / mid if mid > 0 else 0.0
            return {"bid": bid, "ask": ask, "mid": mid, "spread_pct": spread_pct}
        except Exception as e:
            logger.debug("get_quote(%s) failed: %s", symbol, e)
            return None

    def get_bid(self, symbol: str) -> Optional[float]:
        """Return ONLY the NBBO bid (the executable sale price for a long), independent of ask
        validity. `get_quote` returns None if EITHER side is non-positive, so a one-sided book
        (missing/zero ask during a halt or violent gap-down) would null it — exactly when a long most
        needs its real bid to confirm a crash exit. Returns None if there is no positive bid."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            _rate_limiter.acquire()
            quotes = _retry_call(self.data_client.get_stock_latest_quote, request)
            quote = quotes.get(symbol)
            if quote is None:
                return None
            bid = float(quote.bid_price)
            return bid if bid > 0 else None
        except Exception as e:
            logger.debug("get_bid(%s) failed: %s", symbol, e)
            return None

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol"""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                limit=1,
            )
            _rate_limiter.acquire()
            bars = _retry_call(self.data_client.get_stock_bars, request)

            try:
                symbol_bars = bars[symbol]
                if symbol_bars:
                    return float(symbol_bars[-1].close)
            except (KeyError, TypeError, IndexError):
                pass
            return None
        except Exception as e:
            logger.error(f"Error fetching latest price for {symbol}: {e}")
            return None

    def get_clock(self) -> Optional[Dict[str, Any]]:
        """Return the market clock: {is_open, next_open, next_close, timestamp}.

        Used by the trend sleeve's weekly rebalance to fail-closed on market
        holidays (a weekday cron still fires on holiday Mondays). Returns None on
        error so callers can treat "unknown" as closed (do-not-trade).
        """
        try:
            _rate_limiter.acquire()
            clock = _retry_call(self.trading_client.get_clock)
            return {
                "is_open": bool(clock.is_open),
                "next_open": clock.next_open.isoformat() if clock.next_open else None,
                "next_close": clock.next_close.isoformat() if clock.next_close else None,
                "timestamp": clock.timestamp.isoformat() if clock.timestamp else None,
            }
        except Exception as e:
            logger.warning(f"get_clock failed: {e}")
            return None

    def health_check(self) -> bool:
        """Check if Alpaca API is accessible"""
        try:
            self.get_account()
            return True
        except Exception as e:
            logger.error(f"Alpaca API health check failed: {e}")
            return False


# Global instance
alpaca_client = None
_alpaca_client_lock = threading.Lock()


def get_alpaca_client() -> AlpacaClient:
    """Get or create Alpaca client instance (thread-safe singleton)."""
    global alpaca_client
    if alpaca_client is None:
        with _alpaca_client_lock:
            if alpaca_client is None:
                alpaca_client = AlpacaClient()
    return alpaca_client
