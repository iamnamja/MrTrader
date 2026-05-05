import logging
import threading
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
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


def _notify_circuit_breaker():
    """Inform the circuit breaker of a network error (best-effort, no circular import)."""
    try:
        from app.agents.circuit_breaker import circuit_breaker
        circuit_breaker.record_network_error()
    except Exception:
        pass


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
            account = self.trading_client.get_account()
            return {
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
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
            positions = self.trading_client.get_all_positions()
            result = []
            for pos in positions:
                result.append({
                    "symbol": pos.symbol,
                    "qty": int(pos.qty),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                    "current_price": float(pos.current_price),
                })
            return result
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            raise

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for a specific symbol"""
        try:
            _rate_limiter.acquire()
            position = self.trading_client.get_open_position(symbol)
            return {
                "symbol": position.symbol,
                "qty": int(position.qty),
                "avg_entry_price": float(position.avg_entry_price),
                "market_value": float(position.market_value),
                "unrealized_pl": float(position.unrealized_pl),
                "unrealized_plpc": float(position.unrealized_plpc),
                "current_price": float(position.current_price),
            }
        except Exception as e:
            logger.debug(f"Position not found for {symbol}: {e}")
            return None

    def place_market_order(
        self, symbol: str, quantity: int, side: str, client_order_id: str | None = None
    ) -> Dict[str, Any]:
        """
        Place a market order

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: 'buy' or 'sell'
            client_order_id: Optional identifier we control (proposal UUID); queryable via Alpaca
        """
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
                "qty": int(order.qty),
                "side": str(order.side),
                "status": order.status,
                "created_at": order.created_at.isoformat() if order.created_at else None,
            }
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
                "qty": int(order.qty),
                "side": str(order.side),
                "limit_price": float(order.limit_price),
                "status": order.status,
                "created_at": order.created_at.isoformat() if order.created_at else None,
            }
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
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
                "qty": int(order.qty),
                "filled_qty": int(order.filled_qty) if order.filled_qty else 0,
                "side": str(order.side),
                "status": order.status,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            }
        except Exception as e:
            logger.error(f"Error fetching order status: {e}")
            return None

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
            bars = self.data_client.get_stock_bars(request)

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
                bars_resp = self.data_client.get_stock_bars(request)
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
            quotes = self.data_client.get_stock_latest_quote(request)
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

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol"""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                limit=1,
            )
            _rate_limiter.acquire()
            bars = self.data_client.get_stock_bars(request)

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
