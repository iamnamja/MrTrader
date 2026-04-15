import logging
from typing import List, Optional, Dict, Any
import pandas as pd
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from app.config import settings
import asyncio

logger = logging.getLogger(__name__)


class AlpacaClient:
    """Alpaca API client wrapper for trading and market data"""

    def __init__(self):
        self.trading_client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            base_url=settings.alpaca_base_url,
        )
        self.data_client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
        logger.info(f"Alpaca client initialized (Mode: {settings.trading_mode})")

    def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        try:
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
            positions = self.trading_client.get_all_positions()
            result = []
            for pos in positions:
                result.append({
                    "symbol": pos.symbol,
                    "qty": int(pos.qty),
                    "avg_fill_price": float(pos.avg_fill_price),
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
            position = self.trading_client.get_open_position(symbol)
            return {
                "symbol": position.symbol,
                "qty": int(position.qty),
                "avg_fill_price": float(position.avg_fill_price),
                "market_value": float(position.market_value),
                "unrealized_pl": float(position.unrealized_pl),
                "unrealized_plpc": float(position.unrealized_plpc),
                "current_price": float(position.current_price),
            }
        except Exception as e:
            logger.debug(f"Position not found for {symbol}: {e}")
            return None

    def place_market_order(self, symbol: str, quantity: int, side: str) -> Dict[str, Any]:
        """
        Place a market order

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: 'buy' or 'sell'
        """
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
            order = self.trading_client.submit_order(order_request)
            logger.info(f"Market order placed: {symbol} {side} {quantity} shares")
            return {
                "order_id": order.id,
                "symbol": order.symbol,
                "qty": int(order.qty),
                "side": str(order.side),
                "status": order.status,
                "created_at": order.created_at.isoformat() if order.created_at else None,
            }
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            raise

    def place_limit_order(
        self, symbol: str, quantity: int, side: str, limit_price: float
    ) -> Dict[str, Any]:
        """
        Place a limit order

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: 'buy' or 'sell'
            limit_price: Limit price
        """
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                limit_price=limit_price,
                time_in_force=TimeInForce.DAY,
            )
            order = self.trading_client.submit_order(order_request)
            logger.info(f"Limit order placed: {symbol} {side} {quantity} @ ${limit_price}")
            return {
                "order_id": order.id,
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
    ) -> pd.DataFrame:
        """
        Get historical bars (OHLCV data)

        Args:
            symbol: Stock symbol
            timeframe: '1Min', '5Min', '15Min', '1H', '1D'
            limit: Number of bars to fetch
        """
        try:
            # Map timeframe strings to TimeFrame enum
            timeframe_map = {
                "1Min": TimeFrame.MINUTE,
                "5Min": TimeFrame.FIVE_MIN,
                "15Min": TimeFrame.FIFTEEN_MIN,
                "1H": TimeFrame.HOUR,
                "1D": TimeFrame.DAY,
            }

            tf = timeframe_map.get(timeframe, TimeFrame.FIVE_MIN)

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                limit=limit,
            )

            bars = self.data_client.get_stock_bars(request)

            if symbol in bars:
                df = bars[symbol].df
                df.index.name = "timestamp"
                return df
            else:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            raise

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol"""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.MINUTE,
                limit=1,
            )
            bars = self.data_client.get_stock_bars(request)

            if symbol in bars:
                df = bars[symbol].df
                if not df.empty:
                    return float(df["close"].iloc[-1])
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


def get_alpaca_client() -> AlpacaClient:
    """Get or create Alpaca client instance"""
    global alpaca_client
    if alpaca_client is None:
        alpaca_client = AlpacaClient()
    return alpaca_client
