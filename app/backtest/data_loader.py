import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load historical data for backtesting"""

    @staticmethod
    def download_ohlcv(
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Download OHLCV data

        Args:
            symbol: Stock ticker
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            interval: 1m, 5m, 1h, 1d
        """
        logger.info(f"Downloading {symbol} from {start_date} to {end_date}...")

        try:
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return df

            # Flatten MultiIndex columns if present (yfinance >= 0.2.x)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Normalize column names to lowercase
            df.columns = [c.lower() for c in df.columns]

            # Ensure index is named 'datetime'
            df.index.name = "datetime"

            logger.info(f"Downloaded {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
            raise

    @staticmethod
    def download_multiple(
        symbols: list,
        start_date: str,
        end_date: str,
    ) -> Dict[str, pd.DataFrame]:
        """Download data for multiple symbols"""
        data = {}

        for symbol in symbols:
            try:
                df = DataLoader.download_ohlcv(symbol, start_date, end_date)
                if not df.empty:
                    data[symbol] = df
            except Exception as e:
                logger.warning(f"Could not download {symbol}: {e}")

        return data

    @staticmethod
    def get_date_range(years: int) -> tuple:
        """Get start and end dates for backtesting"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)

        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
