from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = ConfigDict(protected_namespaces=('settings_',), env_file=".env", case_sensitive=False)

    # Database
    database_url: str = "postgresql://mrtrader:mrtrader_password@localhost:5432/mrtrader"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Alpaca API
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # NewsAPI
    news_api_key: Optional[str] = None

    # Trading Configuration
    trading_mode: str = "paper"  # 'paper' or 'live'
    initial_capital: float = 100000.0
    max_position_size_pct: float = 0.05  # 5%
    max_sector_concentration_pct: float = 0.20  # 20%
    max_daily_loss_pct: float = 0.02  # 2%
    max_account_drawdown_pct: float = 0.05  # 5%

    # External data
    fred_api_key: Optional[str] = None   # https://fred.stlouisfed.org/docs/api/api_key.html
    anthropic_api_key: Optional[str] = None  # https://console.anthropic.com/
    # Accepts ALPHA_VANTAGE_API_KEY or ALPHA_ADVANTAGE_API_KEY (common typo)
    alpha_vantage_api_key: Optional[str] = None
    alpha_advantage_api_key: Optional[str] = None  # alias for typo-tolerant .env
    # Finnhub — economic calendar, earnings calendar, company news
    finnhub_api_key: Optional[str] = None
    finhub_api_key: Optional[str] = None   # typo-tolerant alias (one 'n')
    # Financial Modeling Prep — fundamentals, earnings history, analyst ratings
    fmp_api_key: Optional[str] = None
    # Polygon.io — intraday bars, options flow, news sentiment
    polygon_api_key: Optional[str] = None
    polygon_s3_access_key: Optional[str] = None   # for bulk flat-file downloads
    polygon_s3_secret_key: Optional[str] = None
    polygon_s3_endpoint: Optional[str] = None
    polygon_s3_bucket: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: str = "MrTrader/1.0"

    # Alerts — Slack
    slack_webhook_url: Optional[str] = None
    # Alerts — Email (SMTP)
    alert_email: Optional[str] = None
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None

    # Application
    log_level: str = "INFO"
    debug: bool = False
    port: int = 8000

    # ML Model
    model_retraining_hour: int = 17  # 5 PM
    historical_data_years: int = 3

    # Market
    market_open_time: str = "09:30"
    market_close_time: str = "16:00"
    trader_check_interval: int = 300  # 5 minutes


settings = Settings()
