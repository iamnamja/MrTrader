from pathlib import Path
from typing import Optional

from pydantic import ConfigDict
from pydantic_settings import BaseSettings

# Resolve .env from the repo root (parent of this file's directory) so the
# correct file is always found regardless of CWD or how uvicorn is launched.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Application settings.

    Precedence (highest to lowest):
      1. Explicit constructor kwargs
      2. OS environment variables
      3. .env file at repo root (resolved via __file__, independent of CWD)
      4. Field defaults
    """

    model_config = ConfigDict(
        protected_namespaces=("settings_",),
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

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

    # Regime-aware position sizing multipliers (Phase regime-sizing)
    # Override via .env: REGIME_SIZING_RISK_ON=1.0, etc.
    regime_sizing_risk_on: float = 1.0       # score >= regime_risk_on_threshold
    regime_sizing_risk_caution: float = 0.6  # regime_risk_off_threshold <= score < regime_risk_on_threshold
    regime_sizing_risk_off: float = 0.3      # score < regime_risk_off_threshold
    regime_sizing_unknown: float = 1.0       # UNKNOWN label or no model (conservative: full size until model is stable)
    regime_risk_on_threshold: float = 0.60   # V2: was 0.65
    regime_risk_off_threshold: float = 0.30  # V2: was 0.35

    # Phase 3d: Volatility-targeting position sizing
    # When enabled, sizes each position to contribute a fixed % of account equity in daily vol.
    # quantity = floor(account * vol_target_pct / (atr_norm * price))
    # Bounded above by max_position_size_pct, below by vol_targeting_min_notional.
    vol_targeting_enabled: bool = False
    vol_target_pct: float = 0.005          # 0.5% of account equity in vol per position
    vol_targeting_min_notional: float = 500.0  # never size below $500

    # Phase 5b: Opportunity score weights (must sum to 1.0)
    # New weights include breadth and dispersion inputs.
    opp_score_vix_weight: float = 0.25
    opp_score_vix_trend_weight: float = 0.15
    opp_score_ma_weight: float = 0.25
    opp_score_mom_weight: float = 0.10
    opp_score_breadth_weight: float = 0.15
    opp_score_dispersion_weight: float = 0.10

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
