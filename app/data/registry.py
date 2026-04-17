"""
Provider registry — central lookup for all data providers.

Usage:
    from app.data import get_provider

    provider = get_provider("yfinance")   # for training
    provider = get_provider("alpaca")     # for live signal generation
    provider = get_provider()             # default (yfinance)

Adding a new provider:
    from app.data.registry import register_provider
    register_provider("polygon", PolygonProvider())
"""

import logging
from typing import Dict, Optional

from app.data.base import DataProvider

logger = logging.getLogger(__name__)

_registry: Dict[str, DataProvider] = {}
_default_provider = "yfinance"


def register_provider(name: str, provider: DataProvider) -> None:
    """Register a provider instance under *name*."""
    _registry[name] = provider
    logger.debug("Registered data provider: %s", name)


def get_provider(name: Optional[str] = None) -> DataProvider:
    """
    Return the named provider, auto-registering built-ins on first call.

    Args:
        name: 'yfinance', 'alpaca', or any registered name.
              Defaults to 'yfinance'.

    Raises:
        ValueError if the name is unknown.
    """
    _ensure_defaults_registered()
    key = name or _default_provider
    if key not in _registry:
        raise ValueError(
            f"Unknown data provider '{key}'. "
            f"Available: {list(_registry.keys())}"
        )
    return _registry[key]


def list_providers() -> list:
    """Return names of all registered providers."""
    _ensure_defaults_registered()
    return list(_registry.keys())


def _ensure_defaults_registered() -> None:
    """Lazily register built-in providers on first access."""
    if "yfinance" not in _registry:
        from app.data.yfinance_provider import YFinanceProvider
        _registry["yfinance"] = YFinanceProvider()

    if "alpaca" not in _registry:
        try:
            from app.data.alpaca_provider import AlpacaProvider
            _registry["alpaca"] = AlpacaProvider()
        except Exception as exc:
            logger.debug("Alpaca provider not registered (keys missing?): %s", exc)

    if "polygon" not in _registry:
        try:
            from app.data.polygon_provider import PolygonProvider
            _registry["polygon"] = PolygonProvider()
        except Exception as exc:
            logger.debug("Polygon provider not registered: %s", exc)
