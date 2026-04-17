"""
Simple in-process TTL cache for API endpoints that call external services.

Usage:
    from app.api.cache import ttl_cache

    @ttl_cache(seconds=15)
    async def my_endpoint():
        ...

Thread-safe for async handlers. Cache is per-process (not shared across
workers), which is fine for a single-worker dev/paper-trading setup.
"""

import functools
import time
from typing import Any, Dict, Tuple


_store: Dict[str, Tuple[float, Any]] = {}


def ttl_cache(seconds: int = 10):
    """
    Decorator that caches the return value of an async function for `seconds`.
    Cache key is the function's qualified name + stringified args/kwargs.
    """
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            key = f"{fn.__qualname__}:{args}:{sorted(kwargs.items())}"
            now = time.monotonic()
            if key in _store:
                ts, val = _store[key]
                if now - ts < seconds:
                    return val
            result = await fn(*args, **kwargs)
            _store[key] = (now, result)
            return result
        return wrapper
    return decorator


def invalidate(fn_qualname: str) -> None:
    """Remove all cache entries for a given function qualname."""
    keys = [k for k in _store if k.startswith(fn_qualname + ":")]
    for k in keys:
        del _store[k]
