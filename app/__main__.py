"""
Launcher: python -m app

Scrubs environment variables that should come exclusively from .env before
any application modules are imported. This prevents stray shell-level env
vars (e.g. from a PS profile or a previous session) from silently overriding
.env values, since os.environ takes precedence over the .env file in
pydantic-settings.

Add any other vars to _SCRUB if they appear incorrectly in os.environ.
"""
import os

_SCRUB = [
    "INITIAL_CAPITAL",
]

for _k in _SCRUB:
    if _k in os.environ:
        print(f"[launcher] Removed stale env var {_k}={os.environ[_k]!r} — .env value will be used instead")
        del os.environ[_k]

import uvicorn  # noqa: E402  (import after env cleanup)
from app.config import settings  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
    )
