# Backward-compatibility shim.
# The canonical session factory lives in app.database.session.
# This module exists solely so that any remaining stale imports
# ("from app.database.db import SessionLocal") don't crash at runtime.
# Do NOT add new imports here — use app.database.session directly.
from app.database.session import SessionLocal, engine, get_session  # noqa: F401
