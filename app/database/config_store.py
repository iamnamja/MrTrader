"""
Configuration key/value store backed by the `configuration` DB table.

Used to persist runtime state that must survive process restarts:
  - capital ramp stage + start time
  - kill-switch active flag
  - trading mode

All values are stored as JSON strings.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_config(db, key: str) -> Optional[Any]:
    """
    Read a value from the configuration table.
    Returns the parsed JSON value, or None if the key doesn't exist.
    """
    from app.database.models import Configuration
    row = db.query(Configuration).filter_by(key=key).first()
    if row is None:
        return None
    try:
        return json.loads(row.value)
    except (json.JSONDecodeError, TypeError):
        return row.value


def set_config(db, key: str, value: Any, description: str = "") -> None:
    """
    Write (upsert) a value into the configuration table.
    value is serialised to JSON.
    """
    from app.database.models import Configuration
    from datetime import datetime

    serialised = json.dumps(value)
    row = db.query(Configuration).filter_by(key=key).first()
    if row is None:
        row = Configuration(key=key, value=serialised, description=description)
        db.add(row)
    else:
        row.value = serialised
        row.updated_at = datetime.utcnow()
    db.commit()
