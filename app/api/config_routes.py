"""
Agent configuration API.

GET  /api/config/schema   — returns full schema (keys, types, bounds, descriptions)
GET  /api/config          — returns current values (DB overrides merged with defaults)
PUT  /api/config/{key}    — update one value
POST /api/config/reset    — reset all values to defaults (deletes DB overrides)
"""
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.database.agent_config import (
    CONFIG_SCHEMA, get_all_agent_config, get_agent_config, set_agent_config,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/config", tags=["config"])


class ConfigValue(BaseModel):
    value: Any


@router.get("/schema")
def get_schema():
    """Return full schema metadata for all tunable parameters."""
    return {"schema": CONFIG_SCHEMA}


@router.get("")
def get_config(db: Session = Depends(get_db)):
    """Return current effective values for all agent config keys."""
    return {"config": get_all_agent_config(db)}


@router.put("/{key:path}")
def update_config(key: str, body: ConfigValue, db: Session = Depends(get_db)):
    """Update a single config value. Validates type and range."""
    try:
        set_agent_config(db, key, body.value)
        new_val = get_agent_config(db, key)
        return {"key": key, "value": new_val}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/reset")
def reset_config(db: Session = Depends(get_db)):
    """Delete all DB overrides — agents revert to hardcoded defaults."""
    from app.database.models import Configuration
    deleted = (
        db.query(Configuration)
        .filter(Configuration.key.like("agent.%"))
        .delete(synchronize_session=False)
    )
    db.commit()
    return {"reset": True, "deleted_keys": deleted}
