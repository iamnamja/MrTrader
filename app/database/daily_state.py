"""
Helpers for reading and writing DailyState — one row per trading day.
Used by PM and Trader to persist workflow flags across restarts.
"""
from __future__ import annotations

from datetime import date as _date
from typing import Optional

from app.database.models import DailyState
from app.database.session import get_session


def _today() -> str:
    return _date.today().isoformat()


def get_state(date_str: Optional[str] = None) -> DailyState:
    """Return (or create) the DailyState row for the given date (default today)."""
    d = date_str or _today()
    db = get_session()
    try:
        row = db.query(DailyState).filter_by(date=d).first()
        if row is None:
            row = DailyState(date=d)
            db.add(row)
            db.commit()
            db.refresh(row)
        return row
    finally:
        db.close()


def set_flag(flag: str, value: bool = True, date_str: Optional[str] = None) -> None:
    """Set a boolean flag on today's DailyState row."""
    d = date_str or _today()
    db = get_session()
    try:
        row = db.query(DailyState).filter_by(date=d).first()
        if row is None:
            row = DailyState(date=d)
            db.add(row)
        setattr(row, flag, value)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_flag(flag: str, date_str: Optional[str] = None) -> bool:
    """Read a boolean flag from today's DailyState row. Returns False if not found."""
    d = date_str or _today()
    db = get_session()
    try:
        row = db.query(DailyState).filter_by(date=d).first()
        if row is None:
            return False
        return bool(getattr(row, flag, False))
    finally:
        db.close()
