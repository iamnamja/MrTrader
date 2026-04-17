"""
Watchlist management API — dynamic ticker universe for the portfolio manager.

GET  /api/watchlist          — list all tickers
POST /api/watchlist          — add a ticker
DELETE /api/watchlist/{sym}  — remove a ticker
PATCH /api/watchlist/{sym}   — update active flag / notes
POST /api/watchlist/bulk     — bulk-add from SP_100 defaults
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database.models import WatchlistTicker
from app.database.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/watchlist", tags=["watchlist"])


class TickerIn(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10)
    sector: Optional[str] = None
    notes: Optional[str] = None
    active: bool = True


class TickerPatch(BaseModel):
    active: Optional[bool] = None
    sector: Optional[str] = None
    notes: Optional[str] = None


def _row_to_dict(t: WatchlistTicker) -> dict:
    return {
        "id": t.id,
        "symbol": t.symbol,
        "sector": t.sector,
        "notes": t.notes,
        "active": bool(t.active),
        "added_at": t.added_at.isoformat() if t.added_at else None,
    }


@router.get("")
def list_tickers(active_only: bool = False, db: Session = Depends(get_db)):
    q = db.query(WatchlistTicker)
    if active_only:
        q = q.filter(WatchlistTicker.active == 1)
    return {"tickers": [_row_to_dict(t) for t in q.order_by(WatchlistTicker.symbol).all()]}


@router.post("", status_code=201)
def add_ticker(body: TickerIn, db: Session = Depends(get_db)):
    sym = body.symbol.upper().strip()
    existing = db.query(WatchlistTicker).filter(WatchlistTicker.symbol == sym).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"{sym} already in watchlist")
    t = WatchlistTicker(symbol=sym, sector=body.sector, notes=body.notes,
                        active=1 if body.active else 0)
    db.add(t)
    db.commit()
    db.refresh(t)
    return _row_to_dict(t)


@router.delete("/{symbol}", status_code=200)
def remove_ticker(symbol: str, db: Session = Depends(get_db)):
    sym = symbol.upper()
    t = db.query(WatchlistTicker).filter(WatchlistTicker.symbol == sym).first()
    if not t:
        raise HTTPException(status_code=404, detail=f"{sym} not found")
    db.delete(t)
    db.commit()
    return {"removed": sym}


@router.patch("/{symbol}", status_code=200)
def update_ticker(symbol: str, body: TickerPatch, db: Session = Depends(get_db)):
    sym = symbol.upper()
    t = db.query(WatchlistTicker).filter(WatchlistTicker.symbol == sym).first()
    if not t:
        raise HTTPException(status_code=404, detail=f"{sym} not found")
    if body.active is not None:
        t.active = 1 if body.active else 0
    if body.sector is not None:
        t.sector = body.sector
    if body.notes is not None:
        t.notes = body.notes
    db.commit()
    db.refresh(t)
    return _row_to_dict(t)


@router.post("/bulk", status_code=200)
def bulk_load_sp100(db: Session = Depends(get_db)):
    """Seed watchlist from the hardcoded SP_100_TICKERS list."""
    from app.utils.constants import SP_100_TICKERS, SECTOR_MAP
    added, skipped = [], []
    for sym in SP_100_TICKERS:
        existing = db.query(WatchlistTicker).filter(WatchlistTicker.symbol == sym).first()
        if existing:
            skipped.append(sym)
            continue
        sector = SECTOR_MAP.get(sym)
        db.add(WatchlistTicker(symbol=sym, sector=sector, active=1))
        added.append(sym)
    db.commit()
    return {"added": added, "skipped": skipped, "total_added": len(added)}
