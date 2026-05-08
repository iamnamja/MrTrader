"""
_model_loader.py — Shared model loading logic for the walkforward package.
"""
from __future__ import annotations

import pickle
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def load_model(model_name: str, version: Optional[int] = None) -> Tuple[object, int]:
    """Load a trained model by name. Returns (model, version) or (None, 0)."""
    try:
        from app.database.models import ModelVersion as MV
        from app.database.session import get_session
        db = get_session()
        try:
            if version is not None:
                mv = db.query(MV).filter_by(model_name=model_name, version=version).first()
            else:
                mv = (
                    db.query(MV)
                    .filter_by(model_name=model_name, status="ACTIVE")
                    .order_by(MV.version.desc())
                    .first()
                )
            if mv and mv.model_path:
                path = Path(mv.model_path)
                if path.exists():
                    with open(path, "rb") as f:
                        obj = pickle.load(f)
                    if hasattr(obj, "is_trained") and obj.is_trained:
                        logger.info("Loaded %s model v%d from %s", model_name, mv.version, path)
                        return obj, mv.version
                    elif not hasattr(obj, "is_trained"):
                        return obj, mv.version
        finally:
            db.close()
    except Exception as exc:
        logger.warning("Model load failed for %s v%s: %s", model_name, version, exc)
    return None, 0
