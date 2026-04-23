"""
Feature store: persists computed (symbol, date) → feature dict to SQLite.

During training, the rolling-window loop recomputes all features for every
symbol×date combination on every run.  With 82 symbols × 150+ windows each
that is ~12,000 engineer_features() calls per retrain — each touching yfinance
fundamentals, FMP, and Polygon news.

The feature store caches the result keyed by (symbol, as_of_date).  A cache
hit avoids all API calls and computation; only new dates require fresh work.

The store is append-only from the training pipeline perspective.  Live
inference never reads from it — it always uses fresh data.

Versioning: SCHEMA_VERSION is stored in a metadata table.  When the version
in the DB doesn't match the current code version, the cache is automatically
cleared so stale feature vectors (missing new columns) are never served.
Bump SCHEMA_VERSION whenever engineer_features() adds or removes features.

Schema:
  features(symbol TEXT, as_of_date TEXT, features_json TEXT, created_at TEXT)
  meta(key TEXT PRIMARY KEY, value TEXT)
"""

import json
import logging
import sqlite3
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_DB = "app/ml/models/feature_store.db"

# Bump this whenever engineer_features() gains or loses columns.
# Mismatch → cache auto-cleared on startup.
SCHEMA_VERSION = "v3"  # v1=66 features, v2=74 features, v3=140 features (Phase 24b: regime interactions)


class FeatureStore:
    """
    SQLite-backed cache for engineered features.

    Usage:
        store = FeatureStore()
        cached = store.get("AAPL", date(2024, 3, 15))
        if cached is None:
            feats = fe.engineer_features(...)
            store.put("AAPL", date(2024, 3, 15), feats)
    """

    def __init__(self, db_path: str = _DEFAULT_DB):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    symbol      TEXT NOT NULL,
                    as_of_date  TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (symbol, as_of_date)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbol_date ON features(symbol, as_of_date)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
        self._check_version()

    def _check_version(self) -> None:
        """Clear cache if stored schema version doesn't match current code version."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            stored = row["value"] if row else None
        if stored != SCHEMA_VERSION:
            count = self.count()
            if count > 0:
                logger.warning(
                    "Feature store schema version mismatch (stored=%s current=%s) — "
                    "clearing %d stale entries",
                    stored, SCHEMA_VERSION, count,
                )
                self.clear()
            with self._conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                    (SCHEMA_VERSION,),
                )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, symbol: str, as_of: date) -> Optional[Dict[str, float]]:
        """Return cached features or None if not found."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT features_json FROM features WHERE symbol=? AND as_of_date=?",
                (symbol, str(as_of)),
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["features_json"])
        except Exception as exc:
            logger.warning("Feature store corrupt entry %s/%s: %s", symbol, as_of, exc)
            return None

    def get_batch(
        self, symbol: str, dates: List[date]
    ) -> Dict[date, Dict[str, float]]:
        """Return all cached feature dicts for a symbol across multiple dates."""
        if not dates:
            return {}
        date_strs = [str(d) for d in dates]
        placeholders = ",".join("?" * len(date_strs))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT as_of_date, features_json FROM features "
                f"WHERE symbol=? AND as_of_date IN ({placeholders})",
                [symbol] + date_strs,
            ).fetchall()
        result: Dict[date, Dict[str, float]] = {}
        for row in rows:
            try:
                d = date.fromisoformat(row["as_of_date"])
                result[d] = json.loads(row["features_json"])
            except Exception:
                pass
        return result

    # ── Write ─────────────────────────────────────────────────────────────────

    def put(self, symbol: str, as_of: date, features: Dict[str, float]) -> None:
        """Store features for (symbol, as_of_date).  Overwrites on conflict."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO features (symbol, as_of_date, features_json)
                   VALUES (?, ?, ?)""",
                (symbol, str(as_of), json.dumps(features)),
            )

    def put_batch(self, rows: List[Tuple[str, date, Dict[str, float]]]) -> None:
        """Bulk insert (symbol, as_of_date, features) tuples.  Faster than put() in a loop."""
        if not rows:
            return
        data = [(sym, str(d), json.dumps(feats)) for sym, d, feats in rows]
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO features (symbol, as_of_date, features_json) VALUES (?,?,?)",
                data,
            )
        logger.debug("FeatureStore: inserted %d rows", len(data))

    # ── Maintenance ───────────────────────────────────────────────────────────

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]

    def evict_before(self, cutoff: date) -> int:
        """Delete entries older than cutoff. Returns number of rows deleted."""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM features WHERE as_of_date < ?", (str(cutoff),)
            )
            return cursor.rowcount

    def clear(self) -> None:
        """Delete all cached feature rows (preserves schema version record)."""
        with self._conn() as conn:
            conn.execute("DELETE FROM features")
