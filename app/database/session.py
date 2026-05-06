from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from app.config import settings
from app.database.models import Base
import logging

logger = logging.getLogger(__name__)

_is_postgres = settings.database_url.startswith("postgresql") or settings.database_url.startswith("postgres")
engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600,
    echo=settings.debug,
    connect_args={"connect_timeout": 5} if _is_postgres else {},
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """Get database session for dependency injection"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database - create all tables and run column migrations."""
    logger.info("Initializing database...")
    Base.metadata.create_all(bind=engine)
    _migrate_columns()
    logger.info("Database initialization complete")


def _migrate_columns() -> None:
    """Add new columns to existing tables (idempotent — safe to run on every startup)."""
    migrations = [
        # (table, column, definition)
        ("trades", "alpaca_order_id", "VARCHAR(50)"),
        ("trades", "proposal_id",    "VARCHAR(36)"),
        # Phase R — regime context columns
        ("proposal_log", "regime_score_at_scan",   "REAL"),
        ("proposal_log", "regime_label_at_scan",   "VARCHAR(15)"),
        ("proposal_log", "regime_trigger_at_scan", "VARCHAR(30)"),
        ("decision_audit", "regime_score_at_decision", "REAL"),
        ("daily_state",    "regime_score_premarket",   "REAL"),
        ("daily_state",    "regime_label_premarket",   "VARCHAR(15)"),
        ("daily_state",    "regime_last_updated_at",   "TIMESTAMP"),
    ]
    with engine.connect() as conn:
        for table, col, col_def in migrations:
            try:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}"))
                conn.commit()
                logger.info("Migration: added %s.%s", table, col)
            except Exception:
                # Column already exists — expected on every subsequent startup
                conn.rollback()


def get_session() -> Session:
    """Get a new database session"""
    return SessionLocal()


# Health check
def check_db_connection():
    """Check if database connection is working"""
    try:
        db = get_session()
        db.execute(text("SELECT 1"))
        db.close()
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
