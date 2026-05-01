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
    """Initialize database - create all tables"""
    logger.info("Initializing database...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialization complete")


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
