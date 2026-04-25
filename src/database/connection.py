"""Database connection management for TrialIntel."""

import os
import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""

    _instance: Optional["DatabaseManager"] = None

    def __init__(
        self,
        database_url: Optional[str] = None,
        echo: bool = False,
    ):
        """
        Initialize database manager.

        Args:
            database_url: SQLAlchemy database URL. Defaults to DATABASE_URL env var or SQLite
            echo: Whether to echo SQL statements (for debugging)
        """
        if database_url is None:
            # Check for DATABASE_URL environment variable first (Railway, Heroku, etc.)
            database_url = os.environ.get("DATABASE_URL")

            if database_url:
                # Railway/Heroku sometimes use postgres:// but SQLAlchemy needs postgresql://
                if database_url.startswith("postgres://"):
                    database_url = database_url.replace("postgres://", "postgresql://", 1)
                logger.info(f"Using DATABASE_URL from environment")
            else:
                # Fall back to SQLite in data directory for local development
                data_dir = os.environ.get("TRIALINTEL_DATA_DIR", "./data")
                os.makedirs(data_dir, exist_ok=True)
                database_url = f"sqlite:///{data_dir}/trials.db"
                logger.info(f"Using SQLite database at {database_url}")

        self.database_url = database_url
        self.echo = echo
        self._engine = None
        self._session_factory = None

    @property
    def engine(self):
        """Get or create the database engine."""
        if self._engine is None:
            # SQLite-specific configuration
            if self.database_url.startswith("sqlite"):
                self._engine = create_engine(
                    self.database_url,
                    echo=self.echo,
                    connect_args={"check_same_thread": False},
                    poolclass=StaticPool,
                )
                # Enable foreign keys for SQLite
                @event.listens_for(self._engine, "connect")
                def set_sqlite_pragma(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA foreign_keys=ON")
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.close()
            else:
                # PostgreSQL or other databases
                self._engine = create_engine(
                    self.database_url,
                    echo=self.echo,
                    pool_size=5,
                    max_overflow=10,
                    pool_pre_ping=True,
                )

            logger.info(f"Database engine created: {self.database_url}")

        return self._engine

    @property
    def session_factory(self):
        """Get or create the session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
            )
        return self._session_factory

    def create_tables(self):
        """Create all tables if they don't exist."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")

    def drop_tables(self):
        """Drop all tables (use with caution!)."""
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Database tables dropped")

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope around a series of operations.

        Usage:
            with db.session() as session:
                session.add(trial)
                session.commit()
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_session(self) -> Session:
        """Get a new session (caller must manage lifecycle)."""
        return self.session_factory()

    def execute_raw(self, sql: str, params: dict = None) -> list:
        """Execute raw SQL and return results."""
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            return result.fetchall()

    def get_stats(self) -> dict:
        """Get database statistics."""
        stats = {}

        with self.session() as session:
            # Total trials
            result = session.execute(text("SELECT COUNT(*) FROM trials"))
            stats["total_trials"] = result.scalar() or 0

            # By status
            result = session.execute(text("""
                SELECT status, COUNT(*) as count
                FROM trials
                GROUP BY status
                ORDER BY count DESC
            """))
            stats["by_status"] = {row[0]: row[1] for row in result}

            # By phase
            result = session.execute(text("""
                SELECT phase, COUNT(*) as count
                FROM trials
                GROUP BY phase
                ORDER BY count DESC
            """))
            stats["by_phase"] = {row[0]: row[1] for row in result}

            # By therapeutic area
            result = session.execute(text("""
                SELECT therapeutic_area, COUNT(*) as count
                FROM trials
                GROUP BY therapeutic_area
                ORDER BY count DESC
                LIMIT 20
            """))
            stats["by_therapeutic_area"] = {row[0]: row[1] for row in result}

            # Sites count
            try:
                result = session.execute(text("SELECT COUNT(*) FROM sites"))
                stats["total_sites"] = result.scalar() or 0
            except Exception:
                stats["total_sites"] = 0

            # Endpoints count
            try:
                result = session.execute(text("SELECT COUNT(*) FROM endpoints"))
                stats["total_endpoints"] = result.scalar() or 0
            except Exception:
                stats["total_endpoints"] = 0

        return stats

    @classmethod
    def get_instance(cls, **kwargs) -> "DatabaseManager":
        """Get singleton instance of DatabaseManager."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (for testing)."""
        if cls._instance is not None:
            if cls._instance._engine is not None:
                cls._instance._engine.dispose()
            cls._instance = None


# Dependency injection helper for FastAPI
def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.

    Usage:
        @app.get("/trials")
        def get_trials(db: Session = Depends(get_db)):
            return db.query(Trial).all()
    """
    db_manager = DatabaseManager.get_instance()
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()


# Convenience function for scripts
def init_database(database_url: Optional[str] = None, create_tables: bool = True) -> DatabaseManager:
    """
    Initialize the database.

    Args:
        database_url: Optional database URL (defaults to SQLite)
        create_tables: Whether to create tables if they don't exist

    Returns:
        DatabaseManager instance
    """
    db = DatabaseManager.get_instance(database_url=database_url)
    if create_tables:
        db.create_tables()
    return db
