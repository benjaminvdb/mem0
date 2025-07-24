import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

from pydantic import BaseModel, Field

# SQLAlchemy and database drivers
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    inspect,
    select,
    text,
    update,
)
from sqlalchemy.engine import Connection

# For URL parsing
from sqlalchemy.engine.url import make_url
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)


class HistoryDBConfig(BaseModel):
    """Configuration for the history database."""

    type: str = Field(
        "sqlite", description="Type of database ('sqlite', 'postgresql', 'mysql')"
    )
    url: str = Field(
        "sqlite:///:memory:", description="Complete database connection URL or path"
    )


class SQLDatabaseManager:
    """
    Database manager that supports SQLite, PostgreSQL, and MySQL databases
    with a consistent API across all database types.
    """

    def __init__(
        self,
        db_type: str = "sqlite",
        db_url: str = "sqlite:///:memory:",
        **kwargs,
    ):
        """
        Initialize database manager with the specified database type and connection URL.
        Args:
            db_type: Type of database (SQLite, PostgreSQL, MySQL)
            db_url: Complete database connection URL or path
            **kwargs: Additional connection parameters
        """
        self.db_type = db_type
        self._lock = threading.Lock()

        # Process URLs for SQLite that might be direct file paths
        if db_type == "sqlite":
            if not db_url.startswith("sqlite:///") and not db_url.startswith(
                "sqlite://"
            ):
                if db_url == ":memory:" or not db_url:
                    db_url = "sqlite:///:memory:"
                else:
                    db_url = f"sqlite:///{db_url}"
            kwargs.setdefault("connect_args", {"check_same_thread": False})

        # For PostgreSQL: create the database if it does not exist.
        if db_type == "postgresql":
            url_obj = make_url(db_url)
            db_name = url_obj.database
            if not db_name:
                raise ValueError("No database name specified in db_url for PostgreSQL")
            # Change database to "postgres" so we can check for and create the target database
            master_url = url_obj.set(database="postgres")
            kwargs.pop("connect_args", None)
            master_engine = create_engine(
                master_url.render_as_string(hide_password=False),
                isolation_level="AUTOCOMMIT",
                **kwargs,
            )
            with master_engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT 1 FROM pg_database WHERE datname = :db_name"
                    ).bindparam(db_name=db_name)
                )
                if not result.fetchone():
                    conn.execute(text(f'CREATE DATABASE "{db_name}"'))
            master_engine.dispose()

        # For MySQL: create the database if it does not exist.
        if db_type == "mysql":
            url_obj = make_url(db_url)
            db_name = url_obj.database
            if not db_name:
                raise ValueError("No database name specified in db_url for MySQL")
            # Change database to None to connect to the MySQL server without a specific database
            master_url = url_obj.set(database=None)
            kwargs.pop("connect_args", None)
            master_engine = create_engine(
                master_url.render_as_string(hide_password=False), **kwargs
            )
            with master_engine.connect() as conn:
                result = conn.execute(
                    text(f"SHOW DATABASES LIKE '{db_name}'")
                )
                if not result.fetchone():
                    conn.execute(text(f"CREATE DATABASE `{db_name}`"))
            master_engine.dispose()

        # Create engine and connect
        engine_kwargs = kwargs.copy()
        if db_type == "sqlite":
            engine_kwargs.setdefault("poolclass", QueuePool)
            engine_kwargs.setdefault("pool_size", 10)
            engine_kwargs.setdefault("max_overflow", 20)

        self._engine = create_engine(db_url, **engine_kwargs)

        # Define table schema with new group chat fields
        self.metadata = MetaData()
        self.history_table = Table(
            "history",
            self.metadata,
            Column("id", String(36), primary_key=True),
            Column("memory_id", String(255), nullable=False),
            Column("old_memory", String, nullable=True),
            Column("new_memory", String, nullable=True),
            Column("new_value", String, nullable=True),  # Keep for compatibility
            Column("event", String(50), nullable=False),
            Column("created_at", DateTime, nullable=True),
            Column("updated_at", DateTime, nullable=True),
            Column("is_deleted", Integer, default=0),
            # New group chat fields
            Column("actor_id", String(255), nullable=True),
            Column("role", String(100), nullable=True),
        )

        # Create tables
        self.metadata.create_all(self._engine)
        self._migrate_history_table()

    def _migrate_history_table(self) -> None:
        """
        Migrate existing history table to support new schema with group chat fields.
        This handles both old schemas and adds the new actor_id/role columns if needed.
        """
        with self._lock:
            try:
                inspector = inspect(self._engine)
                if not inspector.has_table("history"):
                    return  # No table to migrate

                # Get current columns
                current_columns = {col["name"] for col in inspector.get_columns("history")}
                expected_columns = {
                    "id", "memory_id", "old_memory", "new_memory", "event",
                    "created_at", "updated_at", "is_deleted", "actor_id", "role", "new_value"
                }

                missing_columns = expected_columns - current_columns
                if not missing_columns:
                    return  # Already up to date

                logger.info(f"Migrating history table, adding columns: {missing_columns}")

                # Add missing columns
                with self._engine.begin() as conn:
                    if "actor_id" in missing_columns:
                        if self.db_type == "sqlite":
                            conn.execute(text("ALTER TABLE history ADD COLUMN actor_id TEXT"))
                        elif self.db_type == "postgresql":
                            conn.execute(text("ALTER TABLE history ADD COLUMN actor_id VARCHAR(255)"))
                        elif self.db_type == "mysql":
                            conn.execute(text("ALTER TABLE history ADD COLUMN actor_id VARCHAR(255)"))

                    if "role" in missing_columns:
                        if self.db_type == "sqlite":
                            conn.execute(text("ALTER TABLE history ADD COLUMN role TEXT"))
                        elif self.db_type == "postgresql":
                            conn.execute(text("ALTER TABLE history ADD COLUMN role VARCHAR(100)"))
                        elif self.db_type == "mysql":
                            conn.execute(text("ALTER TABLE history ADD COLUMN role VARCHAR(100)"))

                    if "new_value" in missing_columns:
                        if self.db_type == "sqlite":
                            conn.execute(text("ALTER TABLE history ADD COLUMN new_value TEXT"))
                        elif self.db_type == "postgresql":
                            conn.execute(text("ALTER TABLE history ADD COLUMN new_value TEXT"))
                        elif self.db_type == "mysql":
                            conn.execute(text("ALTER TABLE history ADD COLUMN new_value TEXT"))

            except Exception as e:
                logger.error(f"Failed to migrate history table: {e}")
                raise

    @contextmanager
    def _get_connection(self) -> Generator[Connection, None, None]:
        """Context manager for database connections."""
        connection = self._engine.connect()
        try:
            yield connection
        finally:
            connection.close()

    def _ensure_datetime(self, value):
        """Ensure the value is a datetime object or convert from ISO string."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                # Try parsing ISO format
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
        raise ValueError(
            "Value must be a datetime object or an ISO format datetime string."
        )

    def add_history(
        self,
        memory_id: str,
        old_memory: Optional[str],
        new_memory: Optional[str],
        event: str,
        *,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        is_deleted: int = 0,
        actor_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> str:
        """
        Add a history record to the database.
        Returns:
            The ID of the newly created history record.
        """
        now = datetime.now()
        if created_at is None:
            created_at = now
        if updated_at is None:
            updated_at = now
        created_at = self._ensure_datetime(created_at)
        updated_at = self._ensure_datetime(updated_at)
        record_id = str(uuid.uuid4())
        
        with self._lock, self._engine.begin() as conn:
            stmt = insert(self.history_table).values(
                id=record_id,
                memory_id=memory_id,
                old_memory=old_memory,
                new_memory=new_memory,
                new_value=new_memory,  # Keep for compatibility
                event=event,
                created_at=created_at,
                updated_at=updated_at,
                is_deleted=is_deleted,
                actor_id=actor_id,
                role=role,
            )
            conn.execute(stmt)
        return record_id

    def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get all history records for a specific memory ID."""
        with self._lock, self._get_connection() as conn:
            stmt = (
                select(self.history_table)
                .where(self.history_table.c.memory_id == memory_id)
                .order_by(self.history_table.c.created_at.desc())
            )
            result = conn.execute(stmt)
            return [dict(row._mapping) for row in result]

    def get_all_history(self) -> List[Dict[str, Any]]:
        """Get all history records."""
        with self._lock, self._get_connection() as conn:
            stmt = select(self.history_table).order_by(
                self.history_table.c.created_at.desc()
            )
            result = conn.execute(stmt)
            return [dict(row._mapping) for row in result]

    def reset(self) -> bool:
        """Reset the database by dropping and recreating all tables."""
        try:
            with self._lock:
                self.metadata.drop_all(self._engine)
                self.metadata.create_all(self._engine)
            return True
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            return False

    def close(self):
        """Close the database connection."""
        if hasattr(self, "_engine"):
            self._engine.dispose()

    def __del__(self):
        self.close()


class SQLiteManager:
    """
    Simple SQLite manager for backward compatibility and simple use cases.
    Includes the new group chat functionality from upstream.
    """
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._migrate_history_table()
        self._create_history_table()

    def _migrate_history_table(self) -> None:
        """
        If a pre-existing history table had the old group-chat columns,
        rename it, create the new schema, copy the intersecting data, then
        drop the old table.
        """
        with self._lock:
            try:
                # Start a transaction
                self.connection.execute("BEGIN")
                cur = self.connection.cursor()

                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='history'")
                if cur.fetchone() is None:
                    self.connection.execute("COMMIT")
                    return  # nothing to migrate

                cur.execute("PRAGMA table_info(history)")
                old_cols = {row[1] for row in cur.fetchall()}

                expected_cols = {
                    "id",
                    "memory_id",
                    "old_memory",
                    "new_memory",
                    "event",
                    "created_at",
                    "updated_at",
                    "is_deleted",
                    "actor_id",
                    "role",
                }

                if old_cols == expected_cols:
                    self.connection.execute("COMMIT")
                    return

                logger.info("Migrating history table to new schema (no convo columns).")

                # Clean up any existing history_old table from previous failed migration
                cur.execute("DROP TABLE IF EXISTS history_old")

                # Rename the current history table
                cur.execute("ALTER TABLE history RENAME TO history_old")

                # Create the new history table with updated schema
                cur.execute(
                    """
                    CREATE TABLE history (
                        id           TEXT PRIMARY KEY,
                        memory_id    TEXT,
                        old_memory   TEXT,
                        new_memory   TEXT,
                        event        TEXT,
                        created_at   DATETIME,
                        updated_at   DATETIME,
                        is_deleted   INTEGER,
                        actor_id     TEXT,
                        role         TEXT
                    )
                """
                )

                # Copy data from old table to new table
                intersecting = list(expected_cols & old_cols)
                if intersecting:
                    cols_csv = ", ".join(intersecting)
                    cur.execute(f"INSERT INTO history ({cols_csv}) SELECT {cols_csv} FROM history_old")

                # Drop the old table
                cur.execute("DROP TABLE history_old")
                self.connection.execute("COMMIT")
                logger.info("History table migration completed successfully.")

            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to migrate history table: {e}")
                raise

    def _create_history_table(self) -> None:
        """Create the history table if it doesn't exist."""
        with self._lock:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id           TEXT PRIMARY KEY,
                    memory_id    TEXT,
                    old_memory   TEXT,
                    new_memory   TEXT,
                    event        TEXT,
                    created_at   DATETIME,
                    updated_at   DATETIME,
                    is_deleted   INTEGER,
                    actor_id     TEXT,
                    role         TEXT
                )
            """
            )
            self.connection.commit()

    def add_history(
        self,
        memory_id: str,
        old_memory: Optional[str],
        new_memory: Optional[str],
        event: str,
        *,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        is_deleted: int = 0,
        actor_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    INSERT INTO history (
                        id, memory_id, old_memory, new_memory, event,
                        created_at, updated_at, is_deleted, actor_id, role
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        str(uuid.uuid4()),
                        memory_id,
                        old_memory,
                        new_memory,
                        event,
                        created_at,
                        updated_at,
                        is_deleted,
                        actor_id,
                        role,
                    ),
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to add history record: {e}")
                raise

    def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.connection.execute(
                """
                SELECT id, memory_id, old_memory, new_memory, event,
                       created_at, updated_at, is_deleted, actor_id, role
                FROM history
                WHERE memory_id = ?
                ORDER BY created_at DESC
            """,
                (memory_id,),
            )
            columns = [col[0] for col in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def get_all_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.connection.execute(
                """
                SELECT id, memory_id, old_memory, new_memory, event,
                       created_at, updated_at, is_deleted, actor_id, role
                FROM history
                ORDER BY created_at DESC
            """
            )
            columns = [col[0] for col in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def reset(self) -> bool:
        """Reset the database by dropping and recreating the history table."""
        try:
            with self._lock:
                self.connection.execute("DROP TABLE IF EXISTS history")
                self.connection.commit()
                self._create_history_table()
            return True
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            return False

    def close(self):
        """Close the database connection."""
        if hasattr(self, "connection") and self.connection:
            self.connection.close()

    def __del__(self):
        self.close()