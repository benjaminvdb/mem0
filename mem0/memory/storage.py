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
            db_url: Complete database connection URL
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
            try:
                master_engine = create_engine(
                    master_url.render_as_string(hide_password=False),
                    isolation_level="AUTOCOMMIT",
                    **kwargs,
                )
                with master_engine.connect() as conn:
                    result = conn.execute(
                        text("SELECT 1 FROM pg_database WHERE datname=:dbname"),
                        {"dbname": db_name},
                    )
                    if result.scalar() is None:
                        conn.execute(text(f"CREATE DATABASE {db_name}"))
            except Exception as e:
                raise ConnectionError(
                    f"Failed to connect to PostgreSQL or create database: {e}"
                ) from e
            finally:
                master_engine.dispose()

        # Create engine for all database types
        self._engine = create_engine(db_url, poolclass=QueuePool, **kwargs)

        # Check the engine by performing a simple connection test
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            raise ConnectionError(f"Failed to establish connection with engine: {e}")

        # Set up database schema using SQLAlchemy MetaData
        self.metadata = MetaData()
        self.history_table = Table(
            "history",
            self.metadata,
            Column("id", String(36), primary_key=True),
            Column("memory_id", String(255), index=True),
            Column("old_memory", String),
            Column("new_memory", String),
            Column("new_value", String),
            Column("event", String(255)),
            Column("created_at", DateTime),
            Column("updated_at", DateTime, index=True),
            Column("is_deleted", Integer, default=0),
        )

        # Migrate and create table
        self._migrate_history_table()
        with self._get_connection() as conn:
            self._create_history_table_sqlalchemy(conn)

    @contextmanager
    def _get_connection(self) -> Generator[Connection, None, None]:
        """Context manager for SQLAlchemy connections."""
        with self._engine.connect() as conn:
            yield conn

    def _create_history_table_sqlalchemy(self, conn: Connection) -> None:
        """
        Create the 'history' table and indexes using raw SQL.
        This function supports SQLite, PostgreSQL, and MySQL by adjusting types and index syntax.
        """
        # Determine SQL types and index syntax based on database type
        if self.db_type == "sqlite":
            id_type = "TEXT"
            memory_id_type = "TEXT"
            event_type = "TEXT"
            datetime_type = "DATETIME"
            index_if_not_exists = "IF NOT EXISTS"
        elif self.db_type == "postgresql":
            id_type = "VARCHAR(36)"
            memory_id_type = "VARCHAR(255)"
            event_type = "VARCHAR(255)"
            datetime_type = "TIMESTAMP"
            index_if_not_exists = "IF NOT EXISTS"
        elif self.db_type == "mysql":
            id_type = "VARCHAR(36)"
            memory_id_type = "VARCHAR(255)"
            event_type = "VARCHAR(255)"
            datetime_type = "DATETIME"
            index_if_not_exists = (
                ""  # MySQL does not support IF NOT EXISTS for indexes
            )
        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}")

        # Create the history table if it does not exist
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS history (
                    id {id_type} PRIMARY KEY,
                    memory_id {memory_id_type},
                    old_memory TEXT,
                    new_memory TEXT,
                    new_value TEXT,
                    event {event_type},
                    created_at {datetime_type},
                    updated_at {datetime_type},
                    is_deleted INTEGER DEFAULT 0
                )
                """
            )
        )

        # Create indexes for memory_id and updated_at columns
        if self.db_type in ("sqlite", "postgresql"):
            conn.execute(
                text(
                    f"CREATE INDEX {index_if_not_exists} idx_history_memory_id ON history(memory_id)"
                )
            )
            conn.execute(
                text(
                    f"CREATE INDEX {index_if_not_exists} idx_history_updated_at ON history(updated_at)"
                )
            )
        elif self.db_type == "mysql":
            # MySQL does not support IF NOT EXISTS for indexes, so check manually
            idxs = conn.execute(
                text(
                    "SHOW INDEX FROM history WHERE Key_name = 'idx_history_memory_id'"
                )
            ).fetchall()
            if not idxs:
                conn.execute(
                    text("CREATE INDEX idx_history_memory_id ON history(memory_id)")
                )
            idxs2 = conn.execute(
                text(
                    "SHOW INDEX FROM history WHERE Key_name = 'idx_history_updated_at'"
                )
            ).fetchall()
            if not idxs2:
                conn.execute(
                    text(
                        "CREATE INDEX idx_history_updated_at ON history(updated_at)"
                    )
                )

    def _migrate_history_table(self) -> None:
        """
        Migrate the 'history' table schema if needed.
        For SQLite, checks for missing columns and migrates data if the schema has changed.
        For other databases, assumes migrations are handled externally.
        """
        with self._lock:
            if self.db_type == "sqlite":
                with self._get_connection() as conn:
                    # Check if the table exists
                    result = conn.execute(
                        text(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name='history'"
                        )
                    ).fetchone()
                    if result:
                        # Get current schema
                        rows = conn.execute(
                            text("PRAGMA table_info(history)")
                        ).fetchall()
                        current_schema = {row[1]: row[2] for row in rows}
                        expected_schema = {
                            "id": "TEXT",
                            "memory_id": "TEXT",
                            "old_memory": "TEXT",
                            "new_memory": "TEXT",
                            "new_value": "TEXT",
                            "event": "TEXT",
                            "created_at": "DATETIME",
                            "updated_at": "DATETIME",
                            "is_deleted": "INTEGER",
                        }
                        # Find missing columns
                        missing_columns = set(expected_schema.keys()) - set(
                            current_schema.keys()
                        )
                        if missing_columns:
                            # If schema changed, rename old table, create new one, and migrate data
                            conn.execute(
                                text("ALTER TABLE history RENAME TO old_history")
                            )
                            self._create_history_table_sqlalchemy(conn)
                            common_columns = list(
                                set(current_schema.keys())
                                & set(expected_schema.keys())
                            )
                            cols = ", ".join(common_columns)
                            conn.execute(
                                text(
                                    f"INSERT INTO history ({cols}) SELECT {cols} FROM old_history"
                                )
                            )
                            conn.execute(text("DROP TABLE old_history"))
            else:
                # For PostgreSQL/MySQL, assume migrations are handled externally (e.g., Alembic)
                with self._engine.begin() as conn:
                    inspector = inspect(conn)
                    tables = inspector.get_table_names()
                    if "history" in tables:
                        pass  # No-op for now

    def _create_history_table(self) -> None:
        """
        Create the 'history' table if it does not exist.
        Uses raw SQL for SQLite/MySQL/PostgreSQL, otherwise uses SQLAlchemy metadata.
        """
        with self._lock:
            if self.db_type in ("sqlite", "postgresql", "mysql"):
                with self._get_connection() as conn:
                    self._create_history_table_sqlalchemy(conn)
            else:
                self.metadata.create_all(self._engine, tables=[self.history_table])

    def _ensure_datetime(self, dt: Any) -> datetime:
        """
        Ensure the input is a datetime object.
        Converts from ISO format string if needed, otherwise raises TypeError.
        """
        if isinstance(dt, datetime):
            return dt
        if isinstance(dt, str):
            try:
                return datetime.fromisoformat(dt)
            except Exception as e:
                raise TypeError(f"Invalid datetime string provided: {dt}") from e
        raise TypeError(
            "Value must be a datetime object or an ISO format datetime string."
        )

    def add_history(
        self,
        memory_id: str,
        old_memory: str,
        new_memory: str,
        event: str,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        is_deleted: int = 0,
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
                new_value=new_memory,
                event=event,
                created_at=created_at,
                updated_at=updated_at,
                is_deleted=is_deleted,
            )
            conn.execute(stmt)
        return record_id

    def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """
        Get all non-deleted history records for a specific memory ID, ordered by updated_at ascending.
        Returns:
            List of history records as dictionaries.
        """
        with self._lock, self._engine.connect() as conn:
            query = (
                select(
                    self.history_table.c.id,
                    self.history_table.c.memory_id,
                    self.history_table.c.old_memory,
                    self.history_table.c.new_memory,
                    self.history_table.c.event,
                    self.history_table.c.created_at,
                    self.history_table.c.updated_at,
                )
                .where(
                    self.history_table.c.memory_id == memory_id,
                    self.history_table.c.is_deleted == 0,
                )
                .order_by(self.history_table.c.updated_at.asc())
            )
            result = conn.execute(query)
            rows = result.fetchall()
            return [
                {
                    "id": row[0],
                    "memory_id": row[1],
                    "old_memory": row[2],
                    "new_memory": row[3],
                    "event": row[4],
                    "created_at": row[5],
                    "updated_at": row[6],
                }
                for row in rows
            ]

    def delete_history(self, memory_id: str) -> int:
        """
        Soft delete all history records for a specific memory ID (set is_deleted=1).
        Returns:
            Number of records marked as deleted.
        """
        with self._lock, self._engine.begin() as conn:
            stmt = (
                update(self.history_table)
                .where(
                    self.history_table.c.memory_id == memory_id,
                    self.history_table.c.is_deleted == 0,
                )
                .values(is_deleted=1)
            )
            result = conn.execute(stmt)
            return result.rowcount

    def reset(self) -> None:
        """
        Reset the database by dropping and recreating the history table.
        Handles DROP TABLE syntax differences between SQLite, PostgreSQL, and MySQL.
        """
        with self._lock, self._engine.begin() as conn:
            if self.db_type == "sqlite":
                conn.execute(text("DROP TABLE IF EXISTS history"))
            elif self.db_type == "postgresql":
                conn.execute(text("DROP TABLE IF EXISTS history CASCADE"))
            elif self.db_type == "mysql":
                conn.execute(text("DROP TABLE IF EXISTS history"))
            self._create_history_table_sqlalchemy(conn)

    def close(self) -> None:
        """
        Properly close and dispose of the SQLAlchemy engine.
        """
        if self._engine:
            self._engine.dispose()
            self._engine = None

    def __enter__(self):
        """
        Enable context manager support for the database manager.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Ensure connections are closed when exiting the context.
        """
        self.close()
        return False
