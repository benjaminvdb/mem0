import datetime
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, List, Optional, Union

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
    select,
    text,
    update,
)
from sqlalchemy.engine import Connection
from sqlalchemy.pool import QueuePool


class DatabaseType(Enum):
    """Supported database types"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    
class HistoryDBConfig(BaseModel):
    db_type: DatabaseType = Field(
        DatabaseType.SQLITE,
        description="Type of database (SQLite, PostgreSQL, MySQL)"
    )
    db_url: str = Field(
        "sqlite:///:memory:",
        description="Complete database connection URL"
    ),


class SQLDatabaseManager:
    """
    Database manager that supports SQLite, PostgreSQL, and MySQL databases
    with a consistent API across all database types.
    """
    
    def __init__(
        self, 
        db_type: DatabaseType = DatabaseType.SQLITE,
        db_url: str = "sqlite:///:memory:",
        **kwargs
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
        self._engine = None
        self.connection = None
        
        # Process URLs for SQLite that might be direct file paths
        if db_type == DatabaseType.SQLITE:
            # Check if the URL is just a file path without the sqlite:/// schema
            if not db_url.startswith('sqlite:///') and not db_url.startswith('sqlite://'):
                # Automatically add the schema prefix for SQLite
                if db_url == ':memory:' or not db_url:
                    db_url = "sqlite:///:memory:"
                else:
                    # Handle both relative and absolute paths
                    db_url = f"sqlite:///{db_url}"
        
        # Connect to appropriate database
        self._engine = create_engine(db_url, poolclass=QueuePool, **kwargs)
        
        # For SQLite, also create direct connection for compatibility
        if db_type == DatabaseType.SQLITE:
            # Extract file path from URL
            if db_url.startswith('sqlite:///'):
                sqlite_path = db_url.replace('sqlite:///', '')
            else:
                sqlite_path = db_url.replace('sqlite://', '')
                
            if not sqlite_path or sqlite_path == ':memory:':
                sqlite_path = ":memory:"
            self.connection = sqlite3.connect(sqlite_path, check_same_thread=False)
        
        # Set up database schema
        self.metadata = MetaData()
        
        # Define history table schema
        self.history_table = Table(
            'history', self.metadata,
            Column('id', String(36), primary_key=True),
            Column('memory_id', String(255), index=True),
            Column('old_memory', String),
            Column('new_memory', String),
            Column('new_value', String),
            Column('event', String(255)),
            Column('created_at', DateTime),
            Column('updated_at', DateTime, index=True),
            Column('is_deleted', Integer, default=0)
        )
        
        # Initialize database
        self._migrate_history_table()
        self._create_history_table()

    @contextmanager
    def _get_connection(self) -> Union[Connection, sqlite3.Connection]:
        """Context manager for database connections."""
        if self.db_type == DatabaseType.SQLITE and self.connection:
            # Use existing direct connection for SQLite
            with self._lock:
                yield self.connection
        else:
            # Use SQLAlchemy connection pool for PostgreSQL/MySQL
            conn = self._engine.connect()
            try:
                yield conn
            finally:
                conn.close()

    def _migrate_history_table(self) -> None:
        """Migrate history table schema if needed."""
        with self._lock:
            if self.db_type == DatabaseType.SQLITE:
                # SQLite-specific migration logic
                with self.connection:
                    cursor = self.connection.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='history'")
                    table_exists = cursor.fetchone() is not None
                    
                    if table_exists:
                        # Check and update schema if needed
                        cursor.execute("PRAGMA table_info(history)")
                        current_schema = {row[1]: row[2] for row in cursor.fetchall()}
                        
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
                        
                        # Simplified migration - just check if columns exist
                        missing_columns = set(expected_schema.keys()) - set(current_schema.keys())
                        if missing_columns:
                            # If columns are missing, create a new table and migrate data
                            cursor.execute("ALTER TABLE history RENAME TO old_history")
                            self._create_history_table_sqlite(cursor)
                            
                            # Map old columns to new columns for data migration
                            old_columns = list(current_schema.keys())
                            new_columns = list(expected_schema.keys())
                            common_columns = list(set(old_columns) & set(new_columns))
                            
                            # Copy data from old format to new format
                            cursor.execute(
                                f"""
                                INSERT INTO history ({', '.join(common_columns)})
                                SELECT {', '.join(common_columns)}
                                FROM old_history
                                """
                            )
                            
                            cursor.execute("DROP TABLE old_history")
                            self.connection.commit()
            else:
                # PostgreSQL/MySQL migration using SQLAlchemy
                with self._engine.begin() as conn:
                    inspector = self._engine.dialect.get_inspector(conn)
                    tables = inspector.get_table_names()
                    if 'history' in tables:
                        # Here we would implement more sophisticated migration logic
                        # For PostgreSQL/MySQL, we typically use migrations frameworks
                        # This is a simplified example
                        pass
    
    def _create_history_table_sqlite(self, cursor) -> None:
        """Create history table for SQLite."""
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id TEXT PRIMARY KEY,
                memory_id TEXT,
                old_memory TEXT,
                new_memory TEXT,
                new_value TEXT,
                event TEXT,
                created_at DATETIME,
                updated_at DATETIME,
                is_deleted INTEGER DEFAULT 0
            )
            """
        )
        
        # Create index for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_memory_id ON history(memory_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_updated_at ON history(updated_at)")

    def _create_history_table(self) -> None:
        """Create history table if it doesn't exist."""
        with self._lock:
            if self.db_type == DatabaseType.SQLITE:
                with self.connection:
                    self._create_history_table_sqlite(self.connection.cursor())
            else:
                # Use SQLAlchemy to create table for PostgreSQL/MySQL
                self.metadata.create_all(self._engine, tables=[self.history_table])

    def add_history(
        self,
        memory_id: str,
        old_memory: str,
        new_memory: str,
        event: str,
        created_at: Optional[datetime.datetime] = None,
        updated_at: Optional[datetime.datetime] = None,
        is_deleted: int = 0,
    ) -> str:
        """
        Add a history record to the database.
        
        Args:
            memory_id: ID of the memory
            old_memory: Previous memory state
            new_memory: New memory state
            event: Type of event
            created_at: Creation timestamp (defaults to current time)
            updated_at: Update timestamp (defaults to current time)
            is_deleted: Deletion flag (0=not deleted)
            
        Returns:
            The ID of the newly created history record
        """
        # Set default timestamps if not provided
        now = datetime.datetime.now()
        if created_at is None:
            created_at = now
        if updated_at is None:
            updated_at = now
        
        record_id = str(uuid.uuid4())
        
        with self._lock:
            if self.db_type == DatabaseType.SQLITE:
                # SQLite-specific implementation
                with self.connection:
                    self.connection.execute(
                        """
                        INSERT INTO history (id, memory_id, old_memory, new_memory, new_value, event, created_at, updated_at, is_deleted)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record_id,
                            memory_id,
                            old_memory,
                            new_memory,
                            new_memory,  # new_value field
                            event,
                            created_at,
                            updated_at,
                            is_deleted,
                        ),
                    )
            else:
                # PostgreSQL/MySQL implementation using SQLAlchemy
                with self._engine.begin() as conn:
                    stmt = insert(self.history_table).values(
                        id=record_id,
                        memory_id=memory_id,
                        old_memory=old_memory,
                        new_memory=new_memory,
                        new_value=new_memory,
                        event=event,
                        created_at=created_at,
                        updated_at=updated_at,
                        is_deleted=is_deleted
                    )
                    conn.execute(stmt)
        
        return record_id

    def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """
        Get history records for a specific memory ID.
        
        Args:
            memory_id: ID of the memory to retrieve history for
            
        Returns:
            List of history records as dictionaries
        """
        with self._lock:
            if self.db_type == DatabaseType.SQLITE:
                # SQLite-specific implementation
                cursor = self.connection.execute(
                    """
                    SELECT id, memory_id, old_memory, new_memory, event, created_at, updated_at
                    FROM history
                    WHERE memory_id = ? AND is_deleted = 0
                    ORDER BY updated_at ASC
                    """,
                    (memory_id,),
                )
                rows = cursor.fetchall()
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
            else:
                # PostgreSQL/MySQL implementation using SQLAlchemy
                with self._engine.connect() as conn:
                    query = select(
                        self.history_table.c.id,
                        self.history_table.c.memory_id,
                        self.history_table.c.old_memory,
                        self.history_table.c.new_memory,
                        self.history_table.c.event,
                        self.history_table.c.created_at,
                        self.history_table.c.updated_at
                    ).where(
                        self.history_table.c.memory_id == memory_id,
                        self.history_table.c.is_deleted == 0
                    ).order_by(
                        self.history_table.c.updated_at.asc()
                    )
                    
                    result = conn.execute(query)
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
                        for row in result.fetchall()
                    ]

    def delete_history(self, memory_id: str) -> int:
        """
        Soft delete history records for a specific memory ID.
        
        Args:
            memory_id: ID of the memory to delete history for
            
        Returns:
            Number of records deleted
        """
        with self._lock:
            if self.db_type == DatabaseType.SQLITE:
                # SQLite-specific implementation
                with self.connection:
                    cursor = self.connection.execute(
                        """
                        UPDATE history
                        SET is_deleted = 1
                        WHERE memory_id = ? AND is_deleted = 0
                        """,
                        (memory_id,),
                    )
                    return cursor.rowcount
            else:
                # PostgreSQL/MySQL implementation using SQLAlchemy
                with self._engine.begin() as conn:
                    stmt = update(self.history_table).where(
                        self.history_table.c.memory_id == memory_id,
                        self.history_table.c.is_deleted == 0
                    ).values(is_deleted=1)
                    
                    result = conn.execute(stmt)
                    return result.rowcount

    def reset(self) -> None:
        """Reset database by dropping and recreating the history table."""
        with self._lock:
            if self.db_type == DatabaseType.SQLITE:
                # SQLite-specific implementation
                with self.connection:
                    self.connection.execute("DROP TABLE IF EXISTS history")
                    self._create_history_table_sqlite(self.connection.cursor())
            else:
                # PostgreSQL/MySQL implementation using SQLAlchemy
                with self._engine.begin() as conn:
                    conn.execute(text("DROP TABLE IF EXISTS history"))
                    self.metadata.create_all(self._engine, tables=[self.history_table])

    def close(self) -> None:
        """Close database connections properly."""
        if self.db_type == DatabaseType.SQLITE and self.connection:
            self.connection.close()
            self.connection = None
        
        if self._engine:
            self._engine.dispose()
            self._engine = None

    def __enter__(self):
        """Enable context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close connections when exiting context."""
        self.close()
        
        
# # Example 1: SQLite in-memory database
# db = SQLDatabaseManager()  # Uses defaults: SQLite, in-memory

# # Example 2: SQLite file database
# db = SQLDatabaseManager(
#     db_type=DatabaseType.SQLITE,
#     url="sqlite:///path/to/database.db"
# )

# # Example 3: PostgreSQL database
# db = SQLDatabaseManager(
#     db_type=DatabaseType.POSTGRESQL,
#     url="postgresql+psycopg2://username:password@localhost:5432/mydatabase"
# )

# # Example 4: MySQL database
# db = SQLDatabaseManager(
#     db_type=DatabaseType.MYSQL,
#     url="mysql+pymysql://username:password@localhost:3306/mydatabase"
# )

# # Using context manager
# with SQLDatabaseManager(url="sqlite:///temp.db") as db:
#     db.add_history("memory123", "old data", "new data", "update")
#     records = db.get_history("memory123")
#     print(f"Found {len(records)} history records")
