import sqlite3 from "sqlite3";
import { Pool as PgPool } from "pg";
import mysql from "mysql2/promise";
import { v4 as uuidv4 } from "uuid";

export enum DatabaseType {
  SQLITE = "sqlite",
  POSTGRESQL = "postgresql",
  MYSQL = "mysql",
}

export interface DatabaseConfig {
  dbType: DatabaseType;
  dbUrl: string;
}

export class SQLDatabaseManager {
  private dbType: DatabaseType;
  private sqliteDb: sqlite3.Database | null = null;
  private pgPool: PgPool | null = null;
  private mysqlPool: mysql.Pool | null = null;
  private lock = new Set<string>();

  constructor(
    dbType: DatabaseType = DatabaseType.SQLITE,
    dbUrl: string = ":memory:",
  ) {
    this.dbType = dbType;

    // Process URLs for SQLite that might be direct file paths
    if (dbType === DatabaseType.SQLITE) {
      // Check if the URL is just a file path without the sqlite:/// schema
      if (!dbUrl.startsWith("sqlite:///") && !dbUrl.startsWith("sqlite://")) {
        // Automatically add the schema prefix for SQLite
        if (dbUrl === ":memory:" || !dbUrl) {
          dbUrl = ":memory:";
        } else {
          // Extract the file path if it has sqlite:/// prefix
          dbUrl = dbUrl;
        }
      } else {
        // Extract the file path if it has sqlite:/// prefix
        dbUrl = dbUrl.replace("sqlite:///", "");
      }

      this.sqliteDb = new sqlite3.Database(dbUrl);
    } else if (dbType === DatabaseType.POSTGRESQL) {
      // For PostgreSQL, create a connection pool
      this.pgPool = new PgPool({ connectionString: dbUrl });
    } else if (dbType === DatabaseType.MYSQL) {
      // For MySQL, create a connection pool
      this.mysqlPool = mysql.createPool(dbUrl);
    }

    this.init().catch((err) =>
      console.error("Failed to initialize database:", err),
    );
  }

  private async init(): Promise<void> {
    switch (this.dbType) {
      case DatabaseType.SQLITE:
        await this.initSqlite();
        break;
      case DatabaseType.POSTGRESQL:
        await this.initPostgres();
        break;
      case DatabaseType.MYSQL:
        await this.initMysql();
        break;
    }
  }

  private async initSqlite(): Promise<void> {
    if (!this.sqliteDb) return;

    await this.runSqlite(`
      CREATE TABLE IF NOT EXISTS history (
        id TEXT PRIMARY KEY,
        memory_id TEXT,
        old_memory TEXT,
        new_memory TEXT,
        new_value TEXT,
        event TEXT,
        created_at TEXT,
        updated_at TEXT,
        is_deleted INTEGER DEFAULT 0
      )
    `);

    await this.runSqlite(
      "CREATE INDEX IF NOT EXISTS idx_history_memory_id ON history(memory_id)",
    );
    await this.runSqlite(
      "CREATE INDEX IF NOT EXISTS idx_history_updated_at ON history(updated_at)",
    );
  }

  private async initPostgres(): Promise<void> {
    if (!this.pgPool) return;

    await this.pgPool.query(`
      CREATE TABLE IF NOT EXISTS history (
        id UUID PRIMARY KEY,
        memory_id TEXT,
        old_memory TEXT,
        new_memory TEXT,
        new_value TEXT,
        event TEXT,
        created_at TIMESTAMP,
        updated_at TIMESTAMP,
        is_deleted INTEGER DEFAULT 0
      )
    `);

    await this.pgPool.query(
      "CREATE INDEX IF NOT EXISTS idx_history_memory_id ON history(memory_id)",
    );
    await this.pgPool.query(
      "CREATE INDEX IF NOT EXISTS idx_history_updated_at ON history(updated_at)",
    );
  }

  private async initMysql(): Promise<void> {
    if (!this.mysqlPool) return;

    await this.mysqlPool.query(`
      CREATE TABLE IF NOT EXISTS history (
        id VARCHAR(36) PRIMARY KEY,
        memory_id VARCHAR(255),
        old_memory TEXT,
        new_memory TEXT,
        new_value TEXT,
        event VARCHAR(255),
        created_at DATETIME,
        updated_at DATETIME,
        is_deleted INTEGER DEFAULT 0
      )
    `);

    // MySQL handles index creation differently
    try {
      await this.mysqlPool.query(
        "CREATE INDEX idx_history_memory_id ON history(memory_id)",
      );
    } catch (e) {
      // Index might already exist
    }

    try {
      await this.mysqlPool.query(
        "CREATE INDEX idx_history_updated_at ON history(updated_at)",
      );
    } catch (e) {
      // Index might already exist
    }
  }

  private async runSqlite(sql: string, params: any[] = []): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.sqliteDb)
        return reject(new Error("SQLite database not initialized"));

      this.sqliteDb.run(sql, params, (err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  }

  private async allSqlite(sql: string, params: any[] = []): Promise<any[]> {
    return new Promise((resolve, reject) => {
      if (!this.sqliteDb)
        return reject(new Error("SQLite database not initialized"));

      this.sqliteDb.all(sql, params, (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }

  async addHistory(
    memoryId: string,
    oldMemory: string | null,
    newMemory: string | null,
    event: string,
    createdAt?: string,
    updatedAt?: string,
    isDeleted: number = 0,
  ): Promise<string> {
    const id = uuidv4();
    const now = new Date().toISOString();

    if (!createdAt) createdAt = now;
    if (!updatedAt) updatedAt = now;

    switch (this.dbType) {
      case DatabaseType.SQLITE:
        if (!this.sqliteDb) throw new Error("SQLite database not initialized");

        await this.runSqlite(
          `INSERT INTO history 
          (id, memory_id, old_memory, new_memory, new_value, event, created_at, updated_at, is_deleted)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
          [
            id,
            memoryId,
            oldMemory,
            newMemory,
            newMemory, // new_value same as new_memory
            event,
            createdAt,
            updatedAt,
            isDeleted,
          ],
        );
        break;

      case DatabaseType.POSTGRESQL:
        if (!this.pgPool)
          throw new Error("PostgreSQL database not initialized");

        await this.pgPool.query(
          `INSERT INTO history 
          (id, memory_id, old_memory, new_memory, new_value, event, created_at, updated_at, is_deleted)
          VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)`,
          [
            id,
            memoryId,
            oldMemory,
            newMemory,
            newMemory, // new_value same as new_memory
            event,
            createdAt,
            updatedAt,
            isDeleted,
          ],
        );
        break;

      case DatabaseType.MYSQL:
        if (!this.mysqlPool) throw new Error("MySQL database not initialized");

        await this.mysqlPool.query(
          `INSERT INTO history 
          (id, memory_id, old_memory, new_memory, new_value, event, created_at, updated_at, is_deleted)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
          [
            id,
            memoryId,
            oldMemory,
            newMemory,
            newMemory, // new_value same as new_memory
            event,
            createdAt,
            updatedAt,
            isDeleted,
          ],
        );
        break;
    }

    return id;
  }

  async getHistory(memoryId: string): Promise<any[]> {
    switch (this.dbType) {
      case DatabaseType.SQLITE:
        if (!this.sqliteDb) throw new Error("SQLite database not initialized");

        const rows = await this.allSqlite(
          `SELECT id, memory_id, old_memory, new_memory, event, created_at, updated_at
          FROM history
          WHERE memory_id = ? AND is_deleted = 0
          ORDER BY updated_at ASC`,
          [memoryId],
        );

        return rows.map((row) => ({
          id: row.id,
          memory_id: row.memory_id,
          old_memory: row.old_memory,
          new_memory: row.new_memory,
          event: row.event,
          created_at: row.created_at,
          updated_at: row.updated_at,
        }));

      case DatabaseType.POSTGRESQL:
        if (!this.pgPool)
          throw new Error("PostgreSQL database not initialized");

        const pgResult = await this.pgPool.query(
          `SELECT id, memory_id, old_memory, new_memory, event, created_at, updated_at
          FROM history
          WHERE memory_id = $1 AND is_deleted = 0
          ORDER BY updated_at ASC`,
          [memoryId],
        );

        return pgResult.rows.map((row) => ({
          id: row.id,
          memory_id: row.memory_id,
          old_memory: row.old_memory,
          new_memory: row.new_memory,
          event: row.event,
          created_at: row.created_at,
          updated_at: row.updated_at,
        }));

      case DatabaseType.MYSQL:
        if (!this.mysqlPool) throw new Error("MySQL database not initialized");

        const [mysqlRows] = await this.mysqlPool.query(
          `SELECT id, memory_id, old_memory, new_memory, event, created_at, updated_at
          FROM history
          WHERE memory_id = ? AND is_deleted = 0
          ORDER BY updated_at ASC`,
          [memoryId],
        );

        return (mysqlRows as any[]).map((row) => ({
          id: row.id,
          memory_id: row.memory_id,
          old_memory: row.old_memory,
          new_memory: row.new_memory,
          event: row.event,
          created_at: row.created_at,
          updated_at: row.updated_at,
        }));

      default:
        return [];
    }
  }

  async deleteHistory(memoryId: string): Promise<number> {
    switch (this.dbType) {
      case DatabaseType.SQLITE:
        if (!this.sqliteDb) throw new Error("SQLite database not initialized");

        await this.runSqlite(
          `UPDATE history
          SET is_deleted = 1
          WHERE memory_id = ? AND is_deleted = 0`,
          [memoryId],
        );

        // SQLite doesn't easily return affected rows count
        return 1;

      case DatabaseType.POSTGRESQL:
        if (!this.pgPool)
          throw new Error("PostgreSQL database not initialized");

        const pgResult = await this.pgPool.query(
          `UPDATE history
          SET is_deleted = 1
          WHERE memory_id = $1 AND is_deleted = 0`,
          [memoryId],
        );

        return pgResult.rowCount || 0;

      case DatabaseType.MYSQL:
        if (!this.mysqlPool) throw new Error("MySQL database not initialized");

        const [mysqlResult] = await this.mysqlPool.query(
          `UPDATE history
          SET is_deleted = 1
          WHERE memory_id = ? AND is_deleted = 0`,
          [memoryId],
        );

        return (mysqlResult as any).affectedRows || 0;

      default:
        return 0;
    }
  }

  async reset(): Promise<void> {
    switch (this.dbType) {
      case DatabaseType.SQLITE:
        if (!this.sqliteDb) throw new Error("SQLite database not initialized");

        await this.runSqlite("DROP TABLE IF EXISTS history");
        await this.initSqlite();
        break;

      case DatabaseType.POSTGRESQL:
        if (!this.pgPool)
          throw new Error("PostgreSQL database not initialized");

        await this.pgPool.query("DROP TABLE IF EXISTS history");
        await this.initPostgres();
        break;

      case DatabaseType.MYSQL:
        if (!this.mysqlPool) throw new Error("MySQL database not initialized");

        await this.mysqlPool.query("DROP TABLE IF EXISTS history");
        await this.initMysql();
        break;
    }
  }

  close(): void {
    if (this.sqliteDb) {
      this.sqliteDb.close();
      this.sqliteDb = null;
    }

    if (this.pgPool) {
      this.pgPool.end();
      this.pgPool = null;
    }

    if (this.mysqlPool) {
      this.mysqlPool.end();
      this.mysqlPool = null;
    }
  }
}
