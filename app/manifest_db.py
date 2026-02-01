"""
SQLite-based Manifest for Fast File Tracking.

Replaces JSON manifest for better performance with large file counts.
Provides O(1) lookups and efficient queries for modified files.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Any

logger = logging.getLogger("rag")


class SQLiteManifest:
    """
    SQLite-based manifest for tracking indexed files.
    
    Features:
    - O(1) file lookups by path
    - Efficient queries for files modified since date
    - Atomic updates with transactions
    - Backward compatible with JSON manifest
    """
    
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                );
                
                CREATE TABLE IF NOT EXISTS files (
                    filepath TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_hash TEXT,
                    mtime REAL,
                    size INTEGER,
                    indexed_at TEXT,
                    chunk_count INTEGER DEFAULT 0
                );
                
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_files_mtime ON files(mtime);
                CREATE INDEX IF NOT EXISTS idx_files_indexed_at ON files(indexed_at);
            """)
            
            # Check/set schema version
            cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
            row = cursor.fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (self.SCHEMA_VERSION,)
                )
    
    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with proper cleanup."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    # ========================================================================
    # File Operations
    # ========================================================================
    
    def get_file(self, filepath: str) -> dict | None:
        """Get file info by path. O(1) lookup."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM files WHERE filepath = ?",
                (filepath,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def file_exists(self, filepath: str) -> bool:
        """Check if file is in manifest."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM files WHERE filepath = ? LIMIT 1",
                (filepath,)
            )
            return cursor.fetchone() is not None
    
    def needs_indexing(self, filepath: Path) -> bool:
        """
        Check if a file needs to be (re)indexed.
        
        Returns True if:
        - File is not in manifest (new file)
        - File modification time has changed
        - File hash has changed
        """
        if not filepath.exists():
            return False
        
        stored = self.get_file(str(filepath))
        if stored is None:
            return True
        
        try:
            stat = filepath.stat()
            
            # Quick check: modification time
            if stat.st_mtime != stored.get("mtime"):
                # Verify with hash to avoid false positives
                current_hash = self.compute_file_hash(filepath)
                return current_hash != stored.get("file_hash")
            
            return False
        except OSError:
            return False
    
    def mark_indexed(
        self,
        filepath: Path,
        chunk_count: int,
        file_hash: str | None = None,
    ) -> None:
        """Mark a file as successfully indexed."""
        try:
            stat = filepath.stat()
            
            with self._connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO files 
                    (filepath, filename, file_hash, mtime, size, indexed_at, chunk_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(filepath),
                    filepath.name,
                    file_hash or self.compute_file_hash(filepath),
                    stat.st_mtime,
                    stat.st_size,
                    datetime.now().isoformat(),
                    chunk_count,
                ))
        except OSError as e:
            logger.warning("Failed to mark file as indexed: %s", e)
    
    def mark_deleted(self, filepath: Path) -> None:
        """Remove a file from the manifest."""
        with self._connection() as conn:
            conn.execute(
                "DELETE FROM files WHERE filepath = ?",
                (str(filepath),)
            )
    
    def mark_deleted_batch(self, filepaths: list[str]) -> int:
        """Remove multiple files from manifest. Returns count deleted."""
        with self._connection() as conn:
            cursor = conn.executemany(
                "DELETE FROM files WHERE filepath = ?",
                [(fp,) for fp in filepaths]
            )
            return cursor.rowcount
    
    # ========================================================================
    # Queries
    # ========================================================================
    
    def get_all_files(self) -> list[dict]:
        """Get all indexed files."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM files ORDER BY indexed_at DESC"
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_files_modified_since(self, since: datetime) -> list[dict]:
        """Get files indexed after a certain date."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM files WHERE indexed_at > ? ORDER BY indexed_at DESC",
                (since.isoformat(),)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_files_by_extension(self, extension: str) -> list[dict]:
        """Get files with a specific extension."""
        # Ensure extension starts with dot
        if not extension.startswith("."):
            extension = "." + extension
        
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM files WHERE filename LIKE ?",
                (f"%{extension}",)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_files_in_directory(self, directory: str) -> list[dict]:
        """Get files within a specific directory."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM files WHERE filepath LIKE ?",
                (f"{directory}%",)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_indexed_filepaths(self) -> set[str]:
        """Get set of all indexed file paths."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT filepath FROM files")
            return {row["filepath"] for row in cursor.fetchall()}
    
    def find_deleted_files(self, current_files: set[str]) -> set[str]:
        """Find files in manifest that no longer exist."""
        indexed = self.get_indexed_filepaths()
        return indexed - current_files
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    def get_stats(self) -> dict:
        """Get statistics about indexed files."""
        with self._connection() as conn:
            # File count
            cursor = conn.execute("SELECT COUNT(*) as count FROM files")
            file_count = cursor.fetchone()["count"]
            
            # Total chunks
            cursor = conn.execute(
                "SELECT COALESCE(SUM(chunk_count), 0) as total FROM files"
            )
            total_chunks = cursor.fetchone()["total"]
            
            # Total size
            cursor = conn.execute(
                "SELECT COALESCE(SUM(size), 0) as total FROM files"
            )
            total_size = cursor.fetchone()["total"]
            
            # Last full scan
            last_scan = self.get_metadata("last_full_scan")
            
            return {
                "total_files": file_count,
                "total_chunks": total_chunks,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "last_full_scan": last_scan,
            }
    
    def get_extension_counts(self) -> dict[str, int]:
        """Get file counts by extension."""
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    LOWER(SUBSTR(filename, INSTR(filename, '.'))) as ext,
                    COUNT(*) as count
                FROM files 
                WHERE filename LIKE '%.%'
                GROUP BY ext
                ORDER BY count DESC
            """)
            return {row["ext"]: row["count"] for row in cursor.fetchall()}
    
    # ========================================================================
    # Metadata
    # ========================================================================
    
    def get_metadata(self, key: str) -> str | None:
        """Get a metadata value."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT value FROM metadata WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            return row["value"] if row else None
    
    def set_metadata(self, key: str, value: str) -> None:
        """Set a metadata value."""
        with self._connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (key, value)
            )
    
    def mark_full_scan_complete(self) -> None:
        """Record that a full scan was completed."""
        self.set_metadata("last_full_scan", datetime.now().isoformat())
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    @staticmethod
    def compute_file_hash(filepath: Path) -> str:
        """Compute MD5 hash of file contents."""
        hasher = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""
    
    def vacuum(self) -> None:
        """Compact the database file."""
        with self._connection() as conn:
            conn.execute("VACUUM")
    
    def export_to_json(self) -> dict:
        """Export manifest to JSON-compatible dict."""
        files = self.get_all_files()
        return {
            "files": {
                f["filepath"]: {
                    "hash": f["file_hash"],
                    "mtime": f["mtime"],
                    "size": f["size"],
                    "indexed_at": f["indexed_at"],
                    "chunk_count": f["chunk_count"],
                }
                for f in files
            },
            "last_full_scan": self.get_metadata("last_full_scan"),
        }
    
    def import_from_json(self, data: dict) -> int:
        """Import from JSON manifest. Returns count imported."""
        files = data.get("files", {})
        count = 0
        
        with self._connection() as conn:
            for filepath, info in files.items():
                conn.execute("""
                    INSERT OR REPLACE INTO files 
                    (filepath, filename, file_hash, mtime, size, indexed_at, chunk_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    filepath,
                    Path(filepath).name,
                    info.get("hash"),
                    info.get("mtime"),
                    info.get("size"),
                    info.get("indexed_at"),
                    info.get("chunk_count", 0),
                ))
                count += 1
            
            # Import metadata
            if data.get("last_full_scan"):
                conn.execute(
                    "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                    ("last_full_scan", data["last_full_scan"])
                )
        
        return count


# ============================================================================
# Migration from JSON to SQLite
# ============================================================================

def migrate_json_to_sqlite(json_path: Path, db_path: Path) -> bool:
    """
    Migrate from JSON manifest to SQLite.
    
    Args:
        json_path: Path to existing JSON manifest
        db_path: Path for new SQLite database
    
    Returns:
        True if migration successful
    """
    import json
    
    if not json_path.exists():
        logger.info("No JSON manifest to migrate")
        return False
    
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        
        manifest = SQLiteManifest(db_path)
        count = manifest.import_from_json(data)
        
        logger.info("Migrated %d files from JSON to SQLite", count)
        
        # Rename old JSON file
        backup_path = json_path.with_suffix(".json.bak")
        json_path.rename(backup_path)
        logger.info("Old manifest backed up to %s", backup_path)
        
        return True
        
    except Exception as e:
        logger.error("Migration failed: %s", e)
        return False
