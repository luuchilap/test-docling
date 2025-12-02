"""
Database module for storing file metadata
Uses SQLite for simplicity (no additional setup required)
"""
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "files.db")


def init_database():
    """Initialize the database and create tables if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            chunks_count INTEGER NOT NULL,
            vectors_count INTEGER NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'completed'
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"✓ Database initialized: {DB_PATH}")


def save_file_metadata(
    file_id: str,
    filename: str,
    file_type: str,
    file_size: int,
    chunks_count: int,
    vectors_count: int,
    status: str = "completed"
) -> None:
    """Save file metadata to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO files (file_id, filename, file_type, file_size, chunks_count, vectors_count, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (file_id, filename, file_type, file_size, chunks_count, vectors_count, status))
        
        conn.commit()
        print(f"  ✓ File metadata saved to database")
    except sqlite3.IntegrityError:
        # File ID already exists, update instead
        cursor.execute("""
            UPDATE files 
            SET filename = ?, file_type = ?, file_size = ?, chunks_count = ?, vectors_count = ?, status = ?
            WHERE file_id = ?
        """, (filename, file_type, file_size, chunks_count, vectors_count, status, file_id))
        conn.commit()
        print(f"  ✓ File metadata updated in database")
    finally:
        conn.close()


def get_file_metadata(file_id: str) -> Optional[Dict]:
    """Get metadata for a specific file"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM files WHERE file_id = ?", (file_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def list_all_files(limit: int = 100) -> List[Dict]:
    """List all uploaded files"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM files 
        ORDER BY uploaded_at DESC 
        LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def delete_file_metadata(file_id: str) -> bool:
    """Delete file metadata from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    return deleted


def get_file_statistics() -> Dict:
    """Get overall statistics about uploaded files"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) as total_files FROM files")
    total_files = cursor.fetchone()[0]
    
    cursor.execute("SELECT SUM(chunks_count) as total_chunks FROM files")
    total_chunks = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT SUM(vectors_count) as total_vectors FROM files")
    total_vectors = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT SUM(file_size) as total_size FROM files")
    total_size = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return {
        "total_files": total_files,
        "total_chunks": total_chunks,
        "total_vectors": total_vectors,
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2) if total_size else 0
    }

