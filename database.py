"""SQLite persistence for JARVIS memories.

This module deliberately stays small: it owns schema migrations and raw database
operations, while ``memory.py`` owns embedding and retrieval policy.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE = PROJECT_ROOT / "data" / "jarvis.db"


def _utc_now():
    """Return an unambiguous, sortable UTC timestamp for a memory event."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect():
    database_path = Path(DATABASE)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(database_path)


def _column_names(connection):
    return {
        column[1]
        for column in connection.execute("PRAGMA table_info(memories)").fetchall()
    }


def initialize_database():
    """Create or additively migrate the memories table without touching data."""
    connection = _connect()
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS interaction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            embedding BLOB,
            memory_type TEXT NOT NULL DEFAULT 'FACT',
            confidence REAL NOT NULL DEFAULT 1.0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    # Older local databases contain only ``id`` and ``content`` (and sometimes
    # ``embedding``). Each migration is additive, preserving EmbeddingGemma data.
    columns = _column_names(connection)
    migrations = {
        "embedding": "ALTER TABLE memories ADD COLUMN embedding BLOB",
        "memory_type": (
            "ALTER TABLE memories ADD COLUMN memory_type TEXT NOT NULL "
            "DEFAULT 'FACT'"
        ),
        "confidence": (
            "ALTER TABLE memories ADD COLUMN confidence REAL NOT NULL "
            "DEFAULT 1.0"
        ),
        "created_at": "ALTER TABLE memories ADD COLUMN created_at TEXT",
        "updated_at": "ALTER TABLE memories ADD COLUMN updated_at TEXT",
    }

    for column, statement in migrations.items():
        if column not in columns:
            connection.execute(statement)

    now = _utc_now()
    connection.execute(
        "UPDATE memories SET created_at = ? "
        "WHERE created_at IS NULL OR created_at = ''",
        (now,),
    )
    connection.execute(
        "UPDATE memories SET updated_at = ? "
        "WHERE updated_at IS NULL OR updated_at = ''",
        (now,),
    )
    connection.commit()
    connection.close()


def add_memory(content, embedding=None, memory_type="FACT", confidence=1.0):
    now = _utc_now()
    connection = _connect()
    cursor = connection.execute(
        """
        INSERT INTO memories (
            content, embedding, memory_type, confidence, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (content, embedding, memory_type, confidence, now, now),
    )
    connection.commit()
    memory_id = cursor.lastrowid
    connection.close()
    return memory_id


def add_interaction(message):
    """Persist one privacy-filtered user interaction and its UTC timestamp."""
    connection = _connect()
    cursor = connection.execute(
        "INSERT INTO interaction_history (message, created_at) VALUES (?, ?)",
        (message, _utc_now()),
    )
    connection.commit()
    interaction_id = cursor.lastrowid
    connection.close()
    return interaction_id


def get_interactions():
    """Return interaction rows as (id, message, created_at)."""
    connection = _connect()
    interactions = connection.execute(
        "SELECT id, message, created_at FROM interaction_history ORDER BY id"
    ).fetchall()
    connection.close()
    return interactions


def get_memories():
    """Return the legacy (id, content) shape used by the current CLI."""
    connection = _connect()
    memories = connection.execute(
        "SELECT id, content FROM memories ORDER BY id"
    ).fetchall()
    connection.close()
    return memories


def get_memory(memory_id):
    connection = _connect()
    memory = connection.execute(
        """
        SELECT id, content, embedding, memory_type, confidence, created_at, updated_at
        FROM memories WHERE id = ?
        """,
        (memory_id,),
    ).fetchone()
    connection.close()
    return memory


def get_memory_by_exact_content(content):
    """Return an exact duplicate, if one is already stored."""
    connection = _connect()
    memory = connection.execute(
        """
        SELECT id, content, embedding, memory_type, confidence, created_at, updated_at
        FROM memories WHERE content = ?
        """,
        (content,),
    ).fetchone()
    connection.close()
    return memory


def get_memories_with_embeddings():
    """Return the existing retrieval shape: (id, content, embedding)."""
    connection = _connect()
    memories = connection.execute(
        "SELECT id, content, embedding FROM memories ORDER BY id"
    ).fetchall()
    connection.close()
    return memories


def update_memory(memory_id, content, embedding, memory_type, confidence):
    """Update a whole memory record and advance its lifecycle timestamp."""
    connection = _connect()
    cursor = connection.execute(
        """
        UPDATE memories
        SET content = ?, embedding = ?, memory_type = ?, confidence = ?, updated_at = ?
        WHERE id = ?
        """,
        (content, embedding, memory_type, confidence, _utc_now(), memory_id),
    )
    connection.commit()
    updated = cursor.rowcount == 1
    connection.close()
    return updated


def update_memory_embedding(memory_id, embedding):
    memory = get_memory(memory_id)
    if memory is None:
        return False

    return update_memory(
        memory_id,
        memory[1],
        embedding,
        memory[3],
        memory[4],
    )


def delete_memory(memory_id):
    connection = _connect()
    cursor = connection.execute(
        "DELETE FROM memories WHERE id = ?",
        (memory_id,),
    )
    connection.commit()
    deleted = cursor.rowcount == 1
    connection.close()
    return deleted


def find_memories(search):
    """Keep the CLI's current simple substring-search return format."""
    connection = _connect()
    memories = connection.execute(
        "SELECT content FROM memories WHERE content LIKE ? ORDER BY id",
        (f"%{search}%",),
    ).fetchall()
    connection.close()
    return memories


initialize_database()
