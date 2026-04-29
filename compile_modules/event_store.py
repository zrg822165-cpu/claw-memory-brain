"""Canonical event-chunk SQLite store."""
from __future__ import annotations

from pathlib import Path

import apsw

from .event_chunker import EventChunk
from .runtime_paths import DATA_DIR


DEFAULT_EVENT_DB_PATH = DATA_DIR / "event_chunks.sqlite"


def _open_event_db(db_path: Path = DEFAULT_EVENT_DB_PATH) -> apsw.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = apsw.Connection(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS canonical_chunks (
            id TEXT PRIMARY KEY,
            source_title TEXT NOT NULL,
            content TEXT NOT NULL,
            chunk_summary TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_canonical_chunks_source_title "
        "ON canonical_chunks(source_title)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_canonical_chunks_content_hash "
        "ON canonical_chunks(content_hash)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_canonical_chunks_status "
        "ON canonical_chunks(status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_canonical_chunks_created_at "
        "ON canonical_chunks(created_at)"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS high_feedback_state (
            chunk_id TEXT PRIMARY KEY,
            penalized INTEGER NOT NULL DEFAULT 0,
            last_feedback_at TEXT NOT NULL
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_delete_queue (
            chunk_id TEXT PRIMARY KEY,
            source_query TEXT NOT NULL DEFAULT '',
            feedback_note TEXT NOT NULL DEFAULT '',
            reported_at TEXT NOT NULL,
            review_status TEXT NOT NULL DEFAULT 'pending'
        );
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pending_delete_queue_review_status "
        "ON pending_delete_queue(review_status)"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS maintenance_runtime_state (
            state_key TEXT PRIMARY KEY,
            state_value TEXT NOT NULL
        );
    """)
    return conn


def _fetchall_dicts(
    conn: apsw.Connection,
    sql: str,
    parameters: tuple[object, ...] = (),
) -> list[dict[str, str]]:
    cursor = conn.cursor()
    cursor.execute(sql, parameters)
    first_row = cursor.fetchone()
    if first_row is None:
        return []
    columns = [column[0] for column in cursor.getdescription()]
    rows = [first_row, *cursor.fetchall()]
    return [{column: value for column, value in zip(columns, row, strict=True)} for row in rows]


def chunk_exists_by_hash(
    content_hash: str,
    db_path: Path = DEFAULT_EVENT_DB_PATH,
) -> bool:
    conn = _open_event_db(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM canonical_chunks WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        )
        row = cursor.fetchone()
        return row is not None
    finally:
        conn.close()


def save_event_chunks(
    chunks: list[EventChunk],
    db_path: Path = DEFAULT_EVENT_DB_PATH,
) -> tuple[Path, int]:
    conn = _open_event_db(db_path)
    inserted = 0
    try:
        with conn:
            cursor = conn.cursor()
            for chunk in chunks:
                cursor.execute(
                    "SELECT 1 FROM canonical_chunks WHERE content_hash = ? LIMIT 1",
                    (chunk.content_hash,),
                )
                exists = cursor.fetchone()
                if exists is not None:
                    continue
                cursor.execute("""
                    INSERT INTO canonical_chunks (
                        id, source_title, content, chunk_summary,
                        content_hash, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.id,
                    chunk.source_title,
                    chunk.content,
                    chunk.chunk_summary,
                    chunk.content_hash,
                    chunk.status,
                    chunk.created_at,
                    chunk.updated_at,
                ))
                inserted += 1
        return db_path, inserted
    except Exception:
        raise
    finally:
        conn.close()


def apply_ingest_changes(
    insert_chunks: list[EventChunk],
    delete_chunk_ids: list[str],
    db_path: Path = DEFAULT_EVENT_DB_PATH,
) -> tuple[Path, int, int, int]:
    conn = _open_event_db(db_path)
    inserted = 0
    deleted = 0
    skipped_duplicates = 0
    try:
        with conn:
            cursor = conn.cursor()
            for chunk_id in delete_chunk_ids:
                cursor.execute(
                    "DELETE FROM canonical_chunks WHERE id = ?",
                    (chunk_id,),
                )
                if conn.changes() > 0:
                    deleted += 1
                conn.execute(
                    "DELETE FROM high_feedback_state WHERE chunk_id = ?",
                    (chunk_id,),
                )
                conn.execute(
                    "DELETE FROM pending_delete_queue WHERE chunk_id = ?",
                    (chunk_id,),
                )

            for chunk in insert_chunks:
                cursor.execute(
                    "SELECT 1 FROM canonical_chunks WHERE content_hash = ? LIMIT 1",
                    (chunk.content_hash,),
                )
                exists = cursor.fetchone()
                if exists is not None:
                    skipped_duplicates += 1
                    continue
                cursor.execute("""
                    INSERT INTO canonical_chunks (
                        id, source_title, content, chunk_summary,
                        content_hash, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.id,
                    chunk.source_title,
                    chunk.content,
                    chunk.chunk_summary,
                    chunk.content_hash,
                    chunk.status,
                    chunk.created_at,
                    chunk.updated_at,
                ))
                inserted += 1
        return db_path, inserted, deleted, skipped_duplicates
    finally:
        conn.close()


def fetch_chunks_by_source_title(
    source_title: str,
    db_path: Path = DEFAULT_EVENT_DB_PATH,
) -> list[dict[str, str]]:
    conn = _open_event_db(db_path)
    try:
        return _fetchall_dicts(conn, """
            SELECT id, source_title, content, chunk_summary,
                   content_hash, status, created_at, updated_at
            FROM canonical_chunks
            WHERE source_title = ?
            ORDER BY created_at, rowid
        """, (source_title,))
    finally:
        conn.close()


def fetch_active_chunks(
    db_path: Path = DEFAULT_EVENT_DB_PATH,
) -> list[dict[str, str]]:
    conn = _open_event_db(db_path)
    try:
        return _fetchall_dicts(conn, """
            SELECT id, source_title, content, chunk_summary,
                   content_hash, status, created_at, updated_at
            FROM canonical_chunks
            WHERE status = 'active'
            ORDER BY created_at, rowid
        """)
    finally:
        conn.close()


def fetch_chunk_by_id(
    chunk_id: str,
    db_path: Path = DEFAULT_EVENT_DB_PATH,
) -> dict[str, str] | None:
    conn = _open_event_db(db_path)
    try:
        rows = _fetchall_dicts(conn, """
            SELECT id, source_title, content, chunk_summary,
                   content_hash, status, created_at, updated_at
            FROM canonical_chunks
            WHERE id = ?
            LIMIT 1
        """, (chunk_id,))
        return rows[0] if rows else None
    finally:
        conn.close()


def fetch_high_feedback_map(
    db_path: Path = DEFAULT_EVENT_DB_PATH,
) -> dict[str, dict[str, str | int]]:
    conn = _open_event_db(db_path)
    try:
        rows = _fetchall_dicts(conn, """
            SELECT chunk_id, penalized, last_feedback_at
            FROM high_feedback_state
        """)
        return {
            str(row["chunk_id"]): {
                "penalized": int(row["penalized"]),
                "last_feedback_at": str(row["last_feedback_at"]),
            }
            for row in rows
        }
    finally:
        conn.close()


def mark_high_not_helpful(
    chunk_id: str,
    feedback_at: str,
    db_path: Path = DEFAULT_EVENT_DB_PATH,
) -> None:
    conn = _open_event_db(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO high_feedback_state (chunk_id, penalized, last_feedback_at)
                VALUES (?, 1, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    penalized = 1,
                    last_feedback_at = excluded.last_feedback_at
                """,
                (chunk_id, feedback_at),
            )
    finally:
        conn.close()


def enqueue_pending_delete(
    chunk_id: str,
    reported_at: str,
    source_query: str = "",
    feedback_note: str = "",
    db_path: Path = DEFAULT_EVENT_DB_PATH,
) -> None:
    conn = _open_event_db(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO pending_delete_queue (
                    chunk_id, source_query, feedback_note, reported_at, review_status
                ) VALUES (?, ?, ?, ?, 'pending')
                ON CONFLICT(chunk_id) DO UPDATE SET
                    source_query = excluded.source_query,
                    feedback_note = excluded.feedback_note,
                    reported_at = excluded.reported_at,
                    review_status = 'pending'
                """,
                (chunk_id, source_query, feedback_note, reported_at),
            )
    finally:
        conn.close()


def fetch_pending_delete_queue(
    db_path: Path = DEFAULT_EVENT_DB_PATH,
    review_status: str = "pending",
) -> list[dict[str, str]]:
    conn = _open_event_db(db_path)
    try:
        return _fetchall_dicts(conn, """
            SELECT chunk_id, source_query, feedback_note, reported_at, review_status
            FROM pending_delete_queue
            WHERE review_status = ?
            ORDER BY reported_at, rowid
        """, (review_status,))
    finally:
        conn.close()


def update_pending_delete_review_status(
    chunk_id: str,
    review_status: str,
    db_path: Path = DEFAULT_EVENT_DB_PATH,
) -> None:
    conn = _open_event_db(db_path)
    try:
        with conn:
            conn.execute(
                """
                UPDATE pending_delete_queue
                SET review_status = ?
                WHERE chunk_id = ?
                """,
                (review_status, chunk_id),
            )
    finally:
        conn.close()


def delete_chunk_by_id(
    chunk_id: str,
    db_path: Path = DEFAULT_EVENT_DB_PATH,
) -> bool:
    conn = _open_event_db(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM canonical_chunks WHERE id = ?",
                (chunk_id,),
            )
            deleted = conn.changes() > 0
            conn.execute(
                "DELETE FROM high_feedback_state WHERE chunk_id = ?",
                (chunk_id,),
            )
            return deleted
    finally:
        conn.close()


def get_runtime_state(
    state_key: str,
    db_path: Path = DEFAULT_EVENT_DB_PATH,
) -> str | None:
    conn = _open_event_db(db_path)
    try:
        rows = _fetchall_dicts(conn, """
            SELECT state_value
            FROM maintenance_runtime_state
            WHERE state_key = ?
            LIMIT 1
        """, (state_key,))
        if not rows:
            return None
        return str(rows[0]["state_value"])
    finally:
        conn.close()


def set_runtime_state(
    state_key: str,
    state_value: str,
    db_path: Path = DEFAULT_EVENT_DB_PATH,
) -> None:
    conn = _open_event_db(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO maintenance_runtime_state (state_key, state_value)
                VALUES (?, ?)
                ON CONFLICT(state_key) DO UPDATE SET
                    state_value = excluded.state_value
                """,
                (state_key, state_value),
            )
    finally:
        conn.close()


def mark_high_error(
    chunk_id: str,
    feedback_at: str,
    source_query: str = "",
    feedback_note: str = "",
    db_path: Path = DEFAULT_EVENT_DB_PATH,
) -> None:
    mark_high_not_helpful(
        chunk_id=chunk_id,
        feedback_at=feedback_at,
        db_path=db_path,
    )
    enqueue_pending_delete(
        chunk_id=chunk_id,
        reported_at=feedback_at,
        source_query=source_query,
        feedback_note=feedback_note,
        db_path=db_path,
    )
