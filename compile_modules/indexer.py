"""High search projection indexer derived from canonical event chunks."""
from __future__ import annotations

import logging
import sqlite3
import re
from pathlib import Path
from typing import Any

import lancedb
import requests

from .event_store import DEFAULT_EVENT_DB_PATH, fetch_active_chunks
from .runtime_paths import DATA_DIR


logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = DATA_DIR
DEFAULT_HIGH_SEARCH_PROJECTION_DIR = DEFAULT_DATA_DIR / "high_search_projection" / "lance"
DEFAULT_HIGH_SEARCH_PROJECTION_TABLE_NAME = "canonical_high_search_projection"
DEFAULT_HIGH_SEARCH_PROJECTION_FTS_PATH = DEFAULT_DATA_DIR / "high_search_projection" / "fts.sqlite"
DEFAULT_HIGH_SEARCH_PROJECTION_FTS_TABLE = "canonical_high_search_projection_fts"
DEFAULT_HIGH_SEARCH_PROJECTION_VECTOR_WEIGHT = 0.7
DEFAULT_HIGH_SEARCH_PROJECTION_FTS_WEIGHT = 0.3
LEGACY_HIGH_LANCE_DIR = DEFAULT_DATA_DIR / "high_lance"
LEGACY_HIGH_FTS_PATH = DEFAULT_DATA_DIR / "high_fts.sqlite"
LEGACY_HIGH_TABLE_NAME = "canonical_high_chunks"
LEGACY_HIGH_FTS_TABLE = "canonical_high_fts"
_MULTIMODAL_EMBEDDING_MODELS = {
    "qwen3-vl-embedding",
    "qwen2.5-vl-embedding",
    "tongyi-embedding-vision-plus-2026-03-06",
    "tongyi-embedding-vision-flash-2026-03-06",
    "tongyi-embedding-vision-plus",
    "tongyi-embedding-vision-flash",
    "multimodal-embedding-v1",
}
_SAFE_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]+")
_SAFE_CJK_BLOCK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_LEADING_CJK_NOISE = ("的", "怎么", "如何", "请问", "目前", "现在", "这个", "那个")
_TRAILING_CJK_NOISE = ("是什么", "是啥", "干什么", "吗", "呢", "呀", "吧", "么")


def _embedding_model_name(config: dict[str, Any]) -> str:
    return str(config.get("embedding_model_name", "")).strip()


def _is_multimodal_embedding(config: dict[str, Any]) -> bool:
    mode = str(config.get("embedding_api_mode", "")).strip().lower()
    if mode:
        return mode == "multimodal"
    return _embedding_model_name(config).lower() in _MULTIMODAL_EMBEDDING_MODELS


def _embedding_endpoint(config: dict[str, Any]) -> str:
    base_url = str(config.get("embedding_base_url", "")).rstrip("/")
    if not base_url:
        raise RuntimeError("Missing embedding_base_url in config")
    if _is_multimodal_embedding(config):
        if "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding" in base_url:
            return base_url
        if base_url.endswith("/compatible-mode/v1"):
            base_url = base_url[: -len("/compatible-mode/v1")]
        return f"{base_url}/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
    return f"{base_url}/embeddings"


def _build_embedding_headers(config: dict[str, Any]) -> dict[str, str]:
    api_key = str(config.get("embedding_api_key", "")).strip()
    if not api_key:
        raise RuntimeError("Missing embedding_api_key in config")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def _should_bypass_proxy(config: dict[str, Any], endpoint: str) -> bool:
    return bool(config.get("embedding_force_no_proxy", False)) or endpoint.startswith("http://")


def _embedding_proxies(config: dict[str, Any], endpoint: str) -> dict[str, str | None]:
    if _should_bypass_proxy(config, endpoint):
        return {"http": None, "https": None}
    return {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}


def _embed_texts(
    texts: list[str],
    config: dict[str, Any],
    timeout: int | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    endpoint = _embedding_endpoint(config)
    model_name = _embedding_model_name(config)
    payload: dict[str, Any]
    dimensions = config.get("embedding_dimensions")
    if _is_multimodal_embedding(config):
        payload = {
            "model": model_name,
            "input": {
                "contents": [{"text": text} for text in texts],
            },
        }
        if dimensions:
            payload["parameters"] = {"dimension": int(dimensions)}
    else:
        payload = {
            "model": model_name,
            "input": texts,
        }
        if dimensions:
            payload["dimensions"] = int(dimensions)

    response = requests.post(
        endpoint,
        json=payload,
        headers=_build_embedding_headers(config),
        timeout=int(timeout or 45),
        proxies=_embedding_proxies(config, endpoint),
    )
    if response.status_code >= 400:
        if len(texts) > 1 and response.status_code == 400:
            midpoint = max(1, len(texts) // 2)
            left = _embed_texts(texts[:midpoint], config=config, timeout=timeout)
            right = _embed_texts(texts[midpoint:], config=config, timeout=timeout)
            return left + right
        raise RuntimeError(
            f"Embedding request failed ({response.status_code}): {response.text[:500]}"
        )
    body = response.json()
    if _is_multimodal_embedding(config):
        data = body.get("output", {}).get("embeddings", [])
    else:
        data = body.get("data", [])
    if not isinstance(data, list) or len(data) != len(texts):
        raise RuntimeError("Embedding response size does not match input size")
    return [list(map(float, item["embedding"])) for item in data]


def _resolve_high_search_projection_dir(lance_dir: Path) -> Path:
    resolved = lance_dir.resolve()
    if resolved.exists():
        return resolved
    default_resolved = DEFAULT_HIGH_SEARCH_PROJECTION_DIR.resolve()
    legacy_resolved = LEGACY_HIGH_LANCE_DIR.resolve()
    if resolved == default_resolved and legacy_resolved.exists():
        return legacy_resolved
    return resolved


def _resolve_high_search_projection_fts_path(fts_path: Path) -> Path:
    resolved = fts_path.resolve()
    if resolved.exists():
        return resolved
    default_resolved = DEFAULT_HIGH_SEARCH_PROJECTION_FTS_PATH.resolve()
    legacy_resolved = LEGACY_HIGH_FTS_PATH.resolve()
    if resolved == default_resolved and legacy_resolved.exists():
        return legacy_resolved
    return resolved


def build_high_search_projection_text(chunk_summary: str, content: str) -> str:
    del chunk_summary
    return str(content or "").strip()


def _open_high_search_projection_fts_db(
    fts_path: Path = DEFAULT_HIGH_SEARCH_PROJECTION_FTS_PATH,
) -> sqlite3.Connection:
    fts_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(fts_path))
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {DEFAULT_HIGH_SEARCH_PROJECTION_FTS_TABLE}
        USING fts5(
            id UNINDEXED,
            chunk_summary UNINDEXED,
            content,
            created_at UNINDEXED,
            tokenize='trigram'
        );
    """)
    conn.commit()
    return conn


def _build_safe_fts_match_query(query: str) -> str:
    stripped = query.strip()
    if not stripped:
        return ""

    parts: list[str] = []
    seen: set[str] = set()
    ascii_tokens = [token.strip() for token in _SAFE_ASCII_TOKEN_RE.findall(stripped) if token.strip()]
    if len(ascii_tokens) >= 2:
        combined = " ".join(ascii_tokens)
        escaped = combined.replace('"', '""')
        if escaped not in seen:
            seen.add(escaped)
            parts.append(f'"{escaped}"')
    for token in ascii_tokens:
        token = token.strip()
        if len(token) < 2:
            continue
        escaped = token.replace('"', '""')
        if escaped and escaped not in seen:
            seen.add(escaped)
            parts.append(f'"{escaped}"')
    for block in _SAFE_CJK_BLOCK_RE.findall(stripped):
        cleaned = block
        for prefix in _LEADING_CJK_NOISE:
            if cleaned.startswith(prefix) and len(cleaned) > len(prefix):
                cleaned = cleaned[len(prefix):]
                break
        for suffix in _TRAILING_CJK_NOISE:
            if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
                cleaned = cleaned[: -len(suffix)]
                break
        cleaned = cleaned.strip()
        if len(cleaned) >= 2:
            escaped = cleaned.replace('"', '""')
            if escaped not in seen:
                seen.add(escaped)
                parts.append(f'"{escaped}"')
        for size in (4, 3):
            if len(cleaned) < size:
                continue
            for idx in range(len(cleaned) - size + 1):
                gram = cleaned[idx: idx + size]
                escaped = gram.replace('"', '""')
                if escaped not in seen:
                    seen.add(escaped)
                    parts.append(f'"{escaped}"')

    if parts:
        return " OR ".join(parts)
    escaped_query = stripped.replace('"', '""')
    return f'"{escaped_query}"'


def rebuild_high_search_projection_lance_index(
    db_path: Path = DEFAULT_EVENT_DB_PATH,
    lance_dir: Path = DEFAULT_HIGH_SEARCH_PROJECTION_DIR,
    config: dict[str, Any] | None = None,
    timeout: int | None = None,
    batch_size: int = 16,
) -> tuple[Path, int]:
    if config is None:
        raise RuntimeError("Config is required to build the High search projection LanceDB index")

    rows = fetch_active_chunks(db_path=db_path)
    lance_dir = lance_dir.resolve()
    lance_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(lance_dir))

    existing_tables = set(db.list_tables().tables)
    if DEFAULT_HIGH_SEARCH_PROJECTION_TABLE_NAME in existing_tables:
        db.drop_table(DEFAULT_HIGH_SEARCH_PROJECTION_TABLE_NAME)

    if not rows:
        db.create_table(DEFAULT_HIGH_SEARCH_PROJECTION_TABLE_NAME, data=[], mode="overwrite")
        return lance_dir, 0

    index_rows: list[dict[str, Any]] = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset: offset + batch_size]
        texts = [build_high_search_projection_text(row["chunk_summary"], row["content"]) for row in batch]
        vectors = _embed_texts(texts, config=config, timeout=timeout)
        for row, vector, text in zip(batch, vectors, texts, strict=True):
            index_rows.append({
                "id": row["id"],
                "chunk_summary": row["chunk_summary"],
                "content": row["content"],
                "created_at": row["created_at"],
                "search_text": text,
                "vector": vector,
            })

    db.create_table(DEFAULT_HIGH_SEARCH_PROJECTION_TABLE_NAME, data=index_rows, mode="overwrite")
    logger.info("High search projection LanceDB index rebuilt with %d row(s)", len(index_rows))
    return lance_dir, len(index_rows)


def rebuild_high_search_projection_fts_index(
    db_path: Path = DEFAULT_EVENT_DB_PATH,
    fts_path: Path = DEFAULT_HIGH_SEARCH_PROJECTION_FTS_PATH,
) -> tuple[Path, int]:
    rows = fetch_active_chunks(db_path=db_path)
    if fts_path.exists():
        fts_path.unlink()
    conn = _open_high_search_projection_fts_db(fts_path)
    try:
        if rows:
            conn.executemany(
                f"""
                INSERT INTO {DEFAULT_HIGH_SEARCH_PROJECTION_FTS_TABLE} (
                    id, chunk_summary, content, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        row["id"],
                        row["chunk_summary"],
                        row["content"],
                        row["created_at"],
                    )
                    for row in rows
                ],
            )
        conn.commit()
        logger.info("High search projection FTS5 index rebuilt with %d row(s)", len(rows))
        return fts_path, len(rows)
    finally:
        conn.close()


def search_high_projection_lance(
    query: str,
    config: dict[str, Any],
    lance_dir: Path = DEFAULT_HIGH_SEARCH_PROJECTION_DIR,
    top_k: int = 5,
    timeout: int | None = None,
) -> list[dict[str, Any]]:
    if not query.strip():
        return []

    lance_dir = _resolve_high_search_projection_dir(lance_dir)
    db = lancedb.connect(str(lance_dir))

    query_vector = _embed_texts([query], config=config, timeout=timeout)[0]
    try:
        table = db.open_table(DEFAULT_HIGH_SEARCH_PROJECTION_TABLE_NAME)
    except Exception as exc:
        try:
            table = db.open_table(LEGACY_HIGH_TABLE_NAME)
        except Exception:
            raise RuntimeError(f"High search projection LanceDB table not found in {lance_dir}") from exc
    return (
        table.search(query_vector, vector_column_name="vector")
        .select([
            "id",
            "chunk_summary",
            "content",
            "created_at",
        ])
        .limit(top_k)
        .to_list()
    )


def search_high_projection_fts(
    query: str,
    fts_path: Path = DEFAULT_HIGH_SEARCH_PROJECTION_FTS_PATH,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if not query.strip():
        return []

    fts_path = _resolve_high_search_projection_fts_path(fts_path)
    conn = sqlite3.connect(str(fts_path))
    conn.row_factory = sqlite3.Row
    try:
        table_names = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if DEFAULT_HIGH_SEARCH_PROJECTION_FTS_TABLE in table_names:
            fts_table = DEFAULT_HIGH_SEARCH_PROJECTION_FTS_TABLE
        elif LEGACY_HIGH_FTS_TABLE in table_names:
            fts_table = LEGACY_HIGH_FTS_TABLE
        else:
            return []
        match_query = _build_safe_fts_match_query(query)
        if not match_query:
            return []
        rows = conn.execute(
            f"""
            SELECT
                id,
                chunk_summary,
                content,
                created_at,
                bm25({fts_table}, 1.0) AS _bm25
            FROM {fts_table}
            WHERE {fts_table} MATCH ?
            ORDER BY _bm25
            LIMIT ?
            """,
            (match_query, top_k),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def search_high_projection_hybrid(
    query: str,
    config: dict[str, Any],
    lance_dir: Path = DEFAULT_HIGH_SEARCH_PROJECTION_DIR,
    fts_path: Path = DEFAULT_HIGH_SEARCH_PROJECTION_FTS_PATH,
    top_k: int = 5,
    timeout: int | None = None,
    fuse_k: int = 60,
    vector_weight: float = DEFAULT_HIGH_SEARCH_PROJECTION_VECTOR_WEIGHT,
    fts_weight: float = DEFAULT_HIGH_SEARCH_PROJECTION_FTS_WEIGHT,
) -> list[dict[str, Any]]:
    vec_rows = search_high_projection_lance(
        query=query,
        config=config,
        lance_dir=lance_dir,
        top_k=max(top_k, 8),
        timeout=timeout,
    )
    fts_rows = search_high_projection_fts(
        query=query,
        fts_path=fts_path,
        top_k=max(top_k, 8),
    )

    merged: dict[str, dict[str, Any]] = {}
    for rank, row in enumerate(vec_rows, start=1):
        item = merged.setdefault(row["id"], {})
        item.update(row)
        item["_vec_rank"] = rank
        item["_vec_score"] = float(vector_weight) / (fuse_k + rank)
    for rank, row in enumerate(fts_rows, start=1):
        item = merged.setdefault(row["id"], {})
        item.update({k: v for k, v in row.items() if k not in item or not item.get(k)})
        item["_fts_rank"] = rank
        item["_fts_score"] = float(fts_weight) / (fuse_k + rank)

    results: list[dict[str, Any]] = []
    for item in merged.values():
        item["_hybrid_score"] = float(item.get("_vec_score", 0.0)) + float(item.get("_fts_score", 0.0))
        results.append(item)

    results.sort(
        key=lambda row: (
            float(row.get("_hybrid_score", 0.0)),
            float(row.get("_fts_score", 0.0)),
            float(row.get("_vec_score", 0.0)),
        ),
        reverse=True,
    )
    return results[:top_k]


# Compatibility aliases for callers not yet migrated to high_search_projection naming.
DEFAULT_HIGH_LANCE_DIR = DEFAULT_HIGH_SEARCH_PROJECTION_DIR
DEFAULT_HIGH_TABLE_NAME = DEFAULT_HIGH_SEARCH_PROJECTION_TABLE_NAME
DEFAULT_HIGH_FTS_PATH = DEFAULT_HIGH_SEARCH_PROJECTION_FTS_PATH
DEFAULT_HIGH_FTS_TABLE = DEFAULT_HIGH_SEARCH_PROJECTION_FTS_TABLE
DEFAULT_HIGH_VECTOR_WEIGHT = DEFAULT_HIGH_SEARCH_PROJECTION_VECTOR_WEIGHT
DEFAULT_HIGH_FTS_WEIGHT = DEFAULT_HIGH_SEARCH_PROJECTION_FTS_WEIGHT
build_high_search_text = build_high_search_projection_text
_open_high_fts_db = _open_high_search_projection_fts_db
rebuild_high_lance_index = rebuild_high_search_projection_lance_index
rebuild_high_fts_index = rebuild_high_search_projection_fts_index
search_high_lance = search_high_projection_lance
search_high_fts = search_high_projection_fts
search_high_hybrid = search_high_projection_hybrid
