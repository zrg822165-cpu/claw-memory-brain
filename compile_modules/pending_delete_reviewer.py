"""Periodic reviewer for pending delete queue."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .event_chunker import _build_providers, _get_session, _parse_llm_json, now_timestamp
from .event_store import (
    DEFAULT_EVENT_DB_PATH,
    delete_chunk_by_id,
    fetch_chunk_by_id,
    fetch_pending_delete_queue,
    get_runtime_state,
    set_runtime_state,
    update_pending_delete_review_status,
)
from .indexer import (
    DEFAULT_HIGH_SEARCH_PROJECTION_DIR,
    DEFAULT_HIGH_SEARCH_PROJECTION_FTS_PATH,
    rebuild_high_search_projection_fts_index,
    rebuild_high_search_projection_lance_index,
)


logger = logging.getLogger(__name__)

REVIEW_RUNTIME_KEY = "pending_delete_reviewer:last_run_at"

SYSTEM_PROMPT = """你是记忆库待删审查器。

任务：判断一条被用户标记为“错误”的 chunk 是否应该删除。

规则：
- 只输出 JSON
- 只能回答 keep 或 delete
- 保守判断；不确定就 keep
- 如果 chunk 可能仍有部分价值，也 keep
- stack_top / stack_bottom 当前可能为空；为空时只基于目标 chunk 与用户反馈做保守判断
- 只有当 chunk 整体上明显错误、误导或继续保留只会污染检索时，才 delete

输出格式：
{
  "decision": "keep" | "delete",
  "reason": "一句简短理由"
}"""


def _build_review_query(chunk: dict[str, str]) -> str:
    summary = str(chunk.get("chunk_summary", "")).strip()
    if len(summary) >= 8:
        return summary
    content = str(chunk.get("content", "")).strip()
    if not content:
        return summary
    first_line = content.splitlines()[0].strip()
    return first_line[:120] if len(first_line) > 120 else first_line


def _trim_reference(
    row: dict[str, Any] | None,
    *,
    db_path: Path,
) -> dict[str, str] | None:
    if not row:
        return None
    source_title = str(row.get("source_title", ""))
    if not source_title and row.get("id"):
        ref_chunk = fetch_chunk_by_id(str(row["id"]), db_path=db_path)
        if ref_chunk is not None:
            source_title = str(ref_chunk.get("source_title", ""))
    return {
        "id": str(row.get("id", "")),
        "source_title": source_title,
        "chunk_summary": str(row.get("chunk_summary", "")),
        "content": str(row.get("content", "")),
    }


def _build_review_stack(
    *,
    chunk: dict[str, str],
    config: dict[str, Any],
    db_path: Path,
    lance_dir: Path,
    fts_path: Path,
    timeout: int | None,
    top_k: int = 8,
) -> dict[str, Any]:
    query = _build_review_query(chunk)
    return {
        "query": query,
        "stack_top": None,
        "stack_bottom": None,
        "route": "detached",
        "reason": "review_retrieval_detached",
        "stack_size": 0,
    }


def _call_review_llm(prompt: str, provider: dict[str, Any], timeout: int) -> dict[str, Any]:
    endpoint = f"{provider['base_url'].rstrip('/')}/chat/completions"
    payload = {
        "model": provider["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {provider['api_key']}",
    }
    if provider.get("force_no_proxy") or endpoint.startswith("http://"):
        proxies = {"http": None, "https": None}
    else:
        proxies = {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}

    response = _get_session().post(
        endpoint,
        json=payload,
        headers=headers,
        timeout=timeout,
        proxies=proxies,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return _parse_llm_json(content)


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _review_one(
    *,
    chunk: dict[str, str],
    queue_item: dict[str, str],
    config: dict[str, Any],
    db_path: Path,
    lance_dir: Path,
    fts_path: Path,
    timeout: int | None = None,
) -> dict[str, Any]:
    providers = _build_providers(config)
    if not providers:
        raise RuntimeError("No cloud LLM providers configured for pending delete reviewer")

    stack_context = _build_review_stack(
        chunk=chunk,
        config=config,
        db_path=db_path,
        lance_dir=lance_dir,
        fts_path=fts_path,
        timeout=timeout,
    )
    prompt = f"""chunk_summary:
{chunk.get("chunk_summary", "")}

content:
{chunk.get("content", "")}

source_query:
{queue_item.get("source_query", "")}

feedback_note:
{queue_item.get("feedback_note", "")}

stack_top:
{json.dumps(stack_context.get("stack_top"), ensure_ascii=False)}

stack_bottom:
{json.dumps(stack_context.get("stack_bottom"), ensure_ascii=False)}
"""

    request_timeout = int(timeout or config.get("compile_retry_timeout", 60))
    last_error: Exception | None = None
    for provider in providers:
        try:
            parsed = _call_review_llm(prompt, provider, request_timeout)
            decision = str(parsed.get("decision", "")).strip().lower()
            reason = str(parsed.get("reason", "")).strip()
            if decision in {"keep", "delete"}:
                return {
                    "decision": decision,
                    "reason": reason or "reviewed",
                    "stack_context": stack_context,
                }
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Provider %s failed during pending delete review for %s: %s",
                provider["label"],
                chunk.get("id", ""),
                exc,
            )
    if last_error is not None:
        raise RuntimeError(f"Pending delete review failed: {last_error}") from last_error
    raise RuntimeError("Pending delete review produced no valid decision")


def run_pending_delete_review_if_due(
    *,
    config: dict[str, Any],
    db_path: Path = DEFAULT_EVENT_DB_PATH,
    lance_dir: Path = DEFAULT_HIGH_SEARCH_PROJECTION_DIR,
    fts_path: Path = DEFAULT_HIGH_SEARCH_PROJECTION_FTS_PATH,
    timeout: int | None = None,
    interval_minutes: int = 30,
) -> dict[str, Any]:
    pending = fetch_pending_delete_queue(db_path=db_path)
    if not pending:
        return {
            "ran": False,
            "reason": "empty_queue",
            "pending_count": 0,
            "deleted_count": 0,
            "kept_count": 0,
        }

    last_run = _parse_timestamp(get_runtime_state(REVIEW_RUNTIME_KEY, db_path=db_path) or "")
    now = datetime.now(timezone.utc)
    if last_run is not None and now - last_run < timedelta(minutes=interval_minutes):
        return {
            "ran": False,
            "reason": "interval_not_due",
            "pending_count": len(pending),
            "deleted_count": 0,
            "kept_count": 0,
            "last_run_at": last_run.isoformat().replace("+00:00", "Z"),
        }

    deleted_count = 0
    kept_count = 0
    reviewed: list[dict[str, str]] = []
    for queue_item in pending:
        chunk_id = str(queue_item["chunk_id"])
        chunk = fetch_chunk_by_id(chunk_id, db_path=db_path)
        if chunk is None:
            update_pending_delete_review_status(
                chunk_id,
                "missing",
                db_path=db_path,
            )
            reviewed.append({
                "chunk_id": chunk_id,
                "decision": "missing",
                "reason": "chunk already absent",
            })
            continue

        decision = _review_one(
            chunk=chunk,
            queue_item=queue_item,
            config=config,
            db_path=db_path,
            lance_dir=lance_dir,
            fts_path=fts_path,
            timeout=timeout,
        )
        if decision["decision"] == "delete":
            delete_chunk_by_id(chunk_id, db_path=db_path)
            update_pending_delete_review_status(
                chunk_id,
                "deleted",
                db_path=db_path,
            )
            deleted_count += 1
        else:
            update_pending_delete_review_status(
                chunk_id,
                "kept",
                db_path=db_path,
            )
            kept_count += 1
        reviewed.append({
            "chunk_id": chunk_id,
            "decision": decision["decision"],
            "reason": decision["reason"],
            "stack_context": decision.get("stack_context", {}),
        })

    set_runtime_state(
        REVIEW_RUNTIME_KEY,
        now_timestamp(),
        db_path=db_path,
    )

    if deleted_count:
            rebuild_high_search_projection_lance_index(
                db_path=db_path,
                lance_dir=lance_dir,
                config=config,
                timeout=timeout,
            )
            rebuild_high_search_projection_fts_index(
                db_path=db_path,
                fts_path=fts_path,
            )

    return {
        "ran": True,
        "reason": "reviewed",
        "pending_count": len(pending),
        "deleted_count": deleted_count,
        "kept_count": kept_count,
        "reviewed": reviewed,
    }
