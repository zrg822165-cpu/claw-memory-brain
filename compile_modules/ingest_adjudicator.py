"""LLM-assisted pre-ingest adjudicator."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .event_chunker import (
    EventChunk,
    _build_providers,
    _get_session,
    _parse_llm_json,
)
from .indexer import (
    DEFAULT_HIGH_SEARCH_PROJECTION_DIR,
    DEFAULT_HIGH_SEARCH_PROJECTION_FTS_PATH,
)


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是记忆库入库裁决器。

任务：判断新 chunk 相对于记忆库中最接近的已有 chunk 应该怎么处理。

只输出 JSON。

可选动作：
- insert_new
- reject_new
- replace_old
- keep_both

规则：
- 如果新 chunk 只是旧 chunk 的不同表述、没有新增信息，reject_new
- 如果新 chunk 更新了既定事实、既定配置或当前状态，旧 chunk 不应继续保留，replace_old
- 如果新 chunk 和旧 chunk 都有独立价值，应 keep_both
- 如果旧 chunk 不足以构成有效参照，应 insert_new
- created_at 仅作为参考：较新的记录更可能反映当前既定事实，但不能只靠时间判断

输出格式：
{
  "decisions": [
    {
      "new_chunk_index": 0,
      "decision": "insert_new | reject_new | replace_old | keep_both",
      "reason": "一句简短理由"
    }
  ]
}"""


def _call_adjudicator_llm(prompt: str, provider: dict[str, Any], timeout: int) -> dict[str, Any]:
    endpoint = f"{provider['base_url'].rstrip('/')}/chat/completions"
    payload = {
        "model": provider["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1200,
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


def _normalize_decisions(raw: Any, count: int) -> dict[int, dict[str, str]]:
    if not isinstance(raw, list):
        return {}
    allowed = {"insert_new", "reject_new", "replace_old", "keep_both"}
    normalized: dict[int, dict[str, str]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("new_chunk_index"))
        except Exception:
            continue
        if idx < 0 or idx >= count:
            continue
        decision = str(item.get("decision", "")).strip()
        if decision not in allowed:
            continue
        normalized[idx] = {
            "decision": decision,
            "reason": str(item.get("reason", "")).strip(),
        }
    return normalized


def _query_top1_candidates(
    chunks: list[EventChunk],
    *,
    config: dict[str, Any],
    lance_dir: Path = DEFAULT_HIGH_SEARCH_PROJECTION_DIR,
    fts_path: Path = DEFAULT_HIGH_SEARCH_PROJECTION_FTS_PATH,
    timeout: int | None = None,
) -> list[dict[str, Any] | None]:
    del config, lance_dir, fts_path, timeout
    return [None for _ in chunks]


def adjudicate_chunks_against_top1(
    chunks: list[EventChunk],
    *,
    config: dict[str, Any],
    lance_dir: Path = DEFAULT_HIGH_SEARCH_PROJECTION_DIR,
    fts_path: Path = DEFAULT_HIGH_SEARCH_PROJECTION_FTS_PATH,
    timeout: int | None = None,
) -> dict[str, Any]:
    if not chunks:
        return {
            "insert_chunks": [],
            "delete_chunk_ids": [],
            "adjudications": [],
        }

    top1_candidates = _query_top1_candidates(
        chunks,
        config=config,
        lance_dir=lance_dir,
        fts_path=fts_path,
        timeout=timeout,
    )

    prompt_payload = {
        "items": [
            {
                "new_chunk_index": idx,
                "new_chunk": {
                    "content": chunk.content,
                    "chunk_summary": chunk.chunk_summary,
                    "created_at": chunk.created_at,
                },
                "top1_chunk": (
                    {
                        "id": top1["id"],
                        "content": top1.get("content", ""),
                        "chunk_summary": top1.get("chunk_summary", ""),
                        "created_at": top1.get("created_at", ""),
                    }
                    if top1 is not None else None
                ),
            }
            for idx, (chunk, top1) in enumerate(zip(chunks, top1_candidates, strict=True))
        ]
    }
    prompt = json.dumps(prompt_payload, ensure_ascii=False, indent=2)

    providers = _build_providers(config)
    if not providers:
        raise RuntimeError("No cloud LLM providers configured for ingest adjudicator")
    request_timeout = int(timeout or config.get("compile_retry_timeout", 60))
    last_error: Exception | None = None
    parsed: dict[str, Any] | None = None
    for provider in providers:
        try:
            parsed = _call_adjudicator_llm(prompt, provider, request_timeout)
            break
        except Exception as exc:
            last_error = exc
            logger.warning("Provider %s failed during ingest adjudication: %s", provider["label"], exc)
    if parsed is None:
        if last_error is not None:
            raise RuntimeError(f"All providers failed during ingest adjudication: {last_error}") from last_error
        raise RuntimeError("All providers failed during ingest adjudication")

    decisions = _normalize_decisions(parsed.get("decisions", []), len(chunks))
    insert_chunks: list[EventChunk] = []
    delete_ids: list[str] = []
    adjudications: list[dict[str, Any]] = []
    seen_delete_ids: set[str] = set()

    for idx, chunk in enumerate(chunks):
        top1 = top1_candidates[idx]
        default_decision = "insert_new" if top1 is None else "keep_both"
        resolved = decisions.get(idx, {"decision": default_decision, "reason": "default_fallback"})
        decision = resolved["decision"]
        reason = resolved["reason"]
        top1_id = str(top1["id"]) if top1 is not None else ""

        if decision == "reject_new":
            pass
        elif decision == "replace_old":
            insert_chunks.append(chunk)
            if top1_id and top1_id not in seen_delete_ids:
                seen_delete_ids.add(top1_id)
                delete_ids.append(top1_id)
        else:
            insert_chunks.append(chunk)

        adjudications.append({
            "new_chunk_id": chunk.id,
            "new_chunk_index": idx,
            "decision": decision,
            "reason": reason,
            "top1_chunk_id": top1_id,
            "top1_summary": "" if top1 is None else str(top1.get("chunk_summary", "")),
            "top1_created_at": "" if top1 is None else str(top1.get("created_at", "")),
        })

    return {
        "insert_chunks": insert_chunks,
        "delete_chunk_ids": delete_ids,
        "adjudications": adjudications,
    }
