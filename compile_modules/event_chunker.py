"""Event-chunk compiler.

将 memory markdown 交给云端 LLM，切分成 event chunks。
这一层只生成：
- content
- chunk_summary

source_title 与管理字段由系统在落库前补充。
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests

from .runtime_paths import PROJECT_ROOT

logger = logging.getLogger(__name__)
_thread_local = threading.local()


SYSTEM_PROMPT = """你是记忆库事件切分器。

把一份 memory markdown 切分成一个或多个事件，只输出 JSON。

每个事件只输出：
- content
- chunk_summary

要求：
- 按语义切分事件
- 保留关键细节，不发明事实
- 输出顺序与原文一致
- chunk_summary 用一句话概括该事件的稳定经验、配置、用途或结论
- 不要输出其他字段

输出格式：
{
  "events": [
    {
      "content": "...",
      "chunk_summary": "..."
    }
  ]
}"""

USER_PROMPT_TEMPLATE = """以下是待切分的 memory markdown 原文：

{markdown_text}"""


@dataclass
class EventChunk:
    id: str
    source_title: str
    content: str
    chunk_summary: str
    content_hash: str
    status: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "source_title": self.source_title,
            "content": self.content,
            "chunk_summary": self.chunk_summary,
            "content_hash": self.content_hash,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def now_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def normalized_content_hash(content: str) -> str:
    normalized = "\n".join(line.rstrip() for line in content.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_config(config_path: Path) -> dict[str, Any]:
    resolved = config_path.expanduser()
    if not resolved.is_absolute():
        project_candidate = PROJECT_ROOT / resolved
        if project_candidate.exists():
            resolved = project_candidate
    config = json.loads(resolved.read_text(encoding="utf-8"))
    local_override = resolved.with_name("config.local.json")
    if local_override.exists() and local_override != resolved:
        local_config = json.loads(local_override.read_text(encoding="utf-8"))
        if not isinstance(local_config, dict):
            raise RuntimeError(f"Local config override must be a JSON object: {local_override}")
        config.update(local_config)
    return config


def _build_providers(config: dict[str, Any]) -> list[dict[str, Any]]:
    providers = [
        {
            "label": "primary",
            "base_url": config.get("cloud_llm_base_url", ""),
            "api_key": config.get("cloud_llm_api_key", ""),
            "model": config.get("cloud_llm_model", ""),
            "force_no_proxy": bool(config.get("cloud_llm_force_no_proxy", False)),
        },
        {
            "label": "fallback",
            "base_url": config.get("cloud_llm_fallback_base_url", ""),
            "api_key": config.get("cloud_llm_fallback_api_key", ""),
            "model": config.get("cloud_llm_fallback_model", ""),
            "force_no_proxy": bool(config.get("cloud_llm_fallback_force_no_proxy", False)),
        },
        {
            "label": "alt",
            "base_url": config.get("cloud_llm_alt_base_url", ""),
            "api_key": config.get("cloud_llm_alt_api_key", ""),
            "model": config.get("cloud_llm_alt_model", ""),
            "force_no_proxy": bool(config.get("cloud_llm_alt_force_no_proxy", False)),
        },
    ]
    return [p for p in providers if p["base_url"] and p["api_key"] and p["model"]]


def _strip_json_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return value


def _coerce_json_root(parsed: Any) -> dict[str, Any]:
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return {"events": parsed}
    raise ValueError("LLM response JSON root is not a supported object/list shape")


def _parse_llm_json(content: str) -> dict[str, Any]:
    value = _strip_json_fence(content)
    try:
        return _coerce_json_root(json.loads(value))
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    start_positions = [pos for pos in (value.find("{"), value.find("[")) if pos >= 0]
    if not start_positions:
        raise ValueError("No JSON object/array found in LLM response")
    parsed, _end = decoder.raw_decode(value, idx=min(start_positions))
    return _coerce_json_root(parsed)


def _get_session() -> requests.Session:
    session: requests.Session | None = getattr(_thread_local, "http_session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=8,
            pool_maxsize=8,
            max_retries=0,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _thread_local.http_session = session
    return session


def _call_llm(markdown_text: str, provider: dict[str, Any], timeout: int) -> dict[str, Any]:
    endpoint = f"{provider['base_url'].rstrip('/')}/chat/completions"
    payload = {
        "model": provider["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(markdown_text=markdown_text)},
        ],
        "temperature": 0.0,
        "max_tokens": 4000,
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


def _normalize_events(raw_events: Any) -> list[dict[str, str]]:
    if not isinstance(raw_events, list):
        return []

    normalized: list[dict[str, str]] = []
    for raw in raw_events:
        if not isinstance(raw, dict):
            continue
        content = str(raw.get("content", "")).strip()
        chunk_summary = str(raw.get("chunk_summary", "")).strip()
        if not content or not chunk_summary:
            continue
        normalized.append({
            "content": content,
            "chunk_summary": chunk_summary,
        })
    return normalized


def _compute_segment_timeout(markdown_text: str) -> int:
    length = len(markdown_text)
    if length <= 1500:
        return 30
    if length <= 3000:
        return 45
    if length <= 5000:
        return 75
    return 90


def _order_providers_for_segmentation(
    providers: list[dict[str, Any]],
    markdown_text: str,
) -> list[dict[str, Any]]:
    if len(markdown_text) <= 3000:
        return providers
    alt_first: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []
    for provider in providers:
        if str(provider.get("label", "")) == "alt":
            alt_first.append(provider)
        else:
            others.append(provider)
    return alt_first + others if alt_first else providers


def segment_memory_markdown(
    markdown_text: str,
    config: dict[str, Any],
    timeout: int | None = None,
) -> list[dict[str, str]]:
    providers = _build_providers(config)
    if not providers:
        raise RuntimeError("No cloud LLM providers configured for event chunk segmentation")
    providers = _order_providers_for_segmentation(providers, markdown_text)

    request_timeout = int(timeout or _compute_segment_timeout(markdown_text))
    last_error: Exception | None = None

    for provider in providers:
        try:
            logger.info(
                "Segmenting memory markdown with %s (%s), timeout=%ss",
                provider["label"],
                provider["model"],
                request_timeout,
            )
            parsed = _call_llm(markdown_text, provider, request_timeout)
            events = _normalize_events(parsed.get("events", []))
            if events:
                return events
            logger.warning("Provider %s returned no valid events", provider["label"])
        except Exception as exc:
            last_error = exc
            logger.warning("Provider %s failed during event segmentation: %s", provider["label"], exc)

    if last_error is not None:
        raise RuntimeError(f"All providers failed during event segmentation: {last_error}") from last_error
    raise RuntimeError("All providers returned no valid events")


def build_event_chunks(
    markdown_text: str,
    source_title: str,
    created_at: str,
    config: dict[str, Any],
    timeout: int | None = None,
) -> list[EventChunk]:
    events = segment_memory_markdown(markdown_text, config=config, timeout=timeout)
    return [
        EventChunk(
            id=str(uuid4()),
            source_title=source_title,
            content=event["content"],
            chunk_summary=event["chunk_summary"],
            content_hash=normalized_content_hash(event["content"]),
            status="active",
            created_at=created_at,
            updated_at=created_at,
        )
        for event in events
    ]
