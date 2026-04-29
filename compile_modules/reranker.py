"""Cloud reranker clients for broad-recall relevance ordering."""
from __future__ import annotations

import os
import threading
from typing import Any

import requests


DEFAULT_RERANK_BASE_URL = "https://dashscope.aliyuncs.com"
DEFAULT_RERANK_MODEL_NAME = "qwen3-vl-rerank"
DEFAULT_RERANK_TIMEOUT_SECONDS = 20
DEFAULT_RERANK_INSTRUCT = (
    "Rank memory chunks by whether a single chunk can directly answer the full user query. "
    "Prefer chunks whose main content is the direct answer. "
    "Demote same-topic background, summaries, references, or chunks that only mention the subject in passing."
)
DEFAULT_RERANK_PROVIDER = "dashscope"
DEFAULT_RERANK_DOCUMENT_MODE = "content_only"

_thread_local = threading.local()


def _get_session() -> requests.Session:
    session: requests.Session | None = getattr(_thread_local, "http_session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4,
            pool_maxsize=4,
            max_retries=0,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _thread_local.http_session = session
    return session


def _resolve_api_key(config: dict[str, Any]) -> str:
    env_name = str(config.get("rerank_api_key_env", "")).strip()
    if env_name:
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            return env_value
    direct = str(config.get("rerank_api_key", "")).strip()
    if direct:
        return direct
    # Reuse the existing DashScope embedding key by default.
    return str(config.get("embedding_api_key", "")).strip()


def _resolve_endpoint(config: dict[str, Any]) -> str:
    provider = str(config.get("rerank_provider", DEFAULT_RERANK_PROVIDER)).strip().lower()
    base_url = str(config.get("rerank_base_url", DEFAULT_RERANK_BASE_URL)).rstrip("/")
    if not base_url:
        base_url = DEFAULT_RERANK_BASE_URL
    if provider == "siliconflow":
        if base_url.endswith("/v1/rerank"):
            return base_url
        return f"{base_url}/v1/rerank"
    if base_url.endswith("/api/v1/services/rerank/text-rerank/text-rerank"):
        return base_url
    return f"{base_url}/api/v1/services/rerank/text-rerank/text-rerank"


def _should_bypass_proxy(config: dict[str, Any], endpoint: str) -> bool:
    return bool(config.get("rerank_force_no_proxy", False)) or endpoint.startswith("http://")


def _proxies_for_request(config: dict[str, Any], endpoint: str) -> dict[str, str | None]:
    if _should_bypass_proxy(config, endpoint):
        return {"http": None, "https": None}
    return {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}


def _resolve_document_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("rerank_document_mode", DEFAULT_RERANK_DOCUMENT_MODE)).strip().lower()
    if mode in {"summary_content", "content_only", "content_summary"}:
        return mode
    return DEFAULT_RERANK_DOCUMENT_MODE


def build_rerank_document(row: dict[str, Any], *, config: dict[str, Any]) -> str:
    summary = str(row.get("chunk_summary", "")).strip()
    content = str(row.get("content", "")).strip()
    mode = _resolve_document_mode(config)
    parts = []
    if mode == "content_only":
        # Keep a non-empty fallback if a row somehow lacks canonical content.
        return content or summary
    if mode == "content_summary":
        if content:
            parts.append("[content]")
            parts.append(content)
        if summary:
            parts.append("")
            parts.append("[summary]")
            parts.append(summary)
        return "\n".join(parts).strip()
    if summary:
        parts.append("[summary]")
        parts.append(summary)
    if content:
        parts.append("")
        parts.append("[content]")
        parts.append(content)
    return "\n".join(parts).strip()


def _request_rerank(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    instruction: str,
    top_k: int | None,
    timeout: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provider = str(config.get("rerank_provider", DEFAULT_RERANK_PROVIDER)).strip().lower() or DEFAULT_RERANK_PROVIDER
    endpoint = _resolve_endpoint(config)
    model_name = str(config.get("rerank_model_name", DEFAULT_RERANK_MODEL_NAME)).strip() or DEFAULT_RERANK_MODEL_NAME
    documents = [build_rerank_document(row, config=config) for row in candidates]
    if provider == "siliconflow":
        payload = {
            "model": model_name,
            "query": query,
            "documents": documents,
            "return_documents": False,
        }
        if instruction:
            payload["instruction"] = instruction
        if top_k is not None and top_k > 0:
            payload["top_n"] = int(top_k)
    else:
        payload = {
            "model": model_name,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "instruct": instruction,
            },
        }
        if top_k is not None and top_k > 0:
            payload["parameters"]["top_n"] = int(top_k)

    response = _get_session().post(
        endpoint,
        json=payload,
        headers={
            "Authorization": f"Bearer {_resolve_api_key(config)}",
            "Content-Type": "application/json",
        },
        timeout=int(timeout or config.get("rerank_timeout_seconds", DEFAULT_RERANK_TIMEOUT_SECONDS)),
        proxies=_proxies_for_request(config, endpoint),
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Rerank request failed ({response.status_code}): {response.text[:500]}")

    body = response.json()
    if provider == "siliconflow":
        results = body.get("results", [])
        usage_total_tokens = 0
        meta = body.get("meta", [])
        if isinstance(meta, list) and meta:
            tokens = meta[0].get("tokens", {}) if isinstance(meta[0], dict) else {}
            if isinstance(tokens, dict):
                usage_total_tokens = int(tokens.get("input_tokens", 0) or 0) + int(tokens.get("output_tokens", 0) or 0)
        request_id = str(body.get("id", ""))
    else:
        results = body.get("output", {}).get("results", [])
        usage_total_tokens = int(body.get("usage", {}).get("total_tokens", 0) or 0)
        request_id = str(body.get("request_id", ""))
    if not isinstance(results, list):
        raise RuntimeError("Unexpected rerank response: output.results is not a list")
    return results, {
        "model": payload["model"],
        "request_id": request_id,
        "usage_total_tokens": usage_total_tokens,
    }


def _build_score_maps(
    candidates: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> tuple[list[int], dict[int, float], dict[int, int]]:
    ordered_indexes: list[int] = []
    score_by_index: dict[int, float] = {}
    rank_by_index: dict[int, int] = {}
    seen_indexes: set[int] = set()

    for rank, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        try:
            source_index = int(item.get("index"))
        except Exception:
            continue
        if source_index < 0 or source_index >= len(candidates):
            continue
        if source_index in seen_indexes:
            continue
        seen_indexes.add(source_index)
        ordered_indexes.append(source_index)
        score_by_index[source_index] = float(item.get("relevance_score", 0.0))
        rank_by_index[source_index] = rank

    for source_index in range(len(candidates)):
        if source_index in seen_indexes:
            continue
        ordered_indexes.append(source_index)
        score_by_index[source_index] = 0.0
        rank_by_index[source_index] = len(ordered_indexes)

    return ordered_indexes, score_by_index, rank_by_index


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    top_k: int | None = None,
    timeout: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provider = str(config.get("rerank_provider", DEFAULT_RERANK_PROVIDER)).strip().lower()
    if provider not in {"", "dashscope", "siliconflow"}:
        raise RuntimeError(f"Unsupported rerank_provider: {provider}")
    if not candidates:
        return [], {"provider": provider or DEFAULT_RERANK_PROVIDER, "applied": False, "reason": "empty_candidates"}

    api_key = _resolve_api_key(config)
    if not api_key:
        return list(candidates), {"provider": provider or DEFAULT_RERANK_PROVIDER, "applied": False, "reason": "missing_api_key"}

    instruction = str(config.get("rerank_instruct", DEFAULT_RERANK_INSTRUCT)).strip() or DEFAULT_RERANK_INSTRUCT
    rerank_results, rerank_meta = _request_rerank(
        query=query,
        candidates=candidates,
        config=config,
        instruction=instruction,
        top_k=top_k,
        timeout=timeout,
    )
    ordered_indexes, rerank_score_by_index, rerank_rank_by_index = _build_score_maps(
        candidates,
        rerank_results,
    )

    enriched: list[dict[str, Any]] = []
    for source_index in ordered_indexes:
        row = dict(candidates[source_index])
        row["_rerank_score"] = float(rerank_score_by_index.get(source_index, 0.0))
        row["_rerank_rank"] = int(rerank_rank_by_index.get(source_index, len(candidates) + 1))
        row["_pre_rerank_rank"] = source_index
        enriched.append(row)

    return enriched, {
        "provider": provider or DEFAULT_RERANK_PROVIDER,
        "applied": True,
        "model": rerank_meta["model"],
        "request_id": str(rerank_meta.get("request_id", "")),
        "usage_total_tokens": int(rerank_meta.get("usage_total_tokens", 0) or 0),
    }
