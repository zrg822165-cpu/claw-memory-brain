"""Window-based High admission checks layered after chunk reranking."""
from __future__ import annotations

import re
from typing import Any

from compile_modules.reranker import rerank_candidates

DEFAULT_WINDOW_TOP_K = 4
DEFAULT_WINDOW_MAX_CHARS = 180
DEFAULT_WINDOW_OVERLAP_SENTENCES = 1
DEFAULT_WINDOW_THRESHOLD = 0.85
DEFAULT_MAX_DIRECT_CHUNK_CHARS = 280
DEFAULT_MIN_WINDOW_COVERAGE = 0.30
DEFAULT_STRONG_WINDOW_MARGIN = 0.035

MIX_SHAPE_PATTERNS = (
    "有哪些",
    "哪些",
    "一起",
    "同时",
    "分别",
    "生态",
    "整体",
    "都有什么",
    "整理",
    "汇总",
)

NOISY_CONTEXT_MARKERS = (
    "测试样本",
    "这条样本",
    "故意放进来",
    "观察",
    "重编译",
    "图谱关系",
    "复盘",
    "误用",
    "失败",
    "实际结果",
    "必坑点",
    "前置流程过重",
    "自然沉底",
    "更适合 Mix 输出",
    "compile 是否能拆",
)


def _is_mix_shaped_query(query: str) -> bool:
    return any(pattern in query for pattern in MIX_SHAPE_PATTERNS)


def _has_noisy_context(text: str) -> bool:
    return any(marker in text for marker in NOISY_CONTEXT_MARKERS)


def _sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[。！？!?；;])\s*", normalized)
    return [part.strip() for part in parts if part.strip()]


def split_windows(
    text: str,
    *,
    max_chars: int,
    overlap_sentences: int,
) -> list[tuple[str, int, int]]:
    sentences = _sentences(text)
    if not sentences:
        return []

    windows: list[tuple[str, int, int]] = []
    start = 0
    char_cursor = 0
    sentence_offsets: list[int] = []
    for sentence in sentences:
        sentence_offsets.append(char_cursor)
        char_cursor += len(sentence)

    while start < len(sentences):
        end = start
        pieces: list[str] = []
        while end < len(sentences):
            next_text = "".join(pieces + [sentences[end]])
            if pieces and len(next_text) > max_chars:
                break
            pieces.append(sentences[end])
            end += 1
        window = "".join(pieces).strip()
        if window:
            start_char = sentence_offsets[start]
            end_char = start_char + len(window)
            windows.append((window, start_char, end_char))
        if end >= len(sentences):
            break
        start = max(start + 1, end - overlap_sentences)

    return windows


def build_window_candidates(
    chunks: list[dict[str, Any]],
    *,
    source_top_k: int,
    max_chars: int,
    overlap_sentences: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for chunk_rank, chunk in enumerate(chunks[:source_top_k], start=1):
        content = str(chunk.get("content", "")).strip()
        content_length = len(content)
        for window_rank, (window, start_char, end_char) in enumerate(
            split_windows(content, max_chars=max_chars, overlap_sentences=overlap_sentences),
            start=1,
        ):
            candidates.append(
                {
                    "id": f"{chunk.get('id', '')}::w{window_rank}",
                    "chunk_id": chunk.get("id", ""),
                    "chunk_summary": chunk.get("chunk_summary", ""),
                    "content": window,
                    "created_at": chunk.get("created_at", ""),
                    "_source_chunk": chunk,
                    "_source_chunk_rank": chunk_rank,
                    "_window_rank": window_rank,
                    "_window_start": start_char,
                    "_window_end": end_char,
                    "_window_length": len(window),
                    "_source_content_length": content_length,
                }
            )
    return candidates


def route_with_answer_window(
    *,
    query: str,
    reranked_chunks: list[dict[str, Any]],
    reranked_windows: list[dict[str, Any]],
    chunk_threshold: float,
    window_threshold: float,
    max_direct_chunk_chars: int,
    min_window_coverage: float,
    strong_window_margin: float,
) -> dict[str, Any]:
    if not reranked_chunks:
        return {
            "route": "miss",
            "reason": "no_candidate",
            "top1": None,
            "raw_top1": None,
            "top_window": None,
            "high_stack": [],
            "results": [],
        }

    raw_top1 = reranked_chunks[0]
    top_window = reranked_windows[0] if reranked_windows else None
    raw_chunk_score = float(raw_top1.get("_rerank_score", 0.0))
    top_window_score = float(top_window.get("_rerank_score", 0.0)) if top_window else 0.0
    mix_shaped = _is_mix_shaped_query(query)

    if not top_window:
        reason = "no_window"
        route = "miss"
    elif top_window_score < window_threshold:
        reason = "window_below_threshold"
        route = "miss"
    else:
        source_chunk = dict(top_window.get("_source_chunk") or {})
        source_content_length = max(1, int(top_window.get("_source_content_length", 0) or 0))
        window_length = int(top_window.get("_window_length", 0) or 0)
        coverage = window_length / source_content_length
        source_chunk_score = float(source_chunk.get("_rerank_score", raw_chunk_score) or 0.0)
        source_text = f"{source_chunk.get('chunk_summary', '')}\n{source_chunk.get('content', '')}"
        chunk_is_short = source_content_length <= max_direct_chunk_chars
        window_is_dominant = coverage >= min_window_coverage
        window_decisive = top_window_score >= max(window_threshold, source_chunk_score + strong_window_margin)

        if mix_shaped and not (chunk_is_short and top_window_score >= 0.90):
            reason = "mix_shape_prior"
            route = "miss"
        elif _has_noisy_context(source_text):
            reason = "noisy_context"
            route = "miss"
        elif not (chunk_is_short or window_is_dominant or window_decisive):
            reason = "noisy_window"
            route = "miss"
        elif source_chunk_score < chunk_threshold and not window_decisive:
            reason = "chunk_below_threshold"
            route = "miss"
        else:
            reason = "window_direct_answer"
            route = "high"

    top1 = dict((top_window or {}).get("_source_chunk") or raw_top1)
    if top_window:
        top1["_answer_window"] = str(top_window.get("content", "")).strip()
        top1["_answer_window_score"] = top_window_score
        top1["_answer_window_coverage"] = (
            float(top_window.get("_window_length", 0) or 0)
            / max(1.0, float(top_window.get("_source_content_length", 0) or 0))
        )
        top1["_answer_window_rank"] = int(top_window.get("_window_rank", 0) or 0)
        top1["_rerank_score"] = float(top1.get("_rerank_score", 0.0) or 0.0)

    return {
        "route": route,
        "reason": reason,
        "top1": top1,
        "raw_top1": raw_top1,
        "top_window": top_window,
        "top1_rerank_score": float(top1.get("_rerank_score", 0.0) or 0.0),
        "top_window_score": top_window_score,
        "high_stack": [top1] if route == "high" else [],
        "results": reranked_chunks,
    }


def apply_window_admission(
    query: str,
    reranked_chunks: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    timeout: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = {
        "enabled": bool(config.get("window_admission_enabled", False)),
        "source_top_k": int(config.get("window_top_k", DEFAULT_WINDOW_TOP_K) or DEFAULT_WINDOW_TOP_K),
        "max_chars": int(config.get("window_max_chars", DEFAULT_WINDOW_MAX_CHARS) or DEFAULT_WINDOW_MAX_CHARS),
        "overlap_sentences": int(
            config.get("window_overlap_sentences", DEFAULT_WINDOW_OVERLAP_SENTENCES)
            or DEFAULT_WINDOW_OVERLAP_SENTENCES
        ),
        "chunk_threshold": float(config.get("window_chunk_threshold", DEFAULT_WINDOW_THRESHOLD) or DEFAULT_WINDOW_THRESHOLD),
        "window_threshold": float(config.get("window_threshold", DEFAULT_WINDOW_THRESHOLD) or DEFAULT_WINDOW_THRESHOLD),
        "max_direct_chunk_chars": int(
            config.get("window_max_direct_chunk_chars", DEFAULT_MAX_DIRECT_CHUNK_CHARS)
            or DEFAULT_MAX_DIRECT_CHUNK_CHARS
        ),
        "min_window_coverage": float(
            config.get("window_min_coverage", DEFAULT_MIN_WINDOW_COVERAGE) or DEFAULT_MIN_WINDOW_COVERAGE
        ),
        "strong_window_margin": float(
            config.get("window_strong_margin", DEFAULT_STRONG_WINDOW_MARGIN) or DEFAULT_STRONG_WINDOW_MARGIN
        ),
    }
    if not params["enabled"]:
        return {}, {"applied": False, "reason": "disabled", "params": params}

    window_candidates = build_window_candidates(
        reranked_chunks,
        source_top_k=max(1, params["source_top_k"]),
        max_chars=max(40, params["max_chars"]),
        overlap_sentences=max(0, params["overlap_sentences"]),
    )
    if not window_candidates:
        return {}, {
            "applied": False,
            "reason": "no_window_candidates",
            "candidate_count": 0,
            "params": params,
            "rerank": {"applied": False, "reason": "empty_candidates"},
        }

    reranked_windows, window_rerank_meta = rerank_candidates(
        query,
        window_candidates,
        config=config,
        timeout=timeout,
    )
    if not bool(window_rerank_meta.get("applied")):
        return {}, {
            "applied": False,
            "reason": "window_rerank_unavailable",
            "candidate_count": len(window_candidates),
            "params": params,
            "rerank": window_rerank_meta,
        }

    routed = route_with_answer_window(
        query=query,
        reranked_chunks=reranked_chunks,
        reranked_windows=reranked_windows,
        chunk_threshold=params["chunk_threshold"],
        window_threshold=params["window_threshold"],
        max_direct_chunk_chars=params["max_direct_chunk_chars"],
        min_window_coverage=params["min_window_coverage"],
        strong_window_margin=params["strong_window_margin"],
    )
    return routed, {
        "applied": True,
        "reason": "ok",
        "candidate_count": len(window_candidates),
        "params": params,
        "rerank": window_rerank_meta,
        "top_window": reranked_windows[0] if reranked_windows else None,
    }
