"""Thin router between reranked High answers and fallback miss/mix candidates."""
from __future__ import annotations

from typing import Any


DEFAULT_HIGH_THRESHOLD = 0.017
DEFAULT_HIGH_STACK_SIZE = 4
DEFAULT_STACK_SUPPORT_FLOOR = 0.08
DEFAULT_RELEVANCE_STACK_MARGIN = 0.004
DEFAULT_RERANK_SCORE_THRESHOLD = 0.85
DEFAULT_RERANK_STACK_MARGIN = 0.08
DEFAULT_RERANK_TIME_BUCKET = 0.02


def route_score(row: dict[str, Any]) -> float:
    """Normalize backend-specific retrieval signals into one relevance score."""
    if "_hybrid_score" in row:
        return float(row.get("_hybrid_score", 0.0))
    if "_vec_score" in row:
        return float(row.get("_vec_score", 0.0))
    if "_bm25" in row:
        return max(0.0, 1.0 / (1.0 + abs(float(row.get("_bm25", 0.0)))))
    if "_distance" in row:
        return max(0.0, 1.0 / (1.0 + float(row.get("_distance", 0.0))))
    return 0.0


def _created_at_key(value: str) -> int:
    digits = "".join(ch for ch in value if ch.isdigit())
    return int(digits) if digits else 0


def _has_rerank_scores(candidates: list[dict[str, Any]]) -> bool:
    return any("_rerank_score" in row for row in candidates)


def _rerank_bucket(row: dict[str, Any]) -> int:
    score = float(row.get("_rerank_score", 0.0))
    if score <= 0.0:
        return 0
    return int(score / DEFAULT_RERANK_TIME_BUCKET)


def _high_stack_sort_key(row: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    if bool(row.get("_has_rerank_score", False)):
        return (
            int(row.get("_penalized", 0)),
            -_rerank_bucket(row),
            -_created_at_key(str(row.get("created_at", ""))),
            -int(float(row.get("_rerank_score", 0.0)) * 1_000_000),
            -int(float(row.get("_route_score", 0.0)) * 1_000_000),
            int(row.get("_candidate_rank", 0)),
        )
    return (
        int(row.get("_penalized", 0)),
        -_created_at_key(str(row.get("created_at", ""))),
        -int(float(row.get("_route_score", 0.0)) * 1_000_000),
        -int(float(row.get("_direct_support", 0.0)) * 1_000_000),
        int(row.get("_candidate_rank", 0)),
        0,
    )


def _mix_sort_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        -int(float(row.get("_route_score", 0.0)) * 1_000_000),
        -int(float(row.get("_rerank_score", 0.0)) * 1_000_000),
        -int(float(row.get("_direct_support", 0.0)) * 1_000_000),
        int(row.get("_candidate_rank", 0)),
    )


def enrich_candidates(
    query: str,
    results: list[dict[str, Any]],
    feedback_map: dict[str, dict[str, str | int]] | None = None,
) -> list[dict[str, Any]]:
    del query
    feedback_map = feedback_map or {}
    enriched_results: list[dict[str, Any]] = []
    for rank, row in enumerate(results):
        enriched = dict(row)
        fb = feedback_map.get(str(row["id"]), {})
        enriched["_candidate_rank"] = rank
        enriched["_route_score"] = route_score(row)
        enriched["_direct_support"] = 0.0
        enriched["_has_rerank_score"] = bool("_rerank_score" in row)
        enriched["_rerank_score"] = float(row.get("_rerank_score", 0.0))
        enriched["_rerank_rank"] = int(row.get("_rerank_rank", rank + 1))
        enriched["_penalized"] = int(fb.get("penalized", 0))
        enriched["_last_feedback_at"] = str(fb.get("last_feedback_at", ""))
        enriched_results.append(enriched)
    return enriched_results


def build_high_stack(
    candidates: list[dict[str, Any]],
    *,
    relevance_threshold: float,
    rerank_threshold: float,
    stack_size: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    if _has_rerank_scores(candidates):
        best_rerank_score = max(float(row.get("_rerank_score", 0.0)) for row in candidates)
        if best_rerank_score < rerank_threshold:
            return []

        rerank_floor = max(rerank_threshold, best_rerank_score - DEFAULT_RERANK_STACK_MARGIN)
        eligible = []
        for row in candidates:
            row_rerank = float(row.get("_rerank_score", 0.0))
            if row_rerank < rerank_floor:
                continue
            eligible.append(dict(row))
        eligible.sort(key=_high_stack_sort_key)
        return eligible[:stack_size]

    best_route_score = max(float(row.get("_route_score", 0.0)) for row in candidates)
    relevance_floor = max(relevance_threshold, best_route_score - DEFAULT_RELEVANCE_STACK_MARGIN)
    eligible = [
        dict(row)
        for row in candidates
        if float(row.get("_route_score", 0.0)) >= relevance_floor
    ]
    eligible.sort(key=_high_stack_sort_key)
    return eligible[:stack_size]


def build_mix_candidates(
    candidates: list[dict[str, Any]],
    *,
    relevance_threshold: float,
    stack_size: int,
) -> list[dict[str, Any]]:
    eligible = [
        dict(row)
        for row in candidates
        if float(row.get("_route_score", 0.0)) >= relevance_threshold
    ]
    eligible.sort(key=_mix_sort_key)
    return eligible[:stack_size]


def route_by_top_candidate(
    query: str,
    results: list[dict[str, Any]],
    *,
    relevance_threshold: float = DEFAULT_HIGH_THRESHOLD,
    rerank_threshold: float = DEFAULT_RERANK_SCORE_THRESHOLD,
    stack_size: int = DEFAULT_HIGH_STACK_SIZE,
    feedback_map: dict[str, dict[str, str | int]] | None = None,
) -> dict[str, Any]:
    candidates = enrich_candidates(query, results, feedback_map=feedback_map)
    if not candidates:
        return {
            "route": "miss",
            "legacy_route": "mix_required",
            "reason": "no_candidate",
            "threshold": relevance_threshold,
            "rerank_threshold": rerank_threshold,
            "top1": None,
            "raw_top1": None,
            "answer_text": "",
            "stack": [],
            "high_stack": [],
            "mix_candidates": [],
            "results": [],
        }

    raw_top1 = candidates[0]
    raw_top1_score = float(raw_top1.get("_route_score", 0.0))
    raw_top1_rerank_score = float(raw_top1.get("_rerank_score", 0.0))

    high_stack = build_high_stack(
        candidates,
        relevance_threshold=relevance_threshold,
        rerank_threshold=rerank_threshold,
        stack_size=stack_size,
    )
    mix_candidates = build_mix_candidates(
        candidates,
        relevance_threshold=relevance_threshold,
        stack_size=stack_size,
    )

    if high_stack:
        top1 = high_stack[0]
        route = "high"
        reason = "direct_answer_stack" if _has_rerank_scores(candidates) else "relevance_stack"
        stack = high_stack
        answer_text = str(top1.get("chunk_summary", "")).strip()
    else:
        top1 = raw_top1
        route = "miss"
        if _has_rerank_scores(candidates):
            reason = "rerank_below_threshold" if raw_top1_rerank_score < rerank_threshold else "no_high_stack"
        else:
            reason = "top1_under_threshold" if raw_top1_score < relevance_threshold else "no_high_stack"
        stack = mix_candidates
        answer_text = ""

    return {
        "route": route,
        "legacy_route": "high" if route == "high" else "mix_required",
        "reason": reason,
        "threshold": relevance_threshold,
        "rerank_threshold": rerank_threshold,
        "top1_score": float(top1.get("_route_score", 0.0)) if top1 else 0.0,
        "top1_rerank_score": float(top1.get("_rerank_score", 0.0)) if top1 else 0.0,
        "raw_top1_score": raw_top1_score,
        "raw_top1_rerank_score": raw_top1_rerank_score,
        "top1": top1,
        "raw_top1": raw_top1,
        "answer_text": answer_text,
        "stack": stack,
        "high_stack": high_stack,
        "mix_candidates": mix_candidates,
        "results": candidates,
    }


def route_high_or_mix(
    query: str,
    results: list[dict[str, Any]],
    threshold: float = DEFAULT_HIGH_THRESHOLD,
    stack_size: int = DEFAULT_HIGH_STACK_SIZE,
    feedback_map: dict[str, dict[str, str | int]] | None = None,
    rerank_threshold: float = DEFAULT_RERANK_SCORE_THRESHOLD,
) -> dict[str, Any]:
    return route_by_top_candidate(
        query,
        results,
        relevance_threshold=threshold,
        rerank_threshold=rerank_threshold,
        stack_size=stack_size,
        feedback_map=feedback_map,
    )
