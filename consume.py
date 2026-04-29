"""Minimal High retrieval entrypoint over LanceDB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compile_modules.event_chunker import load_config, now_timestamp
from compile_modules.event_store import (
    DEFAULT_EVENT_DB_PATH,
    fetch_high_feedback_map,
    fetch_pending_delete_queue,
    mark_high_error,
    mark_high_not_helpful,
)
from compile_modules.indexer import (
    DEFAULT_HIGH_SEARCH_PROJECTION_DIR,
    DEFAULT_HIGH_SEARCH_PROJECTION_FTS_PATH,
    DEFAULT_HIGH_SEARCH_PROJECTION_FTS_WEIGHT,
    DEFAULT_HIGH_SEARCH_PROJECTION_VECTOR_WEIGHT,
    search_high_projection_fts,
    search_high_projection_hybrid,
    search_high_projection_lance,
)
from compile_modules.high_router import (
    DEFAULT_HIGH_STACK_SIZE,
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_RERANK_SCORE_THRESHOLD,
    route_high_or_mix,
)
from compile_modules.reranker import rerank_candidates
from compile_modules.window_admission import apply_window_admission

ROOT = Path(__file__).resolve().parent
DEFAULT_HIGH_TOP_K = 8
DEFAULT_RECALL_POOL_SIZE = 30


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search High retrieval indexes over canonical event chunks"
    )
    parser.add_argument("query", nargs="?", help="High retrieval query")
    parser.add_argument(
        "--config",
        default=str(ROOT / "config.json"),
        help="Config JSON path",
    )
    parser.add_argument(
        "--high-search-projection-dir",
        "--high-lance-dir",
        dest="high_search_projection_dir",
        default=str(DEFAULT_HIGH_SEARCH_PROJECTION_DIR),
        help="High search projection LanceDB directory",
    )
    parser.add_argument(
        "--high-search-projection-fts-path",
        "--high-fts-path",
        dest="high_search_projection_fts_path",
        default=str(DEFAULT_HIGH_SEARCH_PROJECTION_FTS_PATH),
        help="High search projection FTS5 SQLite path",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_EVENT_DB_PATH),
        help="Canonical event-chunk SQLite database path",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_HIGH_TOP_K,
        help="Number of results to return",
    )
    parser.add_argument(
        "--recall-pool-size",
        type=int,
        default=DEFAULT_RECALL_POOL_SIZE,
        help="Broad-recall candidate pool size before reranking",
    )
    parser.add_argument(
        "--vector-weight",
        type=float,
        default=DEFAULT_HIGH_SEARCH_PROJECTION_VECTOR_WEIGHT,
        help="Weight applied to vector ranks during hybrid recall",
    )
    parser.add_argument(
        "--fts-weight",
        type=float,
        default=DEFAULT_HIGH_SEARCH_PROJECTION_FTS_WEIGHT,
        help="Weight applied to FTS ranks during hybrid recall",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_HIGH_THRESHOLD,
        help="Relevance threshold for top1 routing; below this routes to mix",
    )
    parser.add_argument(
        "--disable-rerank",
        action="store_true",
        help="Skip cloud reranking and use retrieval order directly",
    )
    parser.add_argument(
        "--rerank-threshold",
        type=float,
        default=DEFAULT_RERANK_SCORE_THRESHOLD,
        help="Minimum rerank relevance score required for a High stack to form",
    )
    parser.add_argument(
        "--stack-size",
        type=int,
        default=DEFAULT_HIGH_STACK_SIZE,
        help="Maximum High stack size",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Override embedding timeout in seconds",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON results",
    )
    parser.add_argument(
        "--backend",
        choices=["hybrid", "vector", "fts"],
        default="hybrid",
        help="High retrieval backend",
    )
    parser.add_argument(
        "--mark-not-helpful",
        default="",
        help="Mark a chunk_id as not helpful in High and exit",
    )
    parser.add_argument(
        "--mark-error",
        default="",
        help="Mark a chunk_id as wrong in High, enqueue it for delete review, and exit",
    )
    parser.add_argument(
        "--feedback-query",
        default="",
        help="Optional original user query to store with --mark-error",
    )
    parser.add_argument(
        "--feedback-note",
        default="",
        help="Optional note to store with --mark-error",
    )
    parser.add_argument(
        "--show-pending-delete",
        action="store_true",
        help="Show current pending delete queue and exit",
    )
    args = parser.parse_args()

    if args.show_pending_delete:
        payload = {
            "pending_delete_queue": fetch_pending_delete_queue(Path(args.db_path)),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.mark_not_helpful:
        mark_high_not_helpful(
            chunk_id=args.mark_not_helpful.strip(),
            feedback_at=now_timestamp(),
            db_path=Path(args.db_path),
        )
        payload = {
            "ok": True,
            "action": "mark_not_helpful",
            "chunk_id": args.mark_not_helpful.strip(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.mark_error:
        mark_high_error(
            chunk_id=args.mark_error.strip(),
            feedback_at=now_timestamp(),
            source_query=args.feedback_query.strip(),
            feedback_note=args.feedback_note.strip(),
            db_path=Path(args.db_path),
        )
        payload = {
            "ok": True,
            "action": "mark_error",
            "chunk_id": args.mark_error.strip(),
            "source_query": args.feedback_query.strip(),
            "feedback_note": args.feedback_note.strip(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if not args.query:
        parser.error("the following arguments are required: query")

    config = load_config(Path(args.config))
    vector_weight = float(config.get("high_vector_weight", DEFAULT_HIGH_SEARCH_PROJECTION_VECTOR_WEIGHT))
    fts_weight = float(config.get("high_fts_weight", DEFAULT_HIGH_SEARCH_PROJECTION_FTS_WEIGHT))
    if args.vector_weight != DEFAULT_HIGH_SEARCH_PROJECTION_VECTOR_WEIGHT:
        vector_weight = args.vector_weight
    if args.fts_weight != DEFAULT_HIGH_SEARCH_PROJECTION_FTS_WEIGHT:
        fts_weight = args.fts_weight
    recall_pool_size = max(args.top_k, args.recall_pool_size)
    if args.backend == "vector":
        results = search_high_projection_lance(
            query=args.query,
            config=config,
            lance_dir=Path(args.high_search_projection_dir),
            top_k=recall_pool_size,
            timeout=args.timeout,
        )
    elif args.backend == "fts":
        results = search_high_projection_fts(
            query=args.query,
            fts_path=Path(args.high_search_projection_fts_path),
            top_k=recall_pool_size,
        )
    else:
        results = search_high_projection_hybrid(
            query=args.query,
            config=config,
            lance_dir=Path(args.high_search_projection_dir),
            fts_path=Path(args.high_search_projection_fts_path),
            top_k=recall_pool_size,
            timeout=args.timeout,
            vector_weight=vector_weight,
            fts_weight=fts_weight,
        )

    rerank_meta: dict[str, Any] = {"applied": False, "reason": "disabled"}
    if not args.disable_rerank:
        results, rerank_meta = rerank_candidates(
            args.query,
            results,
            config=config,
            timeout=args.timeout,
        )

    routed = route_high_or_mix(
        args.query,
        results,
        threshold=args.threshold,
        stack_size=args.stack_size,
        feedback_map=fetch_high_feedback_map(Path(args.db_path)),
        rerank_threshold=args.rerank_threshold,
    )
    routed["rerank"] = rerank_meta

    window_routed: dict[str, Any] = {}
    if args.disable_rerank or not rerank_meta.get("applied"):
        window_meta = {"applied": False, "reason": "upstream_rerank_disabled"}
    else:
        window_routed, window_meta = apply_window_admission(
            args.query,
            results,
            config=config,
            timeout=args.timeout,
        )
    routed["window_admission"] = window_meta
    if window_meta.get("applied"):
        # Keep the main retrieval payload, but let the window layer override final admission.
        routed["route"] = str(window_routed.get("route", routed["route"]))
        routed["legacy_route"] = "high" if routed["route"] == "high" else "mix_required"
        routed["reason"] = str(window_routed.get("reason", routed.get("reason", "")))
        routed["top1"] = window_routed.get("top1")
        routed["raw_top1"] = window_routed.get("raw_top1")
        routed["top1_rerank_score"] = float(window_routed.get("top1_rerank_score", routed.get("top1_rerank_score", 0.0)) or 0.0)
        routed["high_stack"] = list(window_routed.get("high_stack") or [])
        routed["stack"] = list(window_routed.get("high_stack") or routed.get("mix_candidates") or [])
        routed["top_window_score"] = float(window_routed.get("top_window_score", 0.0) or 0.0)
        if routed["route"] == "high":
            routed["answer_text"] = str((routed.get("top1") or {}).get("_answer_window", "")).strip() or str((routed.get("top1") or {}).get("chunk_summary", "")).strip()
        else:
            routed["answer_text"] = ""

    if args.json:
        print(json.dumps(routed, ensure_ascii=False, indent=2))
        return

    if routed["route"] != "high":
        print(f"route: {routed['route']}")
        print(f"reason: {routed.get('reason', '')}")
        if rerank_meta.get("applied"):
            print(f"rerank_model: {rerank_meta.get('model', '')}")
        return

    top1 = routed.get("top1") or {}
    if routed.get("answer_text"):
        print("summary:")
        print(routed["answer_text"])
    elif top1.get("chunk_summary"):
        print("summary:")
        print(top1.get("chunk_summary", ""))

    if top1.get("content"):
        print()
        print("content:")
        print(top1.get("content", ""))

    if rerank_meta.get("applied"):
        print()
        print(f"rerank_model: {rerank_meta.get('model', '')}")

    if bool((routed.get("window_admission") or {}).get("applied")):
        print()
        print(f"window_reason: {routed.get('reason', '')}")


if __name__ == "__main__":
    main()
