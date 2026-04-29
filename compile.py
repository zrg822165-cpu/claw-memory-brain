"""Event-chunk compile entrypoint."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from compile_modules.event_chunker import build_event_chunks, load_config, now_timestamp
from compile_modules.event_filter import hard_filter_candidate
from compile_modules.ingest_adjudicator import adjudicate_chunks_against_top1
from compile_modules.indexer import (
    DEFAULT_HIGH_SEARCH_PROJECTION_DIR,
    DEFAULT_HIGH_SEARCH_PROJECTION_FTS_PATH,
    rebuild_high_search_projection_fts_index,
    rebuild_high_search_projection_lance_index,
)
from compile_modules.event_store import DEFAULT_EVENT_DB_PATH, apply_ingest_changes

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent


def _iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob("*.md") if p.is_file())
    return []


def compile_source_file(
    source_path: Path,
    config: dict,
    db_path: Path,
    lance_dir: Path,
    fts_path: Path,
    dry_run: bool = False,
    timeout: int | None = None,
) -> dict[str, object]:
    markdown_text = source_path.read_text(encoding="utf-8").strip()
    if not markdown_text:
        logger.info("Skipping empty file: %s", source_path)
        return {
            "inserted": 0,
            "deleted": 0,
            "target": None,
            "skipped_duplicates": 0,
            "adjudications": [],
        }

    created_at = now_timestamp()
    source_title = source_path.name
    candidate_chunks = build_event_chunks(
        markdown_text=markdown_text,
        source_title=source_title,
        created_at=created_at,
        config=config,
        timeout=timeout,
    )
    kept_chunks = []
    filtered_count = 0
    for chunk in candidate_chunks:
        decision = hard_filter_candidate(chunk)
        if decision.keep:
            kept_chunks.append(chunk)
        else:
            filtered_count += 1
            logger.info("Filtered candidate chunk from %s: %s", source_title, decision.reason)

    if filtered_count:
        logger.info(
            "Hard filter kept %d/%d candidate chunks from %s",
            len(kept_chunks),
            len(candidate_chunks),
            source_title,
        )
    chunks = kept_chunks

    adjudication = adjudicate_chunks_against_top1(
        chunks,
        config=config,
        lance_dir=lance_dir,
        fts_path=fts_path,
        timeout=timeout,
    )
    chunks = list(adjudication["insert_chunks"])
    delete_chunk_ids = list(adjudication["delete_chunk_ids"])
    adjudications = list(adjudication["adjudications"])

    if dry_run:
        payload = {
            "event_chunks": [chunk.to_dict() for chunk in chunks],
            "delete_chunk_ids": delete_chunk_ids,
            "adjudications": adjudications,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return {
            "inserted": len(chunks),
            "deleted": len(delete_chunk_ids),
            "target": None,
            "skipped_duplicates": 0,
            "adjudications": adjudications,
        }

    target, inserted, deleted, skipped_duplicates = apply_ingest_changes(
        chunks,
        delete_chunk_ids,
        db_path=db_path,
    )
    logger.info(
        "Applied ingest adjudication for %s -> inserted=%d deleted=%d skipped_duplicates=%d",
        source_path.name,
        inserted,
        deleted,
        skipped_duplicates,
    )
    logger.info(
        "Saved adjudicated event chunks from %s to %s",
        source_path.name,
        target,
    )
    return {
        "inserted": inserted,
        "deleted": deleted,
        "target": target,
        "skipped_duplicates": skipped_duplicates,
        "adjudications": adjudications,
    }


def _rebuild_high_indexes(
    *,
    db_path: Path,
    high_search_projection_dir: Path,
    high_search_projection_fts_path: Path,
    config: dict,
    timeout: int | None = None,
) -> tuple[int, int]:
    indexed_target, indexed_count = rebuild_high_search_projection_lance_index(
        db_path=db_path,
        lance_dir=high_search_projection_dir,
        config=config,
        timeout=timeout,
    )
    logger.info(
        "Rebuilt High search projection LanceDB index with %d active chunks at %s",
        indexed_count,
        indexed_target,
    )
    fts_target, fts_count = rebuild_high_search_projection_fts_index(
        db_path=db_path,
        fts_path=high_search_projection_fts_path,
    )
    logger.info(
        "Rebuilt High search projection FTS5 index with %d active chunks at %s",
        fts_count,
        fts_target,
    )
    return indexed_count, fts_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile memory markdown files into canonical event chunks"
    )
    parser.add_argument("input", help="Markdown file or directory")
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_EVENT_DB_PATH),
        help="Canonical event-chunk SQLite database path",
    )
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
        "--timeout",
        type=int,
        default=None,
        help="Override cloud LLM timeout in seconds",
    )
    parser.add_argument(
        "--skip-high-index",
        action="store_true",
        help="Skip rebuilding the High search projection indexes after compile",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print event chunks without writing files")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    files = _iter_input_files(input_path)
    if not files:
        raise SystemExit(f"No markdown files found under: {input_path}")

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    config = load_config(config_path)

    db_path = Path(args.db_path)
    high_search_projection_dir = Path(args.high_search_projection_dir)
    high_search_projection_fts_path = Path(args.high_search_projection_fts_path)
    if not args.dry_run and not args.skip_high_index:
        logger.info("Bootstrapping High search projection before compile for ingest adjudication")
        _rebuild_high_indexes(
            db_path=db_path,
            high_search_projection_dir=high_search_projection_dir,
            high_search_projection_fts_path=high_search_projection_fts_path,
            config=config,
            timeout=args.timeout,
        )

    total_inserted = 0
    total_deleted = 0
    failures: list[Path] = []

    for source_path in files:
        try:
            result = compile_source_file(
                source_path=source_path,
                config=config,
                db_path=db_path,
                lance_dir=high_search_projection_dir,
                fts_path=high_search_projection_fts_path,
                dry_run=args.dry_run,
                timeout=args.timeout,
            )
            inserted = int(result["inserted"])
            deleted = int(result["deleted"])
            total_inserted += inserted
            total_deleted += deleted
            if not args.dry_run and not args.skip_high_index and (inserted > 0 or deleted > 0):
                _rebuild_high_indexes(
                    db_path=db_path,
                    high_search_projection_dir=high_search_projection_dir,
                    high_search_projection_fts_path=high_search_projection_fts_path,
                    config=config,
                    timeout=args.timeout,
                )
        except Exception as exc:
            failures.append(source_path)
            logger.error("Failed to compile %s: %s", source_path, exc)

    logger.info(
        "Done. Inserted %d event chunks and deleted %d old chunk(s) from %d file(s).",
        total_inserted,
        total_deleted,
        len(files) - len(failures),
    )
    if failures:
        logger.error("Failed files: %d", len(failures))
        for failed in failures:
            logger.error("  - %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
