from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from compile_modules.event_chunker import load_config
from compile_modules.event_store import DEFAULT_EVENT_DB_PATH
from compile_modules.indexer import (
    DEFAULT_HIGH_SEARCH_PROJECTION_DIR,
    DEFAULT_HIGH_SEARCH_PROJECTION_FTS_PATH,
)
from compile_modules.pending_delete_reviewer import run_pending_delete_review_if_due


ROOT = Path(__file__).resolve().parent


def _run_review(
    *,
    config: dict,
    db_path: Path,
    high_search_projection_dir: Path,
    high_search_projection_fts_path: Path,
    timeout: int | None,
    interval_minutes: int,
) -> dict[str, object]:
    return run_pending_delete_review_if_due(
        config=config,
        db_path=db_path,
        lance_dir=high_search_projection_dir,
        fts_path=high_search_projection_fts_path,
        timeout=timeout,
        interval_minutes=interval_minutes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Resident memory service")
    parser.add_argument(
        "--config",
        default=str(ROOT / "config.json"),
        help="Config JSON path",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_EVENT_DB_PATH),
        help="Canonical event-chunk SQLite database path",
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
        help="Override review timeout in seconds",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=30,
        help="Pending delete review interval in minutes",
    )
    parser.add_argument(
        "--tick-seconds",
        type=int,
        default=5,
        help="Loop sleep tick in seconds",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one review cycle and exit",
    )
    parser.add_argument(
        "--run-on-start",
        action="store_true",
        help="Run one review attempt immediately on startup before entering the timer loop",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("memory_service")

    config = load_config(Path(args.config))
    db_path = Path(args.db_path)
    high_search_projection_dir = Path(args.high_search_projection_dir)
    high_search_projection_fts_path = Path(args.high_search_projection_fts_path)

    logger.info("memory service started")
    logger.info("db: %s", db_path)
    logger.info("high_search_projection_lance: %s", high_search_projection_dir)
    logger.info("high_search_projection_fts: %s", high_search_projection_fts_path)
    logger.info("pending delete interval: %s minutes", args.interval_minutes)

    def do_review() -> dict[str, object]:
        result = _run_review(
            config=config,
            db_path=db_path,
            high_search_projection_dir=high_search_projection_dir,
            high_search_projection_fts_path=high_search_projection_fts_path,
            timeout=args.timeout,
            interval_minutes=args.interval_minutes,
        )
        logger.info("pending delete review: %s", json.dumps(result, ensure_ascii=False))
        return result

    if args.run_on_start or args.once:
        do_review()
    if args.once:
        return

    next_run_at = time.time() + args.interval_minutes * 60
    logger.info("service loop entered; close this window to stop")
    try:
        while True:
            now = time.time()
            if now >= next_run_at:
                do_review()
                next_run_at = time.time() + args.interval_minutes * 60
            time.sleep(max(1, args.tick_seconds))
    except KeyboardInterrupt:
        logger.info("memory service stopped by keyboard interrupt")
    except Exception as exc:
        logger.exception("memory service crashed: %s", exc)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
