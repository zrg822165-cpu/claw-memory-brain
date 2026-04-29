from __future__ import annotations

import argparse
import json
from pathlib import Path

from compile_modules.event_chunker import load_config
from compile_modules.event_store import DEFAULT_EVENT_DB_PATH
from compile_modules.indexer import (
    DEFAULT_HIGH_SEARCH_PROJECTION_DIR,
    DEFAULT_HIGH_SEARCH_PROJECTION_FTS_PATH,
)
from compile_modules.pending_delete_reviewer import run_pending_delete_review_if_due
from compile_modules.runtime_paths import CONFIG_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Review pending delete queue")
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
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
        default=0,
        help="Minimum interval between runs; default 0 for manual trigger",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    result = run_pending_delete_review_if_due(
        config=config,
        db_path=Path(args.db_path),
        lance_dir=Path(args.high_search_projection_dir),
        fts_path=Path(args.high_search_projection_fts_path),
        timeout=args.timeout,
        interval_minutes=args.interval_minutes,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
