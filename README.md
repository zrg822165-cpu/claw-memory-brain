# claw-memory-brain

An experimental long-term memory and high-confidence retrieval system for AI coding workflows.

This project is built around a practical question: after an AI assistant retrieves something relevant from project memory, how do you decide whether the result is good enough for a direct answer, or whether it is still just a noisy background chunk?

`claw-memory-brain` is an attempt to turn that into an end-to-end pipeline: memory compilation, candidate retrieval, reranking, and final High/Miss style routing.

## What It Tries To Solve

Typical coding assistants have two recurring problems:

- they lose working memory across sessions
- even when retrieval finds the right topic, the returned chunk may still be too noisy to answer directly

This repo focuses on the second problem as much as the first. It does not only store and search memories. It also tries to decide whether a retrieved chunk is actually a direct-answer chunk.

## Core Flow

The current pipeline is roughly:

1. Write project memory as markdown under `memory/`
2. Compile markdown into canonical event chunks
3. Store chunks in SQLite and rebuild High-search indexes
4. Retrieve candidates with vector search, FTS, or hybrid search
5. Rerank candidates with a cloud reranker
6. Route the result:
   - `high`: the chunk body is judged direct enough to answer
   - `miss`: relevant topic exists, but current chunk quality is not good enough for direct answer

There is also an optional window-admission layer in the codebase, but it is currently disabled by default in config.

## Main Components

- `compile.py`
  Compiles one markdown file or a whole directory into canonical chunks, runs ingest adjudication, and rebuilds High indexes.

- `consume.py`
  Query entrypoint for retrieval and routing. Supports `vector`, `fts`, and `hybrid` backends, plus rerank and JSON output.

- `memory_service.py`
  Small resident service loop for pending-delete review tasks.

- `compile_modules/`
  Internal modules for chunking, filtering, storage, indexing, reranking, routing, window admission, and adjudication.

## Repository Layout

- `memory/`: source markdown memories
- `data/`: compiled runtime data and indexes
- `compile_modules/`: main implementation modules
- `config.json`: public-safe base configuration template
- `config.local.json`: optional local secret override file, loaded automatically if present
- `.ai-tmp/`: local experiments, evaluation assets, notes, and temporary runners

## Environment

This project is currently WSL-first.

In practice, you should prefer running the following inside WSL:

- dependency installation
- `python` commands
- `git`
- compile/index rebuild tasks
- service startup

Crossing between Windows and WSL is possible, but path, permission, and auth differences can get in the way.

## Quick Start

### 1. Bootstrap inside WSL

```bash
./bootstrap_wsl.sh
source .venv-wsl/bin/activate
```

### 2. Configure runtime settings

Keep `config.json` as the public-safe base template.

Then create a local override file for secrets:

```bash
cp config.local.example.json config.local.json
```

`config.local.json` is ignored by git and automatically merged on top of `config.json` at runtime.

At minimum, check:

- cloud LLM settings used during compile
- embedding model settings
- rerank settings

### 3. Compile memory files

```bash
python compile.py memory
```

This writes canonical chunks into SQLite and rebuilds the High-search indexes.

### 4. Query the memory system

```bash
python consume.py "你的查询"
```

Or inspect the raw routed payload:

```bash
python consume.py --json --backend hybrid "你的查询"
```

### 5. Run the review service

```bash
./launch_memory_service.sh
```

## Example Commands

Compile a single file:

```bash
python compile.py memory/2026-04-23.md
```

Search with broader recall before rerank:

```bash
python consume.py --json --backend hybrid --top-k 5 --recall-pool-size 12 "review-pack 的输出结构是什么"
```

Inspect pending delete review queue:

```bash
python consume.py --show-pending-delete
```

## Current Status

This is still a working prototype, not a polished product.

What already exists:

- markdown-to-chunk compile flow
- canonical chunk storage in SQLite
- vector + FTS + hybrid retrieval
- rerank-based High routing
- evaluation assets and experiment workspace under `.ai-tmp/`

What is still rough:

- configuration and secret management
- public-facing packaging
- reproducible setup beyond the current WSL-oriented workflow
- evaluation tooling cleanup and stronger separation between retrieval eval and routing eval

## Secret Handling

- Do not put real API keys into tracked `config.json`.
- Put local secrets in `config.local.json` instead.
- The loader reads `config.json` first and then overlays `config.local.json` from the same directory if it exists.

## Evaluation Notes

Most experiment assets live under `.ai-tmp/`.

Two different evaluation targets exist in this repo:

- first-stage retrieval quality: can the candidate pool pull the right chunk?
- High admission quality: should the live corpus allow a direct answer at all?

Those should not be treated as the same problem.

## Why This Repo May Be Useful

If you care about AI coding systems that do more than plain RAG, this repo may be useful as a small, opinionated reference for:

- memory compilation from messy working notes
- direct-answer-aware reranking
- separating retrieval success from answer-admission success
- experimenting with noisy-chunk rejection instead of over-answering

## License

No license file has been added yet.
