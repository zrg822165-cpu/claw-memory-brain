# claw-memory-brain

Paused research prototype for direct-answer-aware memory retrieval in AI coding workflows.

This repository is an experimental codebase, not an actively maintained product. It is published mainly as a reference snapshot of the design, routing ideas, and implementation direction.

## What This Repo Is

`claw-memory-brain` explores a narrow but practical problem in memory-augmented assistants:

- retrieval may find the right topic
- but the returned chunk may still be too noisy for a direct answer

The project tries to separate those two questions.

Instead of stopping at "did retrieval hit something relevant", it also asks:

- should this chunk be admitted as a direct answer candidate?
- or should the system honestly refuse High-confidence answering?

## Current Status

- paused for now
- useful as a reference implementation or experiment skeleton
- not packaged as a polished library or service
- setup is still opinionated and WSL-first

If you are browsing this repo, the best expectation is: read the ideas, inspect the code, and adapt parts of it if useful.

## Core Idea

The rough pipeline is:

1. Write memory notes as markdown
2. Compile notes into canonical chunks
3. Store chunks and rebuild retrieval indexes
4. Retrieve with vector / FTS / hybrid search
5. Rerank candidates
6. Route the result:
   - `high`: current chunk is direct enough to support an answer
   - `miss`: topic may be relevant, but the chunk is still too noisy

There is also window-admission logic in the codebase, but this public snapshot should be read primarily as a retrieval-and-routing prototype.

## Why It May Still Be Interesting

Even in its current paused state, the repo may be useful if you care about:

- memory compilation from messy markdown notes
- separating retrieval quality from answer-admission quality
- rerank-based direct-answer gating
- rejecting noisy chunks instead of over-answering

## Repository Surface

- `compile.py`: compile markdown into canonical chunks and rebuild indexes
- `consume.py`: query entrypoint for retrieval and routing experiments
- `memory_service.py`: small service loop around pending-delete review tasks
- `compile_modules/`: chunking, filtering, indexing, reranking, routing, adjudication
- `config.json`: public-safe base configuration template
- `config.local.example.json`: example local override for secrets
- `memory/README.md`: notes on adding your own local memory data

This public repo intentionally does not include personal/project memory entries or heavy runtime data.

## Setup Notes

This repo is still WSL-first and assumes a hands-on workflow.

Typical local flow:

```bash
./bootstrap_wsl.sh
source .venv-wsl/bin/activate
cp config.local.example.json config.local.json
python compile.py memory
python consume.py "your query"
```

`config.local.json` is ignored by git and overlaid on top of `config.json` at runtime.

## Secret Handling

- keep tracked `config.json` public-safe
- put real local keys in `config.local.json`
- do not commit runtime data, personal memory content, or private experiment assets

## Dependencies

The included requirements files reflect the prototype state of the project.

- `requirements-wsl.txt` is the lighter starting point
- `requirements.txt` contains the fuller experiment-oriented dependency set

Some dependencies are tied to the original experimentation environment, so you should expect a bit of manual adjustment.

Practical reading of the dependency split:

- if you only want to inspect the retrieval/routing prototype, start from `requirements-wsl.txt`
- if you want to reproduce more of the original experiment surface, inspect `requirements.txt`
- graph and clustering related packages should be treated as prototype-era extras unless you specifically need those paths

## Evaluation Caveat

This project historically used more than one evaluation target.

Keep these separate:

- first-stage retrieval quality: did the candidate pool pull the right chunk?
- answer admission quality: should the live chunk be allowed to answer directly?

Those are related, but they are not the same metric.

## Maintenance Note

This repository is intentionally slimmed down for public sharing.

What is missing on purpose:

- original personal memory corpus
- internal experiment workspace
- runtime indexes and databases
- previously tracked secrets and historical sensitive content

## License

MIT. See `LICENSE`.
