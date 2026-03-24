---
name: eval-registry
description: Ingests and queries the SQLite eval registry (data/eval_registry.db) from meta.json and report.json under artifacts/eval_runs; compares run groups with compare_eval_configs.py; backfill, hybrid_enabled, t-tests. Use when the user mentions eval registry, ingest eval runs, SQLite eval history, eval_registry.db, query registry, compare eval configs, or statistical comparison of eval runs.
---

# Eval registry skill

## Purpose

The registry is a **queryable index** over timestamped eval folders. Full artifacts remain under `artifacts/eval_runs/<UTC>/` (`meta.json`, `<dataset_key>/report.json`).

- **Default DB:** `data/eval_registry.db` (gitignored).
- **Ingest:** `scripts/ingest_eval_registry.py`
- **Query:** `scripts/query_eval_registry.py`
- **Stats (t-test):** `scripts/compare_eval_configs.py` (needs **scipy** — `pip install -e '.[dev]'`)

## Commands (use repo venv)

From repo root:

```bash
.venv/bin/python scripts/ingest_eval_registry.py --all
.venv/bin/python scripts/ingest_eval_registry.py artifacts/eval_runs/<UTC>/
.venv/bin/python scripts/query_eval_registry.py last -n 10
.venv/bin/python scripts/compare_eval_configs.py --dataset single --metric Faithfulness --group-a RUN1,RUN2 --group-b RUN3,RUN4
```

## Documentation

- User-facing: [docs/eval-registry.md](docs/eval-registry.md)
- Variance / multi-run medians without DB: [scripts/summarize_eval_runs.py](scripts/summarize_eval_runs.py), [docs/determinism-2026-03-24.md](docs/determinism-2026-03-24.md)

## Rules for agents

- Do not treat a single run’s metrics as definitive; see determinism doc.
- `retrieval_features.hybrid_enabled` is written to new `meta.json` by `run_eval_report.py`; older runs rely on ingest parsing `configs/default.yaml` from `meta.config`.
- Fail-fast: missing `meta.json` or `report.json` for a dataset raises during ingest.
