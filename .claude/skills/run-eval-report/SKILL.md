---
name: run-eval-report
description: Runs scripts/run_eval.py end-to-end in rag-eval-system via .venv/bin/python, dataset selection (starter vs synthetic), and MLflow-backed output inspection. Use when the user asks to run evals or benchmark datasets.
---

# Run eval skill

## Purpose

Execute `scripts/run_eval.py` in `rag-eval-system` with explicit dataset choice:

- `data/eval_datasets/starter.json` — fewer samples; still **many minutes** (cloud judge + per-sample metrics).
- `data/eval_datasets/synthetic.json` — larger; expect **much longer** (tens of minutes to hours depending on size and throttling).

## Runtime expectations (for agents)

- These runs are **not quick**: each sample triggers multiple judge-backed metrics; progress may sit on one test for a long time.
- **Expected wall time:** starter is often **15–25 minutes**; synthetic is longer (tens of minutes to hours). Treat that range as normal, not a hang.
- **Check-ins:** at least **every ~1 minute** while waiting—e.g. confirm the process is still running, tail recent log/terminal output, or peek at CPU; do **not** assume failure after ~60s of quiet spinner, but do **not** go 10+ minutes blind either.
- You may run in background and poll the same way; keep the run alive until it exits or you have evidence of a real failure.
- If the environment’s `source .venv/bin/activate` misbehaves, **still use `.venv/bin/python`** (see below).

## Prerequisites

1. Working directory: repo root (e.g. `/home/ben/repos/rag-eval-system`).
2. **Canonical Python:** use the venv interpreter directly (reliable in automation and when `activate` fails):

```bash
.venv/bin/python scripts/run_eval.py --help
```

3. Optional: `source .venv/bin/activate` then `python ...` only if activation works in your shell; otherwise always use `.venv/bin/python`.
4. If `.venv` is missing, stop and tell the user to create it (e.g. per README); do not invent a fallback interpreter.
5. Judge credentials: ensure the env var named by `configs/eval.yaml` `judge.openai.api_key_env` is set.

## Standard commands

Replace `...` with the same args as below; prefix every invocation with `.venv/bin/python`.

### Fast-er run (starter dataset)

```bash
.venv/bin/python scripts/run_eval.py --dataset data/eval_datasets/starter.json
```

### Fuller run (synthetic dataset)

```bash
.venv/bin/python scripts/run_eval.py --dataset data/eval_datasets/synthetic.json
```

### Multi-dataset run from config

```bash
.venv/bin/python scripts/run_eval.py
```

Runs all **enabled** datasets from `configs/eval.yaml`.

## Useful optional flags

- `--ingest --corpus-dir data/corpus`: ingest before evaluating.

Conservative cloud judge example (slower wall clock, fewer 429s):

```bash
.venv/bin/python scripts/run_eval.py \
  --dataset data/eval_datasets/starter.json \
  --ingest
```

## What to report back

After a run completes:

1. MLflow run id(s).
2. Dataset key(s) evaluated.
3. Logged `pass_rate` and any per-metric pass-rate metrics.
4. Whether the traces include top-level `RETRIEVER` spans.

## Troubleshooting checklist

- Wrong interpreter → use `.venv/bin/python`, not system `python`, unless you know the venv is active and has deps (system `python` often lacks `pydantic` / project deps).
- Dataset missing → verify paths under `data/eval_datasets/`.
- Judge / API → `configs/eval.yaml` `judge` + env.
- MLflow DB → confirm `tracking.uri` points at a writable SQLite path.
- Missing retrieval context → inspect the trace and confirm a top-level `RETRIEVER` span exists.
