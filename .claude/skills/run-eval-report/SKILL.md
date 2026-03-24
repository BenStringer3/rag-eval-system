---
name: run-eval-report
description: Runs scripts/run_eval_report.py end-to-end in rag-eval-system via .venv/bin/python, dataset selection (starter vs synthetic), long-running DeepEval/judge expectations, and optional rate-limit-safe flags. Use when the user asks to run eval reports, benchmark datasets, or regenerate artifacts/eval_runs outputs.
---

# Run eval report skill

## Purpose

Execute `scripts/run_eval_report.py` in `rag-eval-system` with explicit dataset choice:

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
.venv/bin/python scripts/run_eval_report.py --help
```

3. Optional: `source .venv/bin/activate` then `python ...` only if activation works in your shell; otherwise always use `.venv/bin/python`.
4. If `.venv` is missing, stop and tell the user to create it (e.g. per README); do not invent a fallback interpreter.
5. Cloud judge: ensure `OPENCODE_ZEN_API_KEY` is set (see README). Shell automation often uses `set -a && source .env && set +a` from repo root before the command if keys live in `.env`.

## Standard commands

Replace `...` with the same args as below; prefix every invocation with `.venv/bin/python`.

### Fast-er run (starter dataset)

```bash
.venv/bin/python scripts/run_eval_report.py --dataset data/eval_datasets/starter.json
```

### Fuller run (synthetic dataset)

```bash
.venv/bin/python scripts/run_eval_report.py --dataset data/eval_datasets/synthetic.json
```

### Multi-dataset run from config

```bash
.venv/bin/python scripts/run_eval_report.py
```

Runs all **enabled** datasets from `configs/eval.yaml`.

## Useful optional flags

- `--max-concurrent <n>`: lower parallel judge traffic when rate-limited.
- `--judge-throttle-seconds <n>`: delay between judge tasks (increases wall time).
- `--per-task-timeout <seconds>`: raise if judges time out (default 600 in CLI).
- `--max-retries <n>`: full-run retries when DeepEval returns partial results or throws (default 2).
- `--retry-backoff-seconds <n>`: base sleep between retries, multiplied by attempt number (default 5).
- `--include-custom`: extra G-Eval judge calls.
- `--ingest --corpus-dir data/corpus`: ingest before evaluating.

Conservative cloud judge example (slower wall clock, fewer 429s):

```bash
.venv/bin/python scripts/run_eval_report.py \
  --dataset data/eval_datasets/starter.json \
  --max-concurrent 1 \
  --judge-throttle-seconds 5
```

## What to report back

After a run completes:

1. `artifacts/eval_runs/<UTC>/` path.
2. Per-dataset folder: `single/` (when `--dataset` is set), or `starter/` / `synthetic/` from config.
3. Files: `report.md`, `report.json`, `report.csv`, and run root `meta.json`.

## Troubleshooting checklist

- Wrong interpreter → use `.venv/bin/python`, not system `python`, unless you know the venv is active and has deps (system `python` often lacks `pydantic` / project deps).
- Dataset missing → verify paths under `data/eval_datasets/`.
- Judge / API → `configs/eval.yaml` `judge` + env (e.g. `OPENCODE_ZEN_API_KEY`).
- Rate limits / flaky judges → lower `--max-concurrent`, raise `--judge-throttle-seconds`, raise `--per-task-timeout`, and/or `--max-retries` + `--retry-backoff-seconds` for partial DeepEval returns.
- “Stuck” progress → often normal; wait for DeepEval completion or check provider latency before killing the process.
