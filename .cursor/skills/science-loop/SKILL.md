---
name: science-loop
description: Eval "science loop" in rag-eval-system — run hypothesis-driven experiments with `scripts/run_ab_study.py` or repeated `scripts/run_eval.py` runs. Source run lists, tags, and metrics from MLflow (sqlite:///data/mlflow.db); use .cursor/skills/mlflow/SKILL.md for SQL/API queries. Use when the user asks to test a hypothesis with repeated eval runs, run an A/B study, or compare two retrieval/generation configurations.
---

# Science loop skill

## Purpose

Turn a vague “I think X is better than Y” into a **testable hypothesis**, then gather enough repeated eval evidence to **support or refute** it with scientific rigor.

This skill is **experiment-agnostic**: the “treatment” can be *any* configuration change you can express as a temporary config override (retrieval, reranking, prompts, generation kwargs, chunking, etc.). The default built-in orchestrator currently supports a **2‑arm study** where the intervention is “hybrid retrieval on/off”, but the workflow/general rigor rules apply to future experiments too.

## Prerequisites

- Repo root; **`.venv/bin/python`** for all scripts.
- Cloud judge API key in env (same as `run_eval.py`).
- **One dataset per study** (avoid mixing datasets in a single hypothesis test): pass `--dataset …` or ensure only one dataset is enabled in `configs/eval.yaml`.

## Wall time

Each replicate is a full `run_eval.py`-equivalent evaluation. A 2-arm fixed-N study with `N` per arm is **2N** eval runs.

## On invocation (agent behavior)

Before running anything, ask clarifying questions until the experiment is unambiguous and “preregistered” (even if informally).

### Clarifying questions (minimum set)

- **Hypothesis**: What exactly is the claim? Directional (A > B) or non-directional (A ≠ B)?
- **Arms**: What are the *control* and the *treatment* configs? What changes between them (single change preferred)?
- **Primary metric**: Which metric is the *one* decision-driving metric (e.g. `pass_rate`, `faithfulness`, etc.)?
- **Success criterion**: What constitutes “win”? (effect size threshold and/or p-value threshold; specify one primary test)
- **Dataset & scope**: Which dataset path? Any filters? Any expectation the effect differs by subset?
- **Pairing / blocking**: Should runs be paired/block-randomized to reduce variance (recommended when feasible)?
- **Stopping rule**: Fixed‑N (preferred) or exploratory sequential? If sequential, what max N and what honest labeling?
- **Operational constraints**: Max wall time, max judge spend/rate limits, max concurrency, retries policy.

### Rigor rules (do not skip)

- **One primary hypothesis + one primary metric** per study; label everything else exploratory.
- **Fixed‑N preferred** for interpretable p-values. If doing sequential peeking, label “exploratory” and avoid overclaiming.
- **Keep everything else constant**: dataset, corpus/index, prompts, temperature, model, judge config, and seed/determinism settings.
- **Record the configuration deltas** used for each arm (temporary overrides; do not silently mutate repo defaults).
- **Handle failures honestly**: failed/incomplete runs don’t “count”; retry policy must be symmetric across arms.

## Commands

Choose the runner based on whether the hypothesis can be expressed with the existing 2‑arm orchestrator.

### Path A — use the built-in 2‑arm orchestrator (current: hybrid on/off)

Use this when the treatment/control differ by a config override that can be expressed as dotted-key YAML changes. The orchestrator still writes a small **reproducibility** tree under `artifacts/ab_study/<study_id>/` (`study_meta.json`, `manifest.json`, per-arm YAML under `configs/`). That is *not* the old `artifacts/eval_runs/` report layout (retired in favor of MLflow). For **which runs exist, tags, metrics, and comparisons**, query **MLflow** via the Tracking API or SQL on `data/mlflow.db` — follow [`.cursor/skills/mlflow/SKILL.md`](../mlflow/SKILL.md).

#### Fixed‑N (preferred for interpretable p-values)

```bash
.venv/bin/python scripts/run_ab_study.py \
  --dataset data/eval_datasets/synthetic.json \
  --arm-a-label "dense (hybrid off)" --arm-b-label "hybrid on" \
  --arm-a-overrides-json '{"retrieval.hybrid.enabled": false}' \
  --arm-b-overrides-json '{"retrieval.hybrid.enabled": true}' \
  --pairing paired --n-per-arm 5
```

- `--pairing paired`: each block = dense run then hybrid run.
- `--pairing none`: all A runs then all B runs.
- Inspect results in MLflow by filtering `tags.study_id`.

### Path B — experiment not supported by the orchestrator yet (generic workflow)

If the hypothesis is about *any other* config change, run repeated evals yourself (or extend the orchestrator later). The core loop is:

- Create **two temporary config variants** (control/treatment) without changing repo defaults.
- Run enough replicates per arm via `scripts/run_eval.py` or `scripts/run_ab_study.py`.
- Compare arms by querying MLflow (see [`.cursor/skills/mlflow/SKILL.md`](../mlflow/SKILL.md)); do not rely on artifact JSON for metric aggregates.
- Produce a short written report with the primary result and any caveats about judge variance.

### Failures and judge limits (applies to all paths)

- Treat failed eval runs symmetrically across arms.
- Use MLflow traces to confirm retrieval context and scorer wiring before trusting results.

## Related

- **Querying runs/metrics/tags (SQL or API):** [`.cursor/skills/mlflow/SKILL.md`](../mlflow/SKILL.md)
- Run evals / UI: [`.cursor/skills/mlflow-eval/SKILL.md`](../mlflow-eval/SKILL.md) · Query runs: [`.cursor/skills/mlflow/SKILL.md`](../mlflow/SKILL.md) · [docs/mlflow-eval-tracking.md](../../../docs/mlflow-eval-tracking.md).
- Single eval runs: [`.cursor/skills/run-eval-report/SKILL.md`](../run-eval-report/SKILL.md).
- Deep-dive one run (failures, traces): [`.cursor/skills/rag-eval-runs/SKILL.md`](../rag-eval-runs/SKILL.md) · [`.cursor/commands/analyze-eval-run.md`](../../commands/analyze-eval-run.md).
- Variance: [docs/determinism-2026-03-24.md](../../../docs/determinism-2026-03-24.md).

## Rules for agents

- **Treat MLflow as the data plane** for post-hoc analysis: list runs, read `pass_rate` and other metrics, and filter by `study_id` / `arm` / dataset tags using the [mlflow skill](../mlflow/SKILL.md) (SQLite or `mlflow.search_runs`). Use `artifacts/ab_study/` only for the orchestrator’s saved configs and sidecar metadata (still emitted by `run_ab_study.py`); do not use it as the metrics store, and do not point people at `artifacts/eval_runs/` for eval output.
- Ask clarifying questions first; do not start running evals until the hypothesis, arms, primary metric, and stopping rule are specified.
- Prefer **fixed‑N** when the user wants “proof”; label **exploratory** when doing any sequential peeking.
- Keep the experiment to **one primary comparison**; if multiple metrics are reported, explicitly mark non-primary metrics exploratory.
- Poll long runs; do not assume hang after a short idle.
- When using the orchestrator, capture the printed `study_id` (and the `mlflow_filter=tags.study_id = '…'` line) and inspect runs with that filter so studies do not get mixed in the UI or in `search_runs`. Optionally pass `--study-id` to `scripts/run_ab_study.py` for a human-readable id. Query patterns: [`.cursor/skills/mlflow/SKILL.md`](../mlflow/SKILL.md).
