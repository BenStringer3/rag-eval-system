---
name: science-loop
description: Eval "science loop" in rag-eval-system — run hypothesis-driven experiments by repeatedly running `run_eval_report.py` under controlled config changes, ingesting results into the eval registry, and producing statistically honest comparisons (fixed-N preferred; exploratory optional but clearly labeled). Use when the user asks to test a hypothesis with repeated eval runs, run an A/B study, compare two retrieval/generation configurations, or produce a rigorous report with plots.
---

# Science loop skill

## Purpose

Turn a vague “I think X is better than Y” into a **testable hypothesis**, then gather enough repeated eval evidence to **support or refute** it with scientific rigor.

This skill is **experiment-agnostic**: the “treatment” can be *any* configuration change you can express as a temporary config override (retrieval, reranking, prompts, generation kwargs, chunking, etc.). The default built-in orchestrator currently supports a **2‑arm study** where the intervention is “hybrid retrieval on/off”, but the workflow/general rigor rules apply to future experiments too.

## Prerequisites

- Repo root; **`.venv/bin/python`** for all scripts.
- **`scipy`** for stats in the report: `pip install -e '.[dev]'`.
- Cloud judge API key in env (same as `run_eval_report`).
- **One dataset per study** (avoid mixing datasets in a single hypothesis test): pass `--dataset …` or ensure only one dataset is enabled in `configs/eval.yaml`.

## Wall time

Each replicate is a **full** `run_eval_report.py` pass. A 2‑arm fixed‑N study with `N` per arm is **2N** full evals — multiply by single-run duration (synthetic often **tens of minutes to hours**). Use `--max-concurrent 1` and throttle flags if the judge rate-limits.

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

Use this when the treatment/control differ by a config override that can be expressed as dotted-key YAML changes. Artifacts live under `artifacts/ab_study/<study_id>/` (`study_meta.json`, `manifest.jsonl`, `eval_runs/<UTC>/`). Successful runs are ingested into `data/eval_registry.db` unless `--no-ingest`.

#### Fixed‑N (preferred for interpretable p-values)

```bash
.venv/bin/python scripts/run_ab_study.py \
  --dataset data/eval_datasets/synthetic.json \
  --arm-a-label "dense (hybrid off)" --arm-b-label "hybrid on" \
  --arm-a-overrides-json '{"retrieval.hybrid.enabled": false}' \
  --arm-b-overrides-json '{"retrieval.hybrid.enabled": true}' \
  --pairing paired --n-per-arm 5 \
  --max-concurrent 1 --judge-throttle-seconds 5
```

- **`--pairing paired`:** each block = dense run then hybrid run (same corpus/index); report uses **paired** t-tests when blocks align.
- **`--pairing none`:** all dense runs, then all hybrid runs; **Welch** t-test in report.

#### Exploratory sequential (honest labeling: inflates Type I error)

```bash
.venv/bin/python scripts/run_ab_study.py \
  --dataset data/eval_datasets/starter.json \
  --arm-a-label "dense (hybrid off)" --arm-b-label "hybrid on" \
  --arm-a-overrides-json '{"retrieval.hybrid.enabled": false}' \
  --arm-b-overrides-json '{"retrieval.hybrid.enabled": true}' \
  --pairing paired --exploratory \
  --min-per-arm 2 --max-per-arm 10 --alpha 0.05 \
  --primary-metric pass_rate
```

#### After the study — report with figures

```bash
.venv/bin/python scripts/render_ab_study_report.py \
  --study-dir artifacts/ab_study/<study_id>
```

Writes `docs/ab-study-<study_id>.md` and Plotly HTML under `docs/_figures/ab-study-<study_id>/`.

### Path B — experiment not supported by the orchestrator yet (generic workflow)

If the hypothesis is about *any other* config change, run repeated evals yourself (or extend the orchestrator later). The core loop is:

- Create **two temporary config variants** (control/treatment) without changing repo defaults.
- Run enough replicates per arm via `scripts/run_eval_report.py` (or the study runner you build later).
- Ingest results into `data/eval_registry.db`.
- Compare arms using the registry tooling; produce a short written report with the primary result + plots/tables.

Use the eval registry skill for ingest/query/comparisons: [`.cursor/skills/eval-registry/SKILL.md`](../eval-registry/SKILL.md).

### Failures and judge limits (applies to all paths)

- Subprocess **non-zero** or **incomplete** `meta.json` / `report.json` → treat that replicate as **failed**; **do not ingest** it into `data/eval_registry.db`. (In orchestrated studies, failures are recorded in the study `manifest.jsonl` with `status: failed`.)
- **`--retries N`** / **`--retry-backoff-seconds`:** retry the same arm (new UTC folder per attempt).
- **`LengthFinishReasonError`** (judge hits output token cap): logged to the run’s `judge_length_errors.jsonl`; eval aborts — not fixed by throttling alone; see `configs/eval.yaml` `generation_kwargs` and determinism doc. Do **not** use DeepEval `ignore_errors=True` for comparable A/B science.

## Related

- Registry ingest/query: [`.cursor/skills/eval-registry/SKILL.md`](../eval-registry/SKILL.md), [docs/eval-registry.md](docs/eval-registry.md) (Hybrid A/B section).
- Single eval runs: [`.cursor/skills/run-eval-report/SKILL.md`](../run-eval-report/SKILL.md).
- Variance: [docs/determinism-2026-03-24.md](docs/determinism-2026-03-24.md).

## Rules for agents

- Ask clarifying questions first; do not start running evals until the hypothesis, arms, primary metric, and stopping rule are specified.
- Prefer **fixed‑N** when the user wants “proof”; label **exploratory** when doing any sequential peeking.
- Keep the experiment to **one primary comparison**; if multiple metrics are reported, explicitly mark non-primary metrics exploratory.
- Poll long runs; do not assume hang after a short idle.
- When using the orchestrator, pass **`--study-dir`** to `render_ab_study_report.py` as the concrete `artifacts/ab_study/<id>` path printed at the end of the orchestrator.
