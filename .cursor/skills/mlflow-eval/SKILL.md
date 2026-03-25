---
name: mlflow-eval
description: Run rag-eval-system evals logged to MLflow (`scripts/run_eval.py`, `scripts/run_ab_study.py`) and launch the MLflow UI. Does not cover querying the tracking store — use the `mlflow` skill for SQL / `search_runs`. Triggers run_eval.py, run_ab_study.py, mlflow ui, eval tracking.
---

# MLflow eval runs (execute + UI)

## Purpose

**This skill:** start evaluations and open the MLflow UI.

**Not this skill:** listing runs, metrics, or SQL against the store — use [`.cursor/skills/mlflow/SKILL.md`](../mlflow/SKILL.md).

Tracking URI and experiment name come from `configs/eval.yaml` under `tracking` (default file DB: `sqlite:///data/mlflow.db` → `data/mlflow.db` at repo root). Confirm the active URI there before assuming paths.

## Commands (repo venv, from repo root)

```bash
.venv/bin/python scripts/run_eval.py --dataset data/eval_datasets/starter.json
.venv/bin/python scripts/run_ab_study.py \
  --dataset data/eval_datasets/synthetic.json \
  --arm-a-overrides-json '{"retrieval.hybrid.enabled": false}' \
  --arm-b-overrides-json '{"retrieval.hybrid.enabled": true}' \
  --n-per-arm 3
mlflow ui --backend-store-uri sqlite:///data/mlflow.db
```

More variants (all datasets, `--ingest`, A/B labels): [docs/mlflow-eval-tracking.md](../../../docs/mlflow-eval-tracking.md).

## Related

- **Query runs / metrics / tags:** [`.cursor/skills/mlflow/SKILL.md`](../mlflow/SKILL.md)
- **Interpret finished runs (metrics, traces, triage):** [`.cursor/skills/rag-eval-runs/SKILL.md`](../rag-eval-runs/SKILL.md) · [`.cursor/commands/analyze-eval-run.md`](../../commands/analyze-eval-run.md)
- **A/B design and rigor:** [`.cursor/skills/science-loop/SKILL.md`](../science-loop/SKILL.md)
- **Long `run_eval.py` runs (wall time, polling):** [`.cursor/skills/run-eval-report/SKILL.md`](../run-eval-report/SKILL.md)

## Rules for agents

- Do not treat a single run’s metrics as definitive; see [docs/determinism-2026-03-24.md](../../../docs/determinism-2026-03-24.md).
- Use MLflow tags for grouping; do not rebuild local registries.
- Filter A/B studies by `tags.study_id` (see [mlflow skill](../mlflow/SKILL.md) for filters and SQL).
- Verify retrieval via traced `RETRIEVER` spans in MLflow, not legacy `report.json` layouts.
