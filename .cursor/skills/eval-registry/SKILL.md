---
name: eval-registry
description: Use MLflow as the eval tracking backend in rag-eval-system. Run `scripts/run_eval.py`, inspect `data/mlflow.db`, start `mlflow ui`, and filter runs by tags such as `study_id`, dataset, or retrieval mode. Triggers mlflow eval, mlflow ui, eval tracking, run_eval.py, data/mlflow.db.
---

# Eval tracking skill

## Purpose

Evaluation now logs directly to **MLflow**. There is no ingest/query registry layer anymore.

- **Tracking DB:** `data/mlflow.db` (gitignored)
- **Standalone eval:** `scripts/run_eval.py`
- **A/B study:** `scripts/run_ab_study.py`
- **UI:** `mlflow ui --backend-store-uri sqlite:///data/mlflow.db`

## Commands (use repo venv)

From repo root:

```bash
.venv/bin/python scripts/run_eval.py --dataset data/eval_datasets/starter.json
.venv/bin/python scripts/run_ab_study.py \
  --dataset data/eval_datasets/synthetic.json \
  --arm-a-overrides-json '{"retrieval.hybrid.enabled": false}' \
  --arm-b-overrides-json '{"retrieval.hybrid.enabled": true}' \
  --n-per-arm 3
mlflow ui --backend-store-uri sqlite:///data/mlflow.db
```

## Documentation

- User-facing: [docs/eval-registry.md](docs/eval-registry.md)
- Study orchestration: [`.cursor/skills/science-loop/SKILL.md`](../science-loop/SKILL.md)

## Rules for agents

- Do not treat a single run’s metrics as definitive; see determinism doc.
- Use MLflow tags instead of rebuilding local registries.
- Filter A/B studies by `tags.study_id`.
- Retrieval context verification means checking traced `RETRIEVER` spans, not reading `report.json`.
