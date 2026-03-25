# Analyze RAG eval run (MLflow)

You interpret **completed** batch RAG evaluations from `scripts/run_eval.py` and from `scripts/run_ab_study.py` (each replicate/arm is its own MLflow run). The pre-MLflow **`artifacts/eval_runs/`** report tree is not used.

**Tracking:** metrics, tags, params, and GenAI traces live in the **MLflow backend** named in `configs/eval.yaml` → `tracking.uri` (and `tracking.experiment`). Align any `mlflow ui --backend-store-uri …` or `set_tracking_uri` call with that URI — do not hardcode a path unless you have confirmed the config.

**Primary skill — read and apply:** [`.cursor/skills/rag-eval-runs/SKILL.md`](../skills/rag-eval-runs/SKILL.md) for metric definitions, triage buckets, Mermaid templates, and repo touchpoints.

**Also use when needed:**

| Need | Skill / doc |
|------|-------------|
| `search_runs`, SQL on the SQLite store, tag filters (`study_id`, `eval.dataset_key`, …) | [`.cursor/skills/mlflow/SKILL.md`](../skills/mlflow/SKILL.md) |
| Open the MLflow UI with the right backend | [`.cursor/skills/mlflow-eval/SKILL.md`](../skills/mlflow-eval/SKILL.md), [docs/mlflow-eval-tracking.md](../../docs/mlflow-eval-tracking.md) |
| A/B study design, arms, pairing, `study_id` discipline | [`.cursor/skills/science-loop/SKILL.md`](../skills/science-loop/SKILL.md) |

## Workflow

1. Confirm `tracking.uri` / experiment from `configs/eval.yaml`, then locate the run(s) (UI or Tracking API) using tags; use the **mlflow** skill for filter strings and SQL.
2. Pull run-level metrics (`pass_rate`, aggregates from `mlflow.genai.evaluate()`), then open **traces** for sample-level scorer feedback and `RETRIEVER` / `LLM` spans.
3. Explain outcomes using the **Interpretation pattern** and **Troubleshooting quick reference** in **rag-eval-runs** (headline aggregates, per failing sample with rationale, root-cause bucket).
4. Add Mermaid only if the user asks — patterns live under **Diagrams** in **rag-eval-runs**.
5. Write a file only if the user requests it (e.g. a short note under `docs/`). Do not recreate `artifacts/eval_runs/` layouts.
