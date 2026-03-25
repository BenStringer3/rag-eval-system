---
name: mlflow
description: Query the local MLflow tracking store for rag-eval-system runs, metrics, and tags. Use sqlite3 or the MLflow Tracking API; URI from configs/eval.yaml tracking (often sqlite:///data/mlflow.db). Triggers mlflow sql, mlflow database, search runs, study_id, pass_rate, latest_metrics, trace_info.
---

# MLflow tracking store (query)

## Purpose

**This skill:** structured read access to eval history (SQL, `mlflow.search_runs`, tag/metric names).

**Not this skill:** executing `run_eval.py` / `run_ab_study.py` or starting the UI — use [`.cursor/skills/mlflow-eval/SKILL.md`](../mlflow-eval/SKILL.md).

Default local URI is `sqlite:///data/mlflow.db` (file `data/mlflow.db` from repo root). **Always align with `configs/eval.yaml` → `tracking.uri`** if the project uses another backend.

Official schema is owned by MLflow; see [MLflow tracking storage](https://mlflow.org/docs/latest/tracking.html#storage).

## When to use what

| Goal | Prefer |
|------|--------|
| Ad-hoc SQL, joins, exports | `sqlite3 data/mlflow.db` (or any SQLite client) |
| Filter runs in Python, stable API | `mlflow.set_tracking_uri(...)` then `mlflow.search_runs()` |
| Traces / spans | MLflow UI or Trace API; metadata also in `trace_info` / `trace_tags` |

Do **not** treat `artifacts/ab_study/...` JSON as the source of truth for **metrics or run lists** — sidecar files only. **Tags and metrics live in MLflow.**

## Filtering A/B study runs

`run_ab_study.py` prints `study_id=...` and `mlflow_filter=tags.study_id = '…'`. Use that in the UI or in `search_runs` so you do not mix studies. Orchestration options (`--study-id`, pairing, `--ingest`): [`.cursor/skills/science-loop/SKILL.md`](../science-loop/SKILL.md).

## Core tables (typical MLflow 2.x SQLite)

- **`experiments`** — `experiment_id`, `name`, `lifecycle_stage`
- **`runs`** — `run_uuid` (PK), `name`, `experiment_id`, `status`, `start_time`, `end_time`, `lifecycle_stage`
- **`tags`** — `key`, `value`, `run_uuid` (one row per tag per run)
- **`params`** — `key`, `value`, `run_uuid`
- **`latest_metrics`** — latest value per `(run_uuid, key)`; use this for current metric snapshot
- **`metrics`** — full time series of logged metrics (multiple rows per key if re-logged)
- **`trace_info`**, **`trace_tags`** — request-level traces (GenAI eval tracing)

Filter active runs with `runs.lifecycle_stage = 'active'`. Completed evals usually have `runs.status = 'FINISHED'`.

## Repo-specific tags and metrics

Logged by `scripts/run_eval.py` and `scripts/run_ab_study.py` (keys may grow over time):

**Tags (examples):** `study_id`, `arm`, `arm_label`, `block`, `pairing`, `eval.kind`, `eval.dataset_key`, `eval.dataset_name`, `eval.dataset_path`, `rag.config_path`, `rag.retrieval.hybrid_enabled`, …

**Metrics (examples):** `pass_rate`, `faithfulness/mean`, `faithfulness_pass_rate`, `answer_relevancy/mean`, …

## SQL recipes

Run from repo root. Adjust `experiment_id` after checking `SELECT experiment_id, name FROM experiments WHERE lifecycle_stage = 'active';`.

### List experiments

```sql
SELECT experiment_id, name, lifecycle_stage FROM experiments;
```

### Recent runs with `pass_rate`

```sql
SELECT r.run_uuid, r.name, r.status, r.start_time, m.value AS pass_rate
FROM runs r
JOIN latest_metrics m ON r.run_uuid = m.run_uuid AND m.key = 'pass_rate'
WHERE r.lifecycle_stage = 'active'
ORDER BY r.start_time DESC
LIMIT 20;
```

### All runs for an A/B study (`study_id` tag)

```sql
SELECT r.run_uuid, r.name, r.start_time
FROM runs r
JOIN tags t ON r.run_uuid = t.run_uuid AND t.key = 'study_id'
WHERE t.value = 'YOUR_STUDY_ID' AND r.lifecycle_stage = 'active'
ORDER BY r.start_time;
```

### Study arms with `pass_rate` (pivot tags inline)

```sql
SELECT r.run_uuid,
       (SELECT value FROM tags WHERE run_uuid = r.run_uuid AND key = 'arm') AS arm,
       (SELECT value FROM tags WHERE run_uuid = r.run_uuid AND key = 'block') AS block,
       m.value AS pass_rate
FROM runs r
JOIN tags sid ON r.run_uuid = sid.run_uuid AND sid.key = 'study_id' AND sid.value = 'YOUR_STUDY_ID'
JOIN latest_metrics m ON r.run_uuid = m.run_uuid AND m.key = 'pass_rate'
WHERE r.lifecycle_stage = 'active'
ORDER BY r.start_time, arm, block;
```

### Runs for one dataset (`eval.dataset_key`)

```sql
SELECT r.run_uuid, r.name, m.value AS pass_rate
FROM runs r
JOIN tags dk ON r.run_uuid = dk.run_uuid AND dk.key = 'eval.dataset_key' AND dk.value = 'synthetic'
JOIN latest_metrics m ON r.run_uuid = m.run_uuid AND m.key = 'pass_rate'
WHERE r.lifecycle_stage = 'active'
ORDER BY r.start_time DESC;
```

### Distinct tag keys present in the DB

```sql
SELECT key, COUNT(*) AS n FROM tags GROUP BY key ORDER BY key;
```

## Python equivalent (Tracking API)

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///data/mlflow.db")
df = mlflow.search_runs(
    experiment_names=["rag-eval-system"],
    filter_string="tags.eval.kind = 'ab_study' and tags.study_id = 'YOUR_STUDY_ID'",
    order_by=["start_time DESC"],
    output_format="pandas",
)
# Metrics appear as columns metrics.pass_rate, metrics.faithfulness/mean, etc.
```

## Related

- **Run evals and UI:** [`.cursor/skills/mlflow-eval/SKILL.md`](../mlflow-eval/SKILL.md)
- **Explain metrics / traces / failures:** [`.cursor/skills/rag-eval-runs/SKILL.md`](../rag-eval-runs/SKILL.md) (often invoked via [`.cursor/commands/analyze-eval-run.md`](../../commands/analyze-eval-run.md))
- **Experiment design:** [`.cursor/skills/science-loop/SKILL.md`](../science-loop/SKILL.md)
