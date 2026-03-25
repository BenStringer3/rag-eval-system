# MLflow eval tracking

The old SQLite eval registry has been removed. Evaluation writes directly to **MLflow** using `mlflow.genai.evaluate()`, DeepEval scorers, and traced RAG spans.

- **Tracking DB (default):** `data/mlflow.db` (gitignored); URI is set in `configs/eval.yaml` → `tracking.uri`.
- **UI:** `mlflow ui --backend-store-uri sqlite:///data/mlflow.db` (match URI to your config).
- **Primary script:** `scripts/run_eval.py`
- **A/B script:** `scripts/run_ab_study.py`

## Standalone eval

Single dataset:

```bash
.venv/bin/python scripts/run_eval.py --dataset data/eval_datasets/starter.json
```

All enabled datasets from `configs/eval.yaml`:

```bash
.venv/bin/python scripts/run_eval.py
```

Optional re-ingest before eval:

```bash
.venv/bin/python scripts/run_eval.py --dataset data/eval_datasets/synthetic.json --ingest
```

Each run logs root metadata and tags, `pass_rate` and per-metric pass rates, traced `RETRIEVER` / `LLM` spans, and scorer feedback.

## A/B study

Fixed-N paired example:

```bash
.venv/bin/python scripts/run_ab_study.py \
  --dataset data/eval_datasets/synthetic.json \
  --arm-a-label "dense" \
  --arm-b-label "hybrid" \
  --arm-a-overrides-json '{"retrieval.hybrid.enabled": false}' \
  --arm-b-overrides-json '{"retrieval.hybrid.enabled": true}' \
  --pairing paired \
  --n-per-arm 3
```

The script writes `artifacts/ab_study/<study_id>/` (configs + sidecar JSON) for reproducibility; **metrics and run lists** live in MLflow. Filter by `tags.study_id` (see [.cursor/skills/mlflow/SKILL.md](../.cursor/skills/mlflow/SKILL.md)).

## Verification

After a run, confirm retrieval wiring in the MLflow trace: top-level `RETRIEVER` span, chunk payloads with `page_content`, scorer results on the same eval run.

## Cursor commands (agents)

- [`.cursor/commands/run-eval-report.md`](../.cursor/commands/run-eval-report.md) → [run-eval-report skill](../.cursor/skills/run-eval-report/SKILL.md)
- [`.cursor/commands/analyze-eval-run.md`](../.cursor/commands/analyze-eval-run.md) → [rag-eval-runs skill](../.cursor/skills/rag-eval-runs/SKILL.md)

## Cursor skills (agents)

- [`.cursor/skills/mlflow-eval/SKILL.md`](../.cursor/skills/mlflow-eval/SKILL.md) — run `run_eval` / `run_ab_study`, launch UI
- [`.cursor/skills/mlflow/SKILL.md`](../.cursor/skills/mlflow/SKILL.md) — query `data/mlflow.db` (SQL, `search_runs`), study filters
- [`.cursor/skills/run-eval-report/SKILL.md`](../.cursor/skills/run-eval-report/SKILL.md) — long-running `run_eval.py` expectations
- [`.cursor/skills/rag-eval-runs/SKILL.md`](../.cursor/skills/rag-eval-runs/SKILL.md) — interpret metrics and traces
- [`.cursor/skills/science-loop/SKILL.md`](../.cursor/skills/science-loop/SKILL.md) — hypothesis-driven A/B workflow

Architecture: [docs/ARCHITECTURE-2026-03-25T103050Z.md](ARCHITECTURE-2026-03-25T103050Z.md) (*Cursor skills and commands*).
