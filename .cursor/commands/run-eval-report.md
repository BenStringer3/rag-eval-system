# Run eval (`scripts/run_eval.py`)

**Follow the skill:** read and apply [`.cursor/skills/run-eval-report/SKILL.md`](../skills/run-eval-report/SKILL.md). It covers wall-clock expectations, venv usage, and what to report from MLflow.

## What to do

1. Work from the repo root. Run Python as **`.venv/bin/python`** (do not rely on `source .venv/bin/activate` in automation).
2. **Dataset choice** (user preference or ask once):
   - **Faster / smaller:** `data/eval_datasets/starter.json`
   - **Larger / slower:** `data/eval_datasets/synthetic.json`
   - **All enabled in config:** omit `--dataset` (see `configs/eval.yaml`).
3. **Expect long wall clock:** many minutes for starter; much longer for synthetic. Use generous timeouts or run in background and monitor output; stalled-looking progress is often normal.
4. After completion, report **MLflow run id(s)**, dataset key(s), `pass_rate` (and per-metric pass rates if useful), and that traces show `RETRIEVER` / `LLM` spans. Point the user at `mlflow ui --backend-store-uri sqlite:///data/mlflow.db` if they want the UI.

## Minimal invocation examples

```bash
.venv/bin/python scripts/run_eval.py --dataset data/eval_datasets/starter.json
```

```bash
.venv/bin/python scripts/run_eval.py --dataset data/eval_datasets/synthetic.json
```

Optional re-ingest before eval:

```bash
.venv/bin/python scripts/run_eval.py --dataset data/eval_datasets/starter.json --ingest --corpus-dir data/corpus
```

## Related

- **A/B studies:** [`.cursor/skills/science-loop/SKILL.md`](../skills/science-loop/SKILL.md) → `scripts/run_ab_study.py`
- **MLflow (run + UI):** [`.cursor/skills/mlflow-eval/SKILL.md`](../skills/mlflow-eval/SKILL.md) · **Query store:** [`.cursor/skills/mlflow/SKILL.md`](../skills/mlflow/SKILL.md)
- **Interpret results:** [`.cursor/commands/analyze-eval-run.md`](../commands/analyze-eval-run.md) → `rag-eval-runs` skill
