# Eval tracking (MLflow)

The old SQLite eval registry has been removed. Evaluation now writes directly to **MLflow** using `mlflow.genai.evaluate()`, native DeepEval scorers, and traced RAG spans.

- **Tracking DB:** `data/mlflow.db` (gitignored)
- **Run UI:** `mlflow ui --backend-store-uri sqlite:///data/mlflow.db`
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

Each run logs:

- root eval run metadata and tags
- `pass_rate` plus per-metric pass rates
- traced `RETRIEVER` spans containing retrieved context
- traced `LLM` spans for generation
- scorer feedback from MLflow DeepEval scorers

## A/B study

Fixed-N paired hybrid-vs-dense example:

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

The script writes `artifacts/ab_study/<study_id>/study_meta.json` and `manifest.json`, but the source of truth is MLflow. Filter study runs in the UI or with `mlflow.search_runs()` using `tags.study_id`.

## Verification

To confirm retrieval context wiring after a run:

1. Open the MLflow trace for a sample.
2. Check for a top-level `RETRIEVER` span.
3. Confirm its outputs contain chunk payloads with `page_content`.
4. Confirm scorer results are present on the same eval run.

## Cursor skills (agents)

- [`.cursor/skills/eval-registry/SKILL.md`](../.cursor/skills/eval-registry/SKILL.md) — MLflow UI, DB, tags
- [`.cursor/skills/run-eval-report/SKILL.md`](../.cursor/skills/run-eval-report/SKILL.md) — run `scripts/run_eval.py`
- [`.cursor/skills/rag-eval-runs/SKILL.md`](../.cursor/skills/rag-eval-runs/SKILL.md) — interpret metrics and traces
- [`.cursor/skills/science-loop/SKILL.md`](../.cursor/skills/science-loop/SKILL.md) — `scripts/run_ab_study.py`, experiment rigor

Architecture diagram including these paths: [docs/ARCHITECTURE-2026-03-25T103050Z.md](ARCHITECTURE-2026-03-25T103050Z.md) (section *Cursor skills and commands*).
