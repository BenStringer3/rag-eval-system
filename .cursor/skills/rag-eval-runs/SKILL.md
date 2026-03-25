---
name: rag-eval-runs
description: Interprets MLflow-backed RAG evaluation runs produced by scripts/run_eval.py, explains DeepEval-derived metrics (faithfulness, answer relevancy, contextual precision/recall), and troubleshoots retrieval vs generation vs gold issues through MLflow traces and run metrics.
---

# RAG eval runs (MLflow)

**Cursor entrypoint:** [`.cursor/commands/analyze-eval-run.md`](../../commands/analyze-eval-run.md) — when that command is used, follow this skill end-to-end.

## Run layout

After `python scripts/run_eval.py`:

- **MLflow run** — root metrics like `pass_rate` and per-metric pass rates
- **MLflow trace** — root predict call plus child `RETRIEVER` and `LLM` spans
- **MLflow scorer feedback** — row-level rationales and scores attached to the run

Default behavior runs every enabled dataset in `configs/eval.yaml`. A one-off `--dataset` run logs tags with `eval.dataset_key=single`.

## Metric meanings (short)

- **Faithfulness** — Are claims in the answer entailed by **retrieved chunks**?
- **Answer relevancy** — Does the answer address the **user question** (not just related text)?
- **Contextual precision** — Are **relevant** chunks ranked **above** irrelevant ones? (signal-to-noise in top-$k$)
- **Contextual recall** — Do retrieved chunks **support the gold expected answer** (coverage of reference facts)?

## Interpretation pattern

1. Read run metrics and tags in MLflow.
2. Open sample traces with failing scorer feedback.
3. For each failure, map judge rationale to a bucket:
   - **Retrieval** — recall low, or precision low with irrelevant top chunks.
   - **Generation** — recall ok but faithfulness/relevancy low.
   - **Corpus / gold** — mixed unrelated documents in the index; gold assumes facts not present in any reasonable chunk.
   - **Judge / format** — JSON-wrapped outputs, nitpicks (“only”, spacing), or citation artifacts.

## Diagrams

When the user asks for a visual, use **Mermaid** (`flowchart` or `flowchart TB`):

- **Eval attachment:** `Query` → `Retriever` → `Chunks` → `Generator` → `Answer` → metrics split into “uses answer+chunks” vs “uses chunks+gold”.
- **Triage:** start from failed metric → one branch for retrieval fixes, one for generation, one for dataset/gold.

## Troubleshooting quick reference

| Signal | Likely cause | Levers |
|--------|----------------|--------|
| Low **contextual recall**, decent precision | Gold not in retrieved text | Corpus mix, chunk boundaries, `top_k`, query reformulation, separate indexes |
| Low **precision** | Noisy top-$k$ | `top_k`, chunk size, MMR/rerank (if added), cleaner corpus |
| Low **faithfulness**, high recall | Ungrounded or elaborated answer | System prompt, require “only from context”, reduce hallucination |
| Low **answer relevancy** | Off-topic or rambling answer | Same as generation; check for **structured JSON** confusing the judge |
| Inconsistent runs | Judge variance | Lower temperature, stronger judge model, repeat runs |

## Repo touchpoints

- **Script:** `scripts/run_eval.py`
- **Dataset registry:** `configs/eval.yaml` → `datasets`, resolved by `src/eval/eval_config.py`
- **Dataset adapter / scorers:** `src/eval/datasets.py`, `src/eval/scorers.py`
- **Pipeline:** `src/rag/pipeline.py`, `configs/default.yaml`, eval set under `data/eval_datasets/`

## Worked pattern (example class of failures)

- **Broad question (“What is this project about?”) + gold describing only RAG eval** but **retrieval returns another large topic** (e.g. streaming setup) → **contextual recall** collapses (gold facts absent from chunks); relevancy may pass weakly while the answer follows retrieved text (**faithfulness** can still be “ok” relative to wrong chunks). Fix is primarily **retrieval/corpus**, not the judge.

When explaining a specific run, cite **`sample_id`**, the **query**, and the relevant MLflow trace span or scorer rationale.

## Related

- **Find / filter runs (SQL, `search_runs`):** [`.cursor/skills/mlflow/SKILL.md`](../mlflow/SKILL.md)
- **Run new evals or start the UI:** [`.cursor/skills/mlflow-eval/SKILL.md`](../mlflow-eval/SKILL.md) · [docs/mlflow-eval-tracking.md](../../../docs/mlflow-eval-tracking.md)
- **A/B workflow and study rigor:** [`.cursor/skills/science-loop/SKILL.md`](../science-loop/SKILL.md)
