---
name: rag-eval-runs
description: Interprets RAG evaluation runs produced by scripts/run_eval_report.py (artifacts/eval_runs), explains DeepEval metrics (faithfulness, answer relevancy, contextual precision/recall), diagrams failure modes with Mermaid, and troubleshoots retrieval vs generation vs gold issues. Use when the user mentions eval runs, report.md, report.json, eval pass rate, DeepEval metrics, or analyzing artifacts/eval_runs. Reports live under per-dataset subfolders (e.g. starter/, synthetic/).
---

# RAG eval runs (DeepEval batch reports)

## Artifacts layout

After `python scripts/run_eval_report.py` (repo root, package installed):

- **`artifacts/eval_runs/<UTC>/meta.json`** — run metadata, thresholds, a **`datasets`** array (each entry: `key`, `path`, `pass_rate`, `mean_scores`), and **`eval_context`** (generation/judge model ids and temperatures). Aggregate several runs with **`scripts/summarize_eval_runs.py`** (see `docs/determinism-2026-03-24.md`).
- **`artifacts/eval_runs/<UTC>/<dataset_key>/report.md`** — per-dataset readable breakdown with judge reasons (`dataset_key` matches keys under `configs/eval.yaml` → `datasets`, e.g. `starter`, `synthetic`; a one-off `--dataset` run uses that file’s basename).
- **`.../<dataset_key>/report.json`** — full `EvalReport` for that dataset.
- **`.../<dataset_key>/report.csv`** — one row per sample, metric columns.

Default behavior runs **every enabled** dataset listed in `configs/eval.yaml` (`enabled: false` skips one). Latest run: pick the **newest** timestamp folder under `artifacts/eval_runs/` if unspecified.

## Metric meanings (short)

- **Faithfulness** — Are claims in the answer entailed by **retrieved chunks**?
- **Answer relevancy** — Does the answer address the **user question** (not just related text)?
- **Contextual precision** — Are **relevant** chunks ranked **above** irrelevant ones? (signal-to-noise in top-$k$)
- **Contextual recall** — Do retrieved chunks **support the gold expected answer** (coverage of reference facts)?

## Interpretation pattern

1. Read **`meta.json`** for per-dataset aggregates (`datasets` array).
2. List samples where **`passed_all`** is false (from JSON or markdown ❌).
3. For each failure, map judge **reason** to a **bucket**:
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

- **Script:** `scripts/run_eval_report.py`
- **Dataset registry:** `configs/eval.yaml` → `datasets`, resolved by `src/eval/eval_config.py`
- **Runner / aggregation:** `src/eval/runner.py`, `src/eval/report.py`, `src/data/schemas.py` (`EvalReport`, `MetricScore`)
- **Metrics:** `src/eval/metrics.py`, thresholds from `run_evaluation` / CLI flags
- **Pipeline:** `src/rag/pipeline.py`, `configs/default.yaml`, eval set under `data/eval_datasets/`

## Worked pattern (example class of failures)

- **Broad question (“What is this project about?”) + gold describing only RAG eval** but **retrieval returns another large topic** (e.g. streaming setup) → **contextual recall** collapses (gold facts absent from chunks); relevancy may pass weakly while the answer follows retrieved text (**faithfulness** can still be “ok” relative to wrong chunks). Fix is primarily **retrieval/corpus**, not the judge.

When explaining a specific run, cite **`sample_id`**, the **query**, and short excerpts from **`report.json`** retrieval context or judge **reason** fields.
