# Analyze RAG eval run (DeepEval reports)

You are interpreting **batch RAG evaluation** output from this repository: `scripts/run_eval_report.py` and the artifacts it writes under `artifacts/eval_runs/<UTC>/`. Each enabled dataset from `configs/eval.yaml` gets its own subfolder (e.g. `starter/`, `synthetic/`).

## 1. Locate the run

- Default base: `artifacts/eval_runs/` (timestamped subfolders, e.g. `20260323T193707Z`).
- If the user gave a path, use it. Otherwise prefer the **latest** directory by name (lexicographic UTC stamp works for ISO-like folders).
- In that folder, read **`meta.json`** first (run id, thresholds, **`datasets`** array with per-dataset path / pass rate / mean scores).
- Then open that dataset’s reports, e.g. **`starter/report.md`**, plus **`starter/report.json`** or **`starter/report.csv`** in the same subfolder as needed. If the user did not name a dataset, use **`meta.json`**’s `datasets` list and either summarize all or focus on the weakest `pass_rate`.

## 2. Metric cheat sheet (DeepEval, this project)

| Metric | What it measures | Typical failure meaning |
|--------|------------------|-------------------------|
| **Faithfulness** | Answer claims supported by **retrieved** chunks (not full corpus) | Hallucination, overreach, or **format noise** (e.g. bogus `[1]` citations, JSON wrappers confusing the judge) |
| **Answer relevancy** | Answer addresses the **question** | Rambling, wrong focus, or tangents (often when retrieval mixes unrelated docs) |
| **Contextual precision** | Ordering/purity: relevant chunks ranked above irrelevant ones | Noisy retrieval: good answer possible but top-$k$ includes junk |
| **Contextual recall** | Retrieved context covers **expected (gold) answer** | **Wrong or incomplete retrieval**: gold facts not in any retrieved chunk (common when the index is dominated by another topic in a mixed corpus) |

Thresholds (defaults): core metrics **0.7**, retrieval metrics **0.6** — confirm in `meta.json` / `report.json` `config` if present.

## 3. Explain the run (what to produce in chat)

For the selected run, give:

1. **Headline** — overall pass rate; which metrics are weakest on average (use `meta.json` / markdown table).
2. **Per failing sample** — query id, which metric(s) failed, and **one sentence** linking the judge `reason` to behavior.
3. **Root-cause bucket** — label each failure as primarily **retrieval**, **generation**, **data/gold mismatch**, or **metric/judge artifact** (see §5).

Use quotes from the chosen dataset’s `report.md` / `report.json` for judge reasons when useful.

## 4. Diagrams (when asked)

Prefer **Mermaid** in chat:

- **Pipeline + eval attachment:** flow from `Query` → `Embed query` → `Retrieve top-k` → `LLM answer` → four metrics (faithfulness/relevancy read **answer + chunks**; precision/recall read **chunks + query + gold**).
- **Failure attribution:** for a single sample, small flowchart: "Recall low?" → "Gold not in chunks → fix retrieval/corpus/chunking" vs "Faithfulness low?" → "Answer vs chunks → fix prompt/format/grounding".

Keep diagrams minimal (5–12 nodes).

## 5. Troubleshooting playbook (actionable)

- **Contextual recall collapsed but precision high** — Retrieved text is **on-topic for the generator** but **does not contain gold phrasing/facts** (mixed corpus: e.g. streaming/handheld docs + RAG docs). Fixes: filter corpus, metadata boosts, separate collections, re-ingest, or **relax gold** if the question is ambiguous.
- **Answer relevancy low with long JSON answers** — Generator emits structured JSON; judge may score **narrative** poorly. Fixes: prompt for plain prose for eval, or parse `actual_output` before metrics in the pipeline (product change).
- **Faithfulness ding for "[n] citations"** — Chunks lack explicit citation markers; model invented references. Fixes: citation policy in prompt, or treat as known judge noise if answer is otherwise grounded.
- **Recall 1.0 but faithfulness/relevancy low** — Context is enough for gold but **answer** is wrong or off-question. Fixes: generation prompt, temperature, model.
- **Volatile scores across runs** — Local judge (LM Studio) variance. Mitigation: temperature 0, stronger judge model, average over multiple runs.

Link suggestions to this repo: `configs/default.yaml` (retrieval `top_k`, chunking), `configs/eval.yaml` (`datasets` registry), `data/corpus`, `data/eval_datasets`, `src/rag/pipeline.py`, `src/eval/metrics.py`, `src/eval/eval_config.py`.

## 6. Optional: write a run-scoped note

If the user wants a durable artifact, add **`ANALYSIS.md`** inside the same `artifacts/eval_runs/<stamp>/` folder (run-wide summary) or inside a dataset subfolder if the note is dataset-specific — only when they explicitly ask for a file.
