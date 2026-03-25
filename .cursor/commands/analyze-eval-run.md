# Analyze RAG eval run (MLflow)

You are interpreting **batch RAG evaluation** output from this repository: `scripts/run_eval.py` (and paired arms from `scripts/run_ab_study.py`). Results live in **MLflow** (`sqlite:///data/mlflow.db` by default), not under `artifacts/eval_runs/`.

**Follow the skill:** read and apply [`.cursor/skills/rag-eval-runs/SKILL.md`](../skills/rag-eval-runs/SKILL.md) for metric meanings, triage patterns, and repo touchpoints.

## 1. Locate the run

- **MLflow UI:** `mlflow ui --backend-store-uri sqlite:///data/mlflow.db` — open the experiment, sort by start time, match tags (`eval.dataset_key`, `study_id`, etc.).
- **Programmatic:** `mlflow.search_runs()` filtered by experiment id and tags; read logged metrics (`pass_rate`, per-metric pass rates).
- If the user gave a **run id**, fetch that run’s metrics, params, and tags first, then open **traces** for sample-level scorer feedback.

## 2. Metric cheat sheet (DeepEval scorers, this project)

| Metric | What it measures | Typical failure meaning |
|--------|------------------|-------------------------|
| **Faithfulness** | Answer claims supported by **retrieved** chunks | Hallucination, overreach, or format noise confusing the judge |
| **Answer relevancy** | Answer addresses the **question** | Rambling, wrong focus, or tangents |
| **Contextual precision** | Relevant chunks ranked above irrelevant ones | Noisy top-$k$ |
| **Contextual recall** | Retrieved context covers **gold** answer | Wrong or incomplete retrieval vs gold |

Thresholds come from `configs/eval.yaml` / scorer setup — confirm in the run’s logged params or config artifacts if present.

## 3. Explain the run (what to produce in chat)

1. **Headline** — overall `pass_rate`; which metrics are weakest on average.
2. **Per failing sample** — `sample_id`, query, which metric(s) failed, one sentence linking **scorer rationale** (from trace / feedback) to behavior.
3. **Root-cause bucket** — retrieval vs generation vs data/gold vs judge artifact (see skill table).

## 4. Diagrams (when asked)

Prefer **Mermaid** in chat (see `rag-eval-runs` skill): pipeline + eval attachment, or failure attribution flowchart.

## 5. Troubleshooting playbook (actionable)

Use the quick-reference table in [`.cursor/skills/rag-eval-runs/SKILL.md`](../skills/rag-eval-runs/SKILL.md). Common themes: mixed corpus hurting recall, structured JSON hurting relevancy, citation artifacts vs faithfulness.

**Repo levers:** `configs/default.yaml` (retrieval `top_k`, hybrid), `configs/eval.yaml` (`datasets`, judge), `data/corpus`, `data/eval_datasets`, `src/rag/pipeline.py`, `src/eval/datasets.py`, `src/eval/scorers.py`.

## 6. Optional: durable note

Only if the user explicitly asks for a file: add a short `ANALYSIS.md` under `docs/` or a path they specify — do not resurrect `artifacts/eval_runs/` layouts.
