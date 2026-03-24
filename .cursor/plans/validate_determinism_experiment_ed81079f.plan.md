---
name: Validate determinism experiment
overview: Add lightweight automated checks that seed and meta wiring behave as designed, run `scripts/run_eval_report.py` twice back-to-back and diff reports, then update docs/determinism-2026-03-24.md with the measured post-fix findings.
todos:
  - id: tests-seed
    content: "Add unit tests: pipeline.generator.seed from YAML; Generator passes seed to chat.completions.create"
    status: pending
  - id: run-eval-2x
    content: Run scripts/run_eval_report.py twice per .cursor/commands/run-eval-report.md + run-eval-report skill; record artifact folder names
    status: pending
  - id: diff-reports
    content: Python diff of two single/report.json + meta.json determinism blocks; summarize in reply
    status: pending
  - id: update-determinism-doc
    content: Update docs/determinism-2026-03-24.md with new section + tables from the two-run experiment
    status: pending
isProject: false
---

# Validate determinism fixes and repeat two-run experiment

## Prerequisites (for the live experiment)

- **LM Studio** running with embedding + chat models matching [configs/default.yaml](configs/default.yaml) (`text-embedding-nomic-embed-text-v1.5`, `mistralai/devstral-small-2-2512`).
- `**OPENAI_API_KEY`** set (judge uses OpenAI per [configs/eval.yaml](configs/eval.yaml)).
- Existing Chroma index under `data/embeddings/chromadb` (no `--ingest` unless you want a clean re-index; re-ingest changes the experiment baseline).

If any prerequisite fails, automated validation below still passes; only the two full eval runs will be skipped or will error.

---

## How to run `scripts/run_eval_report.py` (for the executing agent)

Do **not** guess shell or timeouts. Before Part B:

1. Read [.cursor/commands/run-eval-report.md](/home/ben/repos/rag-eval-system/.cursor/commands/run-eval-report.md) in this repo.
2. Read and follow the **run-eval-report** skill: [.cursor/skills/run-eval-report/SKILL.md](/home/ben/repos/rag-eval-system/.cursor/skills/run-eval-report/SKILL.md) (same content as `.claude/skills/run-eval-report/SKILL.md` if symlinked).

Non-negotiables from those docs: work from **repo root**, use `**.venv/bin/python`**, expect long wall clock (synthetic is slower than starter), use generous timeouts or background + tail logs, and after each run confirm `**artifacts/eval_runs/<UTC>/`** contains `meta.json` plus per-dataset `report.*`. If the cloud judge rate-limits, apply `**--max-concurrent 1**` and `**--judge-throttle-seconds 5**` (see skill).

For this determinism experiment, use the **same** invocation twice (e.g. all enabled datasets via omitting `--dataset`, matching `configs/eval.yaml` — typically `single/` for synthetic-only runs).

---

## Part A — Validate the code does what we intend

These checks do **not** prove bit-identical LLM output (providers cannot guarantee that); they prove **config is applied** and **API calls include seed**.

1. **Pipeline loads seed from YAML**
  From repo root, short Python snippet (or a tiny test in [tests/test_generation.py](tests/test_generation.py)):
  - `RAGPipeline.from_config("configs/default.yaml")` then `assert pipeline.generator.seed == 42`.
2. **Generator forwards `seed` to the client**
  Unit test with `unittest.mock` patching `OpenAI.chat.completions.create` (or patch on the instance’s `client`):
  - Build `Generator(..., seed=42)` with a mock client.
  - Call `generate(...)` with minimal fake `RetrievedChunk` list (reuse patterns from [tests/test_retrieval.py](tests/test_retrieval.py) or construct minimal schema objects).
  - Assert the mock was called with `seed=42` (and `temperature=0.0`).
3. `**meta.json` includes `determinism`**
  After **one** successful eval run (Part B), open `artifacts/eval_runs/<stamp>/meta.json` and confirm:
  - `determinism.generation.seed == 42`
  - `determinism.judge.generation_kwargs` contains `"seed": 42`
  - `determinism.judge.model` matches the pinned snapshot (`gpt-4o-mini-2024-07-18`)
4. **Judge seed (optional / indirect)**
  DeepEval’s `GPTModel` merges `generation_kwargs` into `chat.completions.create` ([site-packages `deepeval/models/llms/openai_model.py](.venv/lib/python3.14/site-packages/deepeval/models/llms/openai_model.py)`). No need to mock DeepEval unless you want extra assurance; the YAML is already passed through [src/eval/judge.py](src/eval/judge.py) `TracedGPTModel(..., generation_kwargs=...)`.

---

## Part B — Repeat the manual experiment (two runs, no code changes)

1. Run twice from repo root (same flags each time; default dataset is fine):

```bash
   .venv/bin/python scripts/run_eval_report.py
   .venv/bin/python scripts/run_eval_report.py
   

```

   Note the two new directories under `artifacts/eval_runs/<UTC>/`.

1. **Compare** the two runs’ `single/report.json` with the same analysis used before:
  - Count samples with identical `retrieval_context` (expect 25/25 if index unchanged).
  - Count identical `generated_answer`.
  - List sample-level pass-all flips and per-metric pass/fail flips.
  - Summarize mean score deltas per metric.
   A small inline Python script (like the one used in chat) is enough; no need for a new committed script unless you want one under `scripts/`.
2. **Interpret** (feeds Part C)
  - If **answers** are still not 25/25 identical: residual **local inference** non-determinism (GPU / llama.cpp) despite `temperature: 0` and `seed`.
  - If **answers** match but **scores** differ: **judge** variance (OpenAI “mostly deterministic” even with `seed`).
  - Compare `meta.json` `determinism` blocks between runs — should be identical; if not, configs drifted.

---

## Part C — Update [docs/determinism-2026-03-24.md](/home/ben/repos/rag-eval-system/docs/determinism-2026-03-24.md)

After Part B, edit the doc so it reflects **current** behavior (code already wires seed, pinned judge, `meta.determinism`). Do not remove the historical tables; add a dated subsection.

**Add (minimal, factual):**

1. **“Implemented mitigations”** (short list): `generation.seed` + `temperature: 0`; judge `gpt-4o-mini-2024-07-18` + `generation_kwargs.seed`; `meta.json` `determinism` snapshot per run.
2. **“Back-to-back runs after mitigations”** — table with the **two new** `artifacts/eval_runs/<UTC>` run IDs, pass rate, and four mean scores from each run’s `meta.json` (or `report.json`).
3. **Pairwise diff summary** — same structure as the doc’s existing Layer 1 / judge sections: counts for identical `retrieval_context`, identical `generated_answer`, sample-level pass flips, per-metric pass/fail flips, and one sentence comparing to the pre-mitigation temp=0 pair (48% vs 60%) so readers see whether variance **shrunk**.
4. **Adjust outdated prose** where it says seed was absent — e.g. Layer 1 root-cause bullet “absence of a `seed` parameter” should note it is now **set in config** but local backends may still vary.
5. **Recommendations section** — mark which items are **done in repo** (seed, pin, meta) vs still **operational** (median-of-3 runs, strict_mode, log `system_fingerprint` if ever added).

---

## Deliverables

- New or extended tests in [tests/test_generation.py](tests/test_generation.py) (or `tests/test_determinism_config.py`) for items A1–A2.
- Console summary after B: headline numbers + comparison to pre-seed temp=0 pair.
- **Updated** [docs/determinism-2026-03-24.md](/home/ben/repos/rag-eval-system/docs/determinism-2026-03-24.md) per Part C.

No changes to eval thresholds or RAG logic are required for validation alone.