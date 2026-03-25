# Eval variance & comparing runs — 2026-03-24

Single eval runs are **noisy**: pass rate and metric means move between back-to-back
invocations with **no pipeline changes**. Treat **comparability** as a property of
**repeated measurement** (median over several runs), not of one shot.

---

## Comparing runs (recommended workflow)

1. Keep **configs, corpus, and index** fixed while you repeat.
2. Run **`scripts/run_eval.py`** at least **three times** with the same dataset and
   config.
3. Compare the resulting MLflow runs by filtering on dataset tags and reading the
   logged `pass_rate` plus per-metric pass-rate metrics.

Example:

```bash
for _ in 1 2 3; do
  .venv/bin/python scripts/run_eval.py --dataset data/eval_datasets/synthetic.json
done
```

Use the MLflow UI or `mlflow.search_runs()` to compare the runs. The tracking backend
is now `sqlite:///data/mlflow.db`; see [mlflow-eval-tracking.md](mlflow-eval-tracking.md).

---

## Why aggregation matters (empirical noise)

On **`synthetic.json`** (25 samples, GPT-4o-mini judge), observed **single-run**
swings with **no code changes** were on the order of **~8–12 percentage points**
on pass rate depending on conditions; mean metric deltas on noisy metrics (e.g.
faithfulness, contextual recall) were material at the sample level. The tables below
are **historical measurements** from that day’s experiments (mixed config stages).

### Runs at generation `temperature: 0.1` (older default)

| Run | Pass rate | Faithfulness | Relevancy | Precision | Recall |
|-----|-----------|-------------|-----------|-----------|--------|
| `20260324T160253Z` | 44% | 0.778 | 0.882 | 0.738 | 0.879 |
| `20260324T164121Z` | 52% | 0.783 | 0.904 | 0.753 | 0.911 |
| `20260324T165535Z` | 52% | 0.824 | 0.872 | 0.736 | 0.921 |
| `20260324T170419Z` | 44% | 0.789 | 0.889 | 0.741 | 0.907 |

**Range: 44–52% (8 pp swing)**

### Runs at generation `temperature: 0.0`

| Run | Pass rate | Faithfulness | Relevancy | Precision | Recall |
|-----|-----------|-------------|-----------|-----------|--------|
| `20260324T171916Z` | 48% | 0.813 | 0.923 | 0.739 | 0.904 |
| `20260324T172743Z` | 60% | 0.826 | 0.908 | 0.730 | 0.897 |

**Range: 48–60% (12 pp swing)**

### Back-to-back runs (synthetic only; further tightening)

| Run | Pass rate | Faithfulness | Relevancy | Precision | Recall |
|-----|-----------|-------------|-----------|-----------|--------|
| `20260324T175312Z` | 48% | 0.788 | 0.897 | 0.721 | 0.940 |
| `20260324T180009Z` | 60% | 0.867 | 0.889 | 0.736 | 0.897 |

**Range: 48–60% (12 pp)** — same ballpark as the temp=0 pair above.

#### Pairwise diff (`synthetic/report.json`): `175312Z` vs `180009Z`

| Layer / signal | Count / note |
|----------------|--------------|
| Identical `retrieval_context` | **25/25** |
| Identical `generated_answer` | **14/25** |
| Sample-level pass-all flips | **7** |
| Per-metric pass/fail flips | Faithfulness **8**/25; Relevancy **0**/25; Precision **0**/25; Recall **1**/25 |

**Mean score Δ (run 2 − run 1):** Faithfulness **+0.079** (14/25 samples with \|Δ\| > 0.001); Answer Relevancy **−0.009** (5/25); Contextual Precision **+0.015** (4/25); Contextual Recall **−0.043** (5/25).

**Identical generated answer, score behavior (of 25):** 5/25 identical scores; 4/25 scores differ without pass/fail flip; **5/25** differ **with** a pass/fail flip; **11/25** answers differ. Local generation still diverged on **11/25** lines; when the answer matched, judge scores still moved on many rows.

---

## Where the variance comes from

### Layer 1: RAG generator (local LM Studio model)

| | temp=0.1 | temp=0.0 |
|---|---|---|
| Identical `retrieval_context` | 25/25 | 25/25 |
| Identical `generated_answer` | 4/25 | 15/25 |

**Root cause:** LM Studio / llama.cpp does not guarantee bitwise-identical decoding
at **temperature 0** (GPU floating-point, batch scheduling). **`configs/default.yaml`**
uses **`temperature: 0.0`** for factual Q&A; that cuts answer churn versus 0.1 but
does not remove it.

### Layer 2: DeepEval judge (cloud OpenAI)

The judge uses a **dated model snapshot** in **`configs/eval.yaml`** so scorer
behavior does not drift silently when the vendor updates a rolling alias. Even with
**identical** inputs (same answer + same retrieval context), **LLM-as-judge** scores
still vary call-to-call.

| Category | Count (of 25, historical row) |
|----------|------------------------------|
| Answer identical, scores identical | 4 |
| Answer identical, scores differ (pure judge jitter) | **11** |
| Answer identical, scores differ **with pass/fail flip** | **4** |
| Answer differs, scores differ | 7 |
| Answer differs, scores identical | 3 |

Per-metric jitter (temp=0 pair, sample-level Δ):

| Metric | Mean Δ | Stdev | Min | Max | Samples with Δ>0.001 |
|--------|--------|-------|-----|-----|---------------------|
| Faithfulness | +0.013 | 0.106 | −0.250 | +0.233 | 12/25 |
| Answer Relevancy | −0.015 | 0.051 | −0.200 | 0.000 | 2/25 |
| Contextual Precision | −0.009 | 0.047 | −0.200 | +0.083 | 4/25 |
| Contextual Recall | −0.007 | 0.164 | −0.500 | +0.500 | 6/25 |

Faithfulness and contextual recall are the noisiest (multi-step or subjective judge
reasoning).

### Layer 3: Retrieval (stable in these runs)

Retrieval was **25/25 identical** across compared pairs. Chroma + cosine + a fixed
embedding model produced stable top-k for identical queries on an unchanged index.

---

## Repo defaults relevant to eval stability

- **Generation:** `temperature: 0.0` in **`configs/default.yaml`** (reduces sampling
  diversity; does not guarantee identical text).
- **Judge:** dated snapshot id in **`configs/eval.yaml`** (stable **scorer** across
  weeks vs a rolling `gpt-4o-mini` alias).
- **Tracking:** MLflow logs run params, tags, scorer outputs, and traces under
  `sqlite:///data/mlflow.db`.

---

## How to interpret results (rule-of-thumb bands)

Use these as **single-run** noise guides until you have your own multi-run stats:

- **Pass rate changes ≤ ~5 pp** with no pipeline change: often **noise**.
- **Mean metric shifts ≤ ~0.05** on one metric: often **noise**.
- **Per-sample pass/fail flips** on ≤ ~4 samples (of 25): often **noise** — check
  whether **`generated_answer`** actually changed.
- Larger shifts, or the **same direction** across **several** repeated runs, are
  worth investigating.

Faithfulness and contextual recall need the **widest** error bars; contextual
precision and answer relevancy tend to be more stable.

---

## Optional (not default)

DeepEval metrics support **`strict_mode=True`**, which snaps scores toward discrete
pass/fail; it changes how jitter shows up, not the underlying LLM variance. Use only
if you explicitly want coarser gates.
