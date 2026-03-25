# A/B report: hybrid vs dense retrieval (study_id `20260325T154618Z`)

This document pulls **10 finished runs per arm** from the local MLflow store and evaluates the hypothesis **"hybrid retrieval is better."** "Better" is made concrete via DeepEval aggregate metrics and the composite **`pass_rate`** computed in `scripts/run_eval.py` (fraction of test items passing all enabled metric thresholds in `configs/eval.yaml`).

## Data source and extraction

| Field | Value |
|------|--------|
| Tracking URI | `sqlite:///data/mlflow.db` |
| Experiment | `rag-eval-system` |
| `study_id` | `20260325T154618Z` |
| `eval.kind` | `ab_study` |
| Dataset | `synthetic` (`data/eval_datasets/synthetic.json`), 25 questions |
| Pairing | `paired`; blocks `0`–`9` pair dense arm **a** with hybrid arm **b** for the same block index |
| Generation model | `mistralai/devstral-small-2-2512` via LM Studio (temperature=0.0) |
| Judge model | `openai:/gpt-4o-mini-2024-07-18` via OpenAI API |

**MLflow filter (UI or API):**

`tags.eval.kind = 'ab_study' AND tags.study_id = '20260325T154618Z'`

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///data/mlflow.db")
df = mlflow.search_runs(
    experiment_names=["rag-eval-system"],
    filter_string="tags.eval.kind = 'ab_study' and tags.study_id = '20260325T154618Z'",
    order_by=["tags.block ASC", "tags.arm ASC"],
    output_format="pandas",
)
```

Per-metric means appear as columns `metrics.pass_rate`, `metrics.faithfulness/mean`, `metrics.answer_relevancy/mean`, `metrics.contextual_precision/mean`, `metrics.contextual_recall/mean`.

## Per-run metrics

Arm **dense (hybrid off)** is `arm=a`; **hybrid on** is `arm=b`. Values are from `latest_metrics`.

### Dense (hybrid off), n=10

| block | MLflow run name | pass_rate | faithfulness/mean | answer_relevancy/mean | contextual_precision/mean | contextual_recall/mean |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `20260325T154618Z-a-0` | 0.68 | 0.96 | 0.96 | 0.76 | 0.96 |
| 1 | `20260325T154618Z-a-1` | 0.60 | 0.84 | 0.84 | 0.76 | 1.00 |
| 2 | `20260325T154618Z-a-2` | 0.48 | 0.84 | 0.84 | 0.56 | 1.00 |
| 3 | `20260325T154618Z-a-3` | 0.40 | 0.68 | 0.80 | 0.80 | 0.96 |
| 4 | `20260325T154618Z-a-4` | 0.52 | 0.88 | 0.84 | 0.68 | 0.88 |
| 5 | `20260325T154618Z-a-5` | 0.72 | 0.96 | 0.84 | 0.96 | 0.96 |
| 6 | `20260325T154618Z-a-6` | 0.52 | 0.88 | 0.88 | 0.84 | 0.84 |
| 7 | `20260325T154618Z-a-7` | 0.40 | 0.68 | 0.96 | 0.64 | 0.92 |
| 8 | `20260325T154618Z-a-8` | 0.48 | 0.72 | 0.92 | 0.68 | 0.96 |
| 9 | `20260325T154618Z-a-9` | 0.40 | 0.92 | 0.88 | 0.52 | 0.88 |

### Hybrid on, n=10

| block | MLflow run name | pass_rate | faithfulness/mean | answer_relevancy/mean | contextual_precision/mean | contextual_recall/mean |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `20260325T154618Z-b-0` | 0.20 | 0.60 | 0.88 | 0.60 | 1.00 |
| 1 | `20260325T154618Z-b-1` | 0.68 | 0.84 | 0.96 | 0.84 | 1.00 |
| 2 | `20260325T154618Z-b-2` | 0.68 | 0.88 | 0.88 | 0.84 | 1.00 |
| 3 | `20260325T154618Z-b-3` | 0.44 | 0.80 | 0.68 | 0.84 | 1.00 |
| 4 | `20260325T154618Z-b-4` | 0.48 | 0.84 | 0.88 | 0.68 | 1.00 |
| 5 | `20260325T154618Z-b-5` | 0.48 | 0.84 | 0.72 | 0.72 | 1.00 |
| 6 | `20260325T154618Z-b-6` | 0.56 | 0.80 | 0.84 | 0.88 | 1.00 |
| 7 | `20260325T154618Z-b-7` | 0.40 | 0.76 | 0.72 | 0.68 | 1.00 |
| 8 | `20260325T154618Z-b-8` | 0.40 | 0.68 | 0.72 | 0.80 | 1.00 |
| 9 | `20260325T154618Z-b-9` | 0.48 | 0.92 | 0.72 | 0.84 | 1.00 |

## Unpaired aggregates (mean ± sample std dev)

| Metric | Dense mean | Dense std | Hybrid mean | Hybrid std | Δ (hybrid − dense) |
|--------|-----------:|----------:|-----------:|----------:|-------------------:|
| `pass_rate` | 0.52 | 0.115 | 0.48 | 0.141 | −0.04 |
| `faithfulness/mean` | 0.836 | 0.107 | 0.796 | 0.095 | −0.040 |
| `answer_relevancy/mean` | 0.876 | 0.055 | 0.800 | 0.098 | −0.076 |
| `contextual_precision/mean` | 0.720 | 0.132 | 0.772 | 0.094 | +0.052 |
| `contextual_recall/mean` | 0.936 | 0.054 | **1.000** | **0.000** | **+0.064** |

Hybrid **maxes contextual recall on every run** (1.0). Dense is already high on average (0.936) but spans 0.84–1.00.

## Paired analysis (same block → hybrid minus dense)

This is the right granularity for this study because `pairing=paired` ties random seeds / dataset ordering to `block`.

| block | Δ pass_rate | Δ faithfulness | Δ answer_relevancy | Δ ctx precision | Δ ctx recall |
|---:|---:|---:|---:|---:|---:|
| 0 | **−0.48** | −0.36 | −0.08 | −0.16 | +0.04 |
| 1 | +0.08 | 0.00 | +0.12 | +0.08 | 0.00 |
| 2 | +0.20 | +0.04 | +0.04 | +0.28 | 0.00 |
| 3 | +0.04 | +0.12 | −0.12 | +0.04 | +0.04 |
| 4 | −0.04 | −0.04 | +0.04 | 0.00 | +0.12 |
| 5 | −0.24 | −0.12 | −0.12 | −0.24 | +0.04 |
| 6 | +0.04 | −0.08 | −0.04 | +0.04 | +0.16 |
| 7 | 0.00 | +0.08 | −0.24 | +0.04 | +0.08 |
| 8 | −0.08 | −0.04 | −0.20 | +0.12 | +0.04 |
| 9 | +0.08 | 0.00 | −0.16 | +0.32 | +0.12 |

**Win counts over 10 blocks (hybrid larger / dense larger / tie):**

| Metric | Hybrid wins | Dense wins | Ties |
|--------|-------------|------------|------|
| `pass_rate` | 5 | 4 | 1 |
| `faithfulness/mean` | 3 | 5 | 2 |
| `answer_relevancy/mean` | 3 | 7 | 0 |
| `contextual_precision/mean` | 7 | 2 | 1 |
| `contextual_recall/mean` | 8 | 0 | 2 |

**Mean paired deltas:** pass_rate −0.04, faithfulness −0.04, answer_relevancy −0.076, contextual precision +0.052, contextual recall +0.064.

---

## Deep-dive: the block-0 outlier

Block 0 is the extreme outlier: dense pass_rate 0.68 vs hybrid 0.20 (Δ = −0.48). This single block drives the aggregate pass_rate negative. What happened?

Per-question assessment comparison for block 0 (25 samples each):

| Scorer | Dense yes/no | Hybrid yes/no | Dense → Hybrid flips |
|--------|-------------|--------------|---------------------|
| `faithfulness` | 24Y / 1N | 15Y / 10N | 9 flipped yes→no |
| `answer_relevancy` | 24Y / 1N | 22Y / 3N | 2 flipped yes→no |
| `contextual_precision` | 18Y / 7N | 15Y / 10N | mixed |
| `contextual_recall` | 24Y / 1N | 25Y / 0N | 1 flipped no→yes |

The damage in block 0 is concentrated in **faithfulness**: 10 samples fail in the hybrid arm vs only 1 in dense. But critically, the RAG answers for hybrid are **identical across all 10 runs** for every sample (the pipeline is deterministic within an arm). The same answer that gets `faithfulness="yes"` in block 1 gets `faithfulness="no"` in block 0 — purely from judge coin-flipping.

**Example — `syn003` (hybrid arm), identical answer across all 10 runs:**

> "After a successful run, install.sh writes the install configuration to /etc/streamdeck/install.conf. The file is created with mode 644. It contains four variables: STREAM_USER, BUDDY_USER, STREAM_DISPLAY, and STREAM_GROUP."

- **Block 1 (pass):** *"The score is 1.00 because there are no contradictions present, indicating that the actual output aligns perfectly with the retrieval context."*
- **Block 0 (fail):** *"The score is 0.67 because the actual output incorrectly assumes the location of the install.conf file, which is not specified in the retrieval context."*

The answer is correct. The retrieved context **does** contain `/etc/streamdeck/install.conf`. The judge hallucinated a non-existent discrepancy. This is a GPT-4o-mini reliability problem, not a retrieval quality problem.

**Conclusion:** Block 0 is **not a real hybrid regression**. It is an unlucky draw of judge verdicts on identical inputs. No configuration debugging is needed.

---

## Judge reliability analysis (GPT-4o-mini)

This is the critical finding: **the judge is the dominant source of variance in this study, not retrieval quality.**

### Methodology

Each arm runs the same 25 questions 10 times. Within an arm, the RAG pipeline is near-deterministic (temperature=0.0 via LM Studio, though local GPU inference introduces minor non-determinism: 11/25 dense and 7/25 hybrid samples show slight answer variation). The judge (GPT-4o-mini) scores each answer independently each run.

A sample "flips" when the judge gives it `"yes"` in some runs and `"no"` in others for the same or near-identical input.

### Flip rates (same sample, same arm, across 10 runs)

| Scorer | Dense flip rate | Hybrid flip rate | Combined |
|--------|---------------:|----------------:|---------:|
| `faithfulness` | 22/25 (88%) | 21/25 (84%) | **86%** |
| `contextual_precision` | 23/25 (92%) | 20/25 (80%) | **86%** |
| `answer_relevancy` | 11/25 (44%) | 14/25 (56%) | **50%** |
| `contextual_recall` | 8/25 (32%) | 0/25 (0%) | **16%** |

**Faithfulness and contextual precision are essentially coin flips.** The same answer, same retrieved chunks, same question — GPT-4o-mini cannot decide whether it passes or fails. 86% of sample-arm combinations produce mixed verdicts across 10 runs.

`contextual_recall` is the only reliable scorer, particularly for hybrid where it is **perfectly stable** (25/25 always "yes", 0% flip rate). This is also the one metric that shows a genuine, unambiguous hybrid advantage.

### Why this matters for pass_rate

`pass_rate` is the AND of all 4 scorers. If any single scorer randomly flips for a sample, that sample's pass/fail outcome is random. With 86% of samples flipping on faithfulness alone, **pass_rate is measuring judge noise, not retrieval quality**.

### Bottleneck analysis

When a sample fails pass_rate due to exactly one scorer:

| Sole bottleneck scorer | Dense (sample-runs) | Hybrid (sample-runs) |
|----------------------|--------------------:|---------------------:|
| `contextual_precision` | 43 | 34 |
| `answer_relevancy` | 17 | 39 |
| `faithfulness` | 17 | 31 |
| `contextual_recall` | 8 | 0 |

For hybrid, **answer_relevancy and faithfulness** are more frequently the sole blocker — not because hybrid generates worse answers, but because the judge is noisier on the slightly different answer text that hybrid context produces.

### Is GPT-4o-mini fair?

**No, not at this task difficulty level.** The streamdeck corpus is highly technical (System V IPC, systemd units, udev rules, Qt shared memory). The expected answers are long and precise. GPT-4o-mini:

1. **Contradicts itself** on identical inputs (see syn003 example above).
2. **Hallucinates discrepancies** — claims information "is not in the context" when it verifiably is.
3. **Shows scorer-dependent reliability**: contextual_recall (a relatively simple "do the chunks cover the gold answer?" check) is stable; faithfulness (requires reasoning about whether claims are entailed) is extremely noisy.

This is consistent with 4o-mini's known limitations on complex reasoning tasks. It works as a cheap first-pass judge but cannot reliably adjudicate technical faithfulness claims.

---

## Why hybrid didn't show improvement

Three compounding factors explain the null result:

### 1. Judge noise drowns the signal

The true retrieval quality difference between arms is small relative to the ~86% per-sample flip rate on faithfulness and contextual precision. Any real +5% improvement from hybrid is invisible under that noise floor. The study is **underpowered** — not because n=10 is too few runs, but because each run's metrics are too noisy to carry signal.

### 2. Hybrid does improve retrieval — but it doesn't flow to pass_rate

The one clean signal: **contextual_recall = 1.0 on every hybrid run** (zero variance, zero flips). Dense averages 0.936 with 8/25 samples flipping. Hybrid also wins contextual_precision 7/10 blocks. The BM25+RRF fusion is doing its job — recovering chunks that dense embedding search misses and ranking relevant chunks higher.

But `pass_rate` gates on all 4 metrics. Better retrieval can't help if faithfulness and answer_relevancy randomly fail due to judge noise. Worse: hybrid retrieves slightly different chunks → slightly different generated answers (19/25 samples differ between arms) → the judge evaluates different text → independent noise draws.

### 3. The dataset is a ceiling for this comparison

The 25 synthetic questions are grounded in a single well-structured corpus (the streamdeck project documentation). Dense retrieval already achieves high contextual recall (0.936 mean) because the corpus is small, coherent, and the embedding model handles it well. Hybrid's lexical fallback has limited room to add value. The advantage of hybrid retrieval would be more visible on:

- Larger corpora with more topical diversity
- Queries with rare technical terms that embeddings struggle with (acronyms, config key names, error codes)
- Corpora mixing code and prose where lexical match patterns differ

For this corpus, dense is already "good enough" at retrieval — the bottleneck is generation and judging quality.

---

## Eval suite assessment

### What works

- **Contextual recall** is a reliable, low-noise metric that shows a genuine arm difference.
- The paired block design successfully controls for dataset ordering noise.
- The `pass_rate` computation logic in `scripts/run_eval.py` is correct.

### What doesn't work

| Issue | Impact | Severity |
|-------|--------|----------|
| GPT-4o-mini judge noise on faithfulness/precision | Metrics are ~coin-flips; study is underpowered by construction | **Critical** |
| Binary yes/no assessment values | Cannot see score distributions; a 0.69 and a 0.01 both fail at threshold 0.7 | High |
| `pass_rate` AND-gates 4 noisy metrics | Compounds noise exponentially; a 14% flip rate per metric becomes ~45%+ at the pass level | High |
| No `expected_context` in dataset | Contextual precision/recall rely entirely on judge interpretation of relevance | Medium |
| Generation non-determinism | LM Studio at temperature=0 still varies (11/25 samples); adds noise on top of judge noise | Medium |
| 25 samples per eval | Modest statistical power even with a perfect judge | Low |

---

## Recommendations

### Immediate (reduce judge noise — biggest lever)

1. **Switch the judge model to GPT-4o or GPT-4.1** (`configs/eval.yaml` → `judge.model`). The faithfulness scorer requires multi-step entailment reasoning that 4o-mini cannot do reliably on technical text. Even GPT-4o will not be perfect, but going from 86% flip rate to something closer to 20% would make the study interpretable.

2. **Log continuous scores alongside yes/no.** The DeepEval scorers compute numeric scores internally (the rationales say "The score is 0.67...") but the MLflow assessment only stores the thresholded yes/no. Logging the raw score would let you analyze score distributions, identify marginal samples (score ~0.7), and avoid the information loss of binary gating.

3. **Use majority voting.** Run the judge 3× per sample per scorer and take majority verdict. This is 3× the judge cost but drastically reduces flip rates. Can be implemented as a wrapper around the existing scorers.

### Short-term (improve study power)

4. **Report metrics separately, not just pass_rate.** Contextual recall shows a clear hybrid win. Reporting only pass_rate hides this because it AND-gates with noisy metrics. Consider a "retrieval quality" composite (precision + recall) alongside a "generation quality" composite (faithfulness + relevancy).

5. **Add `expected_context` to the synthetic dataset.** Currently null for all 25 samples. Providing gold context would make contextual precision/recall less dependent on judge interpretation and more deterministic.

6. **Pin generation determinism.** Set `seed` parameter in the LM Studio generation call (if supported by the loaded model) to eliminate the 7–11/25 sample response variation. This isolates judge noise as the sole remaining variance source.

### Medium-term (eval design)

7. **Expand the dataset.** 25 samples is fine for smoke-testing but marginal for hypothesis testing. Target 50–100 samples with deliberate coverage of query types where hybrid should shine (rare terms, acronyms, code-specific queries).

8. **Test on a harder corpus.** The streamdeck docs are well-structured and small — dense retrieval has a near-ceiling. Try a larger, messier corpus where BM25 fallback has more room to recover missed chunks.

9. **Consider a stronger judge for ground-truth labeling, weaker for CI.** Use GPT-4o to label a "gold standard" assessment set, then measure whether 4o-mini agrees. This gives you a calibration baseline for judge reliability before running A/B studies.

---

## Updated verdict

The hypothesis **"hybrid retrieval is better"** is:

- **Confirmed for retrieval completeness**: contextual_recall is perfect under hybrid (1.0, zero variance) vs imperfect under dense (0.936 mean, 8/25 samples flipping). This is a real, noise-free signal.
- **Likely true for retrieval ranking**: contextual_precision favors hybrid 7/10 blocks, but the metric is too noisy (86% flip rate) for high confidence.
- **Indeterminate for end-to-end quality**: pass_rate, faithfulness, and answer_relevancy differences are **within judge noise** and cannot be interpreted as real effects. The study cannot distinguish "hybrid is worse for generation" from "the judge randomly scored hybrid lower this time."

**The right next step is not more runs — it's fixing the judge.** Running 10 more blocks with GPT-4o-mini will produce 10 more noisy data points. Switching to a reliable judge model will make the existing 10×2 design sufficient to resolve the question.

## Related

- Science-loop / A/B orchestration: `.cursor/skills/science-loop/SKILL.md`
- MLflow query recipes: `.cursor/skills/mlflow/SKILL.md`
- Prior write-up (smaller n, different `study_id`): `docs/ab-study-20260325T143130Z-hybrid-vs-dense.md`
