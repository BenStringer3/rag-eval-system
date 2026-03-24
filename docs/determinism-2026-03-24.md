# Eval Determinism & Repeatability — 2026-03-24

## Problem

Back-to-back eval runs with **no code or config changes** produce different pass
rates. This makes it impossible to attribute score movements to pipeline changes
vs. random noise.

### Observed variance (all runs on `synthetic.json`, 25 samples, GPT-4o-mini judge)

#### Runs at generation `temperature: 0.1` (original default)

| Run | Pass rate | Faithfulness | Relevancy | Precision | Recall |
|-----|-----------|-------------|-----------|-----------|--------|
| `20260324T160253Z` | 44% | 0.778 | 0.882 | 0.738 | 0.879 |
| `20260324T164121Z` | 52% | 0.783 | 0.904 | 0.753 | 0.911 |
| `20260324T165535Z` | 52% | 0.824 | 0.872 | 0.736 | 0.921 |
| `20260324T170419Z` | 44% | 0.789 | 0.889 | 0.741 | 0.907 |

**Range: 44–52% (8 pp swing)**

#### Runs at generation `temperature: 0.0` (after fix)

| Run | Pass rate | Faithfulness | Relevancy | Precision | Recall |
|-----|-----------|-------------|-----------|-----------|--------|
| `20260324T171916Z` | 48% | 0.813 | 0.923 | 0.739 | 0.904 |
| `20260324T172743Z` | 60% | 0.826 | 0.908 | 0.730 | 0.897 |

**Range: 48–60% (12 pp swing) — still substantial.**

Setting the generator to temperature 0 did not make runs deterministic.

---

## Where the non-determinism comes from

### Layer 1: RAG generator (local LM Studio model)

| | temp=0.1 | temp=0.0 |
|---|---|---|
| Identical `retrieval_context` | 25/25 | 25/25 |
| Identical `generated_answer` | 4/25 | 15/25 |

Dropping temperature from 0.1 to 0.0 improved answer stability from 16% to 60%,
but **10 of 25 answers still differ** across two consecutive runs at temp=0.

**Root cause:** LM Studio / llama.cpp does not guarantee bitwise-deterministic
sampling at temperature 0. GPU floating-point non-associativity, batch scheduling,
and the absence of a `seed` parameter in the current request path all contribute.
The OpenAI cloud API has the same limitation — their `seed` parameter is documented
as providing only "mostly deterministic" outputs, and the `system_fingerprint` can
change across infrastructure updates.

### Layer 2: DeepEval judge (GPT-4o-mini via OpenAI API)

The judge is configured at `temperature: 0` in `configs/eval.yaml`. But even with
**identical inputs** (same answer + same retrieval context), judge scores vary:

| Category | Count (of 25) |
|----------|--------------|
| Answer identical, scores identical | 4 |
| Answer identical, scores differ (pure judge jitter) | **11** |
| Answer identical, scores differ **with pass/fail flip** | **4** |
| Answer differs, scores differ | 7 |
| Answer differs, scores identical | 3 |

**11 of 15 samples with identical answers still got different scores.** 4 of those
flipped pass/fail status. This is the **dominant remaining source** of
non-determinism after setting generator temp to 0.

Per-metric jitter (across all 25 samples, temp=0 pair):

| Metric | Mean Δ | Stdev | Min | Max | Samples with Δ>0.001 |
|--------|--------|-------|-----|-----|-----------------------|
| Faithfulness | +0.013 | 0.106 | −0.250 | +0.233 | 12/25 |
| Answer Relevancy | −0.015 | 0.051 | −0.200 | 0.000 | 2/25 |
| Contextual Precision | −0.009 | 0.047 | −0.200 | +0.083 | 4/25 |
| Contextual Recall | −0.007 | 0.164 | −0.500 | +0.500 | 6/25 |

Faithfulness and contextual recall are the noisiest. Faithfulness uses multi-step
claim extraction + verification (multiple judge calls per sample), amplifying per-call
variance. Contextual recall compares retrieval to gold via LLM reasoning, which is
subjective even for the same inputs.

### Layer 3: Retrieval (stable in these runs)

Retrieval context was **25/25 identical** across all compared run pairs. Chroma
with cosine distance and a deterministic embedding model (nomic-embed-text-v1.5)
produces stable retrieval for identical queries against an unchanged index. This
layer is **not** currently contributing to variance.

---

## Recommendations

### 1. Pass `seed` to both generator and judge API calls

OpenAI's `seed` parameter does not guarantee perfect determinism, but it
significantly reduces variance in practice. For the local LM Studio generator,
llama.cpp also supports a `seed` field; passing a fixed value removes the
random-seed-per-request behavior.

**Generator** — add `seed` to `configs/default.yaml` and thread it through
`Generator.generate()`:

```yaml
generation:
  temperature: 0.0
  seed: 42
```

**Judge** — add `seed` to `configs/eval.yaml` `judge.generation_kwargs` (DeepEval's
`GPTModel` passes `generation_kwargs` directly to the OpenAI client):

```yaml
judge:
  temperature: 0
  generation_kwargs:
    max_completion_tokens: 10000
    seed: 42
```

### 2. Keep generation temperature at 0

Temperature 0.1 caused 84% of answers to vary. Temperature 0.0 reduced that to
40%. For a factual, corpus-grounded Q&A system there is no benefit to sampling
diversity — keep it at 0.

### 3. Average over N runs for meaningful comparisons

Even with seed + temp=0, small jitter will remain (GPU non-determinism, OpenAI
infrastructure changes). For high-confidence comparisons:

- Run the eval **3 times** and report **median** pass rate and metric means.
- A pipeline change is **signal** (not noise) if median scores shift by more than
  the observed single-run jitter band (~5 pp for pass rate, ~0.05 for metric means).

### 4. Consider `strict_mode=True` for DeepEval metrics

DeepEval metrics accept `strict_mode=True`, which snaps scores to 0 or 1 based on
the threshold. This eliminates fractional-score jitter but loses granularity — only
use if you care about pass/fail counts, not continuous improvement tracking.

### 5. Pin the judge model version

`gpt-4o-mini` is a rolling alias. OpenAI periodically updates the weights behind
it (reflected in `system_fingerprint`). For long-term reproducibility, pin to a
dated snapshot (e.g. `gpt-4o-mini-2024-07-18`) in `configs/eval.yaml`.

### 6. Log `system_fingerprint` per run

When using the OpenAI judge, capture and record the `system_fingerprint` from
responses in `meta.json`. If fingerprints differ between compared runs, score
differences may be due to a model update, not your pipeline change.

---

## How to interpret eval results given current noise

Until full determinism is achieved, treat eval results as follows:

- **Pass rate changes ≤ 5 pp** between runs with no pipeline change: **noise**.
- **Mean metric shifts ≤ 0.05** on any single metric: **noise**.
- **Per-sample pass/fail flips** on ≤ 4 samples (of 25): likely **noise** — verify
  by checking whether the answer text actually changed.
- Changes **larger** than these bands, especially if consistent across 2+ runs,
  are **signal** worth investigating.

The noisiest metrics (faithfulness, contextual recall) need the widest error bars.
Contextual precision and answer relevancy are more stable.
