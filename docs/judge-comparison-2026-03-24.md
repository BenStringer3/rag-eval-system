# Judge Comparison: GPT-4o-mini vs Devstral Small 2

**Date:** 2026-03-24
**Dataset:** `data/eval_datasets/synthetic.json` (25 samples)
**Generation model:** `mistralai/devstral-small-2-2512` (identical for both runs)
**Retrieval pipeline:** unchanged between runs (same corpus, embeddings, `top_k`)

| | Devstral Judge | GPT-4o-mini Judge |
|---|---|---|
| **Run** | `20260324T085932Z` | `20260324T142044Z` |
| **Judge model** | `mistralai/devstral-small-2-2512` | `gpt-4o-mini` (OpenAI cloud) |
| **Pass rate** | 32% (8/25) | 40% (10/25) |

Because the generation model and retrieval pipeline are identical, every difference
in scores is attributable to the judge alone. The generated answers and retrieved
chunks are the same across both runs.

---

## 1. Aggregate Metric Comparison

| Metric | Devstral | GPT-4o-mini | Delta | Notes |
|--------|----------|-------------|-------|-------|
| Faithfulness | 0.757 | 0.809 | **+0.052** | Moderate improvement |
| Answer Relevancy | 0.822 | 0.819 | **−0.003** | Effectively flat |
| Contextual Precision | 0.770 | 0.781 | **+0.011** | Minor improvement |
| Contextual Recall | 0.657 | 0.931 | **+0.274** | Massive swing |

**Contextual Recall is the headline.** Devstral scored CR below threshold on 12 of 25
samples; GPT-4o-mini scored CR below threshold on only 1 (syn023, which is a genuine
total retrieval miss). The previous findings doc already flagged devstral's CR scoring
as unreliable — GPT-4o-mini confirms that assessment.

---

## 2. Per-Sample Pass/Fail Shifts

### Flipped to PASS (5 samples: Devstral ❌ → GPT-4o-mini ✅)

| Sample | Devstral failure | GPT-4o-mini fix | Verdict |
|--------|-----------------|-----------------|---------|
| **syn001** | Faith 0.50 — "implies context specifies groups" | Faith 1.00 — "no contradictions" | **GPT-4o-mini correct.** Answer literally says "does not explicitly state." Devstral misread it. |
| **syn008** | AR 0.50 — "incorrectly references moondeck prefix" | AR 0.80 — passed | **GPT-4o-mini correct.** Gold answer itself uses "moondeck" prefix. |
| **syn015** | CR 0.50 — "checking UDP 47999 lacks support" | CR 1.00 + all pass | **GPT-4o-mini likely correct.** Gold facts are in context. |
| **syn020** | CP 0.333 — "relevant node buried at rank 3" | CP 0.756 — passed | **GPT-4o-mini more accurate.** Both agree ranking is imperfect; GPT-4o-mini gives proportional credit. |
| **syn022** | AR 0.667 — "unnecessary detail about setting to false" | AR 1.00 — "no irrelevant statements" | **GPT-4o-mini correct.** Question asks "what value must it be set to?" — answering "false" is the point. |

All five flips resolve issues that the previous findings doc flagged as devstral judge
errors (syn001, syn008, syn022 explicitly; syn015/syn020 via the broader CR/CP underscoring
pattern).

### Flipped to FAIL (3 samples: Devstral ✅ → GPT-4o-mini ❌)

| Sample | Devstral pass | GPT-4o-mini failure | Verdict |
|--------|--------------|---------------------|---------|
| **syn005** | AR 1.00 | AR 0.50 — "many statements irrelevant to the flags question" | **Debatable.** The answer includes detailed Xorg flags (which were asked for) plus surrounding context. GPT-4o-mini may be over-strict on tangential supporting detail. |
| **syn011** | Faith 1.00, AR 1.00 | Faith 0.667 — "keys are fixed, not SHA1-derived"; AR 0.667 | **GPT-4o-mini may be right.** If the generated answer claims SHA1 derivation but the context says keys are fixed, that's a real hallucination devstral missed. |
| **syn014** | Faith 0.857 | Faith 0.625 — "misrepresents permission denied as causing abort" | **Mixed.** Exit 134 IS SIGABRT (128+6), so GPT-4o-mini's phrasing "incorrectly states code 134 indicates abort" is itself wrong. But the underlying point about causation vs correlation may be valid. |

Two of the three regressions are GPT-4o-mini applying stricter standards. syn011 likely
caught a genuine hallucination. syn005 looks like over-penalisation. syn014 is arguable.

### Stable passes (5 samples)

syn009, syn010, syn019, syn021, syn024 — passed under both judges.

### Stable failures (12 samples)

syn002, syn003, syn004, syn006, syn007, syn012, syn013, syn016, syn017, syn018, syn023, syn025.

---

## 3. The Contextual Recall Problem

This is the most important finding. GPT-4o-mini gave CR = 1.0 on **22 of 25** samples.
Devstral gave CR = 1.0 on only 13 of 25. The largest per-sample swings:

| Sample | Devstral CR | GPT-4o-mini CR | Delta |
|--------|------------|----------------|-------|
| syn003 | 0.00 | 1.00 | +1.00 |
| syn006 | 0.00 | 1.00 | +1.00 |
| syn013 | 0.00 | 1.00 | +1.00 |
| syn004 | 0.25 | 1.00 | +0.75 |
| syn002 | 0.33 | 1.00 | +0.67 |
| syn016 | 0.33 | 1.00 | +0.67 |

**Devstral was clearly underscoring CR.** The previous findings doc showed syn003 had a
verbatim chunk match the judge scored 0.0 — pure hallucination by the devstral judge.
GPT-4o-mini correctly scores it 1.0.

**But GPT-4o-mini may be overscoring CR.** 22/25 perfect recall is suspicious. Two specific
contradictions:

- **syn006**: CP = 0.0 (judge says "all nodes irrelevant, none provide capture method/encoder/log
  directory values") but CR = 1.0 (judge says "every aspect of expected output directly
  supported by nodes"). If the nodes don't contain the information, they can't support the
  gold answer that asks for that same information. Cross-metric contradiction.

- **syn013**: CP = 0.0 ("all nodes irrelevant, do not provide details about wrapper actions")
  but CR = 1.0 ("every action described is directly supported"). Same contradiction.

GPT-4o-mini's CR scoring for these two samples is likely wrong — the same nodes can't both
"lack the necessary details" (CP judge) and "directly support every aspect" (CR judge).
This mirrors the cross-metric contradictions devstral exhibited, just at a lower frequency
(2 samples vs ~8).

---

## 4. Faithfulness Scoring Comparison

Faithfulness improved modestly (+0.052 mean). The most significant per-sample changes:

| Sample | Devstral Faith | GPT-4o-mini Faith | Improved? |
|--------|---------------|-------------------|-----------|
| syn001 | 0.50 | 1.00 | ✅ Correctly recognised abstention |
| syn003 | 0.00 | 1.00 | ✅ No longer penalises "I don't know" |
| syn006 | 0.33 | 1.00 | ✅ No longer invents claims |
| syn004 | 1.00 | 0.71 | ⚠️ Found inaccuracies devstral missed |
| syn014 | 0.86 | 0.63 | ⚠️ Stricter on causation claims |
| syn023 | 0.75 | 1.00 | ✅ Correctly scored complete abstention |

GPT-4o-mini handles abstention answers ("the context doesn't say...") consistently and
correctly. Devstral was the opposite — scoring identical abstention patterns anywhere from
0.0 to 1.0. This was the #1 judge quality issue in the previous findings doc, and
GPT-4o-mini resolves it.

Where GPT-4o-mini scores *lower* on faithfulness, the reasons tend to be more specific and
technically grounded (syn004: "misrepresents the command"; syn014: "misrepresents the
permission denied error"). Whether these are correct requires reading the raw context, but
the reasoning is at least coherent.

---

## 5. Summary of Judge Quality

| Dimension | Devstral Small 2 | GPT-4o-mini |
|-----------|-----------------|-------------|
| **Abstention handling** | Inconsistent (0.0–1.0 for identical patterns) | Consistent (correctly scores as faithful) |
| **Contextual Recall** | Severe underscoring; 12/25 below threshold, many incorrect | Likely overscoring; 22/25 perfect, 2 contradictions |
| **Cross-metric contradictions** | ~8 samples | ~2 samples |
| **Domain syntax** (udev, systemd) | Confused by `TAG-=` syntax | Handles better but still imperfect |
| **Faithfulness on real hallucinations** | Missed some (syn011 SHA1 claim) | Catches more, but occasionally wrong (syn014) |
| **Answer Relevancy strictness** | Sometimes penalises correct content | Sometimes over-penalises supporting context |
| **Cost** | Free (local LM Studio) | $0.059 / 25 samples |

---

## 6. Impact on Eval Trustworthiness

The pass rate moved from 32% to 40%. Decomposing this:

- **+5 flips to pass**: Almost all are corrections of devstral judge errors. The "true" pass
  rate was always higher than 32%.
- **−3 flips to fail**: ~1 is a genuine hallucination catch (syn011), ~1 is over-strictness
  (syn005), ~1 is arguable (syn014).
- **Net**: The 40% figure is more trustworthy than 32%, but still not fully reliable due to
  GPT-4o-mini's CR overscoring on 2 samples.

**Estimated "true" pass rate**: 40–44%. The two GPT-4o-mini CR contradictions (syn006,
syn013) don't affect pass/fail because those samples fail on other metrics anyway. syn005's
AR=0.50 may be an over-penalisation — if corrected, the pass rate would be 44% (11/25).

---

## 7. Recommendations

### Judge model: keep GPT-4o-mini as default

It resolves the most critical devstral issues (abstention scoring, CR collapse,
cross-metric contradictions) at low cost ($0.06/run). The remaining inconsistencies are
minor compared to devstral's.

### RAG pipeline: the real bottleneck is now visible

With a more reliable judge, the genuine failure pattern emerges:

| Failure bucket | Samples | Root cause |
|----------------|---------|------------|
| **Retrieval miss** | syn002, syn006, syn023 | Gold facts not in top-5 chunks |
| **Generation doesn't use context** | syn003, syn004 | Chunk present but model says "I don't know" |
| **Faithfulness — hallucinated details** | syn007, syn011, syn012, syn017, syn018, syn025 | Model adds/distorts claims beyond context |
| **Precision — noisy top-k** | syn013, syn016 | Relevant chunks buried or absent |

Priority levers (unchanged from previous findings):
1. **Chunk deduplication** — syn004's 4× duplicate `systemctl status` blobs
2. **top_k bump to 7–8** — helps multi-fact queries (syn002, syn006)
3. **System prompt grounding** — reduce hallucinated details for the 6 faithfulness failures

### Next eval run

Re-run with GPT-4o-mini judge after any RAG changes. The 40% baseline is now a meaningful
signal to improve against.
