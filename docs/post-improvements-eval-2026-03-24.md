# Post-Improvements Eval: 40% → 44%

**Date:** 2026-03-24
**Run:** `20260324T160253Z` (44% pass rate, 11/25), after re-ingest with updated chunking
**Baseline:** `20260324T142044Z` (40% pass rate, 10/25 — from [judge-comparison-2026-03-24.md](judge-comparison-2026-03-24.md))
**Dataset:** `data/eval_datasets/synthetic.json` (25 samples)
**Judge:** GPT-4o-mini (unchanged)
**Generation model:** `mistralai/devstral-small-2-2512` (unchanged)

---

## Changes Applied

Three changes from [rag-improvements-2026-03-24.md](rag-improvements-2026-03-24.md),
applied simultaneously before re-ingesting the corpus:

| # | Change | Rationale |
|---|--------|-----------|
| 1 | Markdown `chunk_size` 512 → 1024, overlap 64 → 128 | Keep documentation sections whole; double context density per retrieval slot |
| 2 | Query-time text-hash dedup in `retriever.py` (over-fetch 2×, return unique top-k) | Eliminate wasted slots from near-duplicate chunks (syn004 had 4× copies) |
| 3 | System prompt: "ONLY the provided context … do not add information from your own knowledge" | Reduce the 6 faithfulness failures caused by pre-training knowledge leaking in |

---

## 1. Aggregate Comparison

| Metric | Baseline | Post-improvements | Delta |
|--------|----------|-------------------|-------|
| **Pass rate** | 40% (10/25) | 44% (11/25) | **+4 pp** |
| Faithfulness | 0.809 | 0.778 | −0.031 |
| Answer Relevancy | 0.819 | 0.882 | **+0.063** |
| Contextual Precision | 0.781 | 0.738 | −0.043 |
| Contextual Recall | 0.931 | 0.879 | −0.052 |

**Pass rate improved** (+1 net sample), but the metric means tell a more nuanced story.
Answer Relevancy is the clear winner — the stricter prompt produces more focused answers.
Faithfulness, Contextual Precision, and Contextual Recall all dipped slightly. The retrieval
metric drops are expected: larger chunks reshape the embedding space, so some queries that
previously matched small focused chunks now match broader (sometimes less precise) ones.

---

## 2. Per-Sample Shifts

### Flipped to PASS (5 samples)

| Sample | Baseline failure | Post-improvements | Likely cause |
|--------|-----------------|-------------------|--------------|
| **syn005** | AR 0.50 — "many statements irrelevant to flags question" | All pass (Faith 0.83, AR 0.83) | Larger chunks kept the full Xorg flags section intact; model could answer specifically instead of padding with tangential context |
| **syn012** | Stable fail (generation said "I don't know" or hallucinated) | All pass (Faith 1.0, AR 1.0, CP 0.89) | TMPDIR-mismatch explanation fits in one 1024-char chunk instead of being split; grounding prompt helped model use it faithfully |
| **syn016** | Stable fail | All pass (Faith 0.83, AR 0.88, CP 0.95, CR 0.80) | Input isolation section kept whole in larger chunk; dedup may have freed a slot for the MatchProduct explanation |
| **syn018** | Stable fail | All pass (Faith 0.80, AR 1.0, CP 1.0, CR 1.0) | Env-var section no longer split mid-sentence; retrieval returns both the regex pattern and the install.sh mechanism |
| **syn025** | Stable fail | All pass (Faith 1.0, AR 1.0, CP 0.89, CR 1.0) | collect-logs.sh section fits in fewer chunks; dedup eliminated repeated log-path mentions across files |

All five are samples where the **combination of larger chunks + dedup** delivered more
coherent, less redundant context — and the **grounding prompt** kept the model from
embellishing.

### Flipped to FAIL (4 regressions)

| Sample | Baseline pass | Post-improvements failure | Analysis |
|--------|--------------|--------------------------|----------|
| **syn001** | Faith 1.0, AR 0.80, all pass | Faith 0.50 — "implies STREAM_USER runs Sunshine and Xorg" | **Generation overreach.** The model inferred roles for STREAM_USER beyond what context states. Grounding prompt didn't prevent this — the inference is plausible enough that the model treats it as context-supported. |
| **syn020** | All pass | Faith 0.50, AR 0.50 — "claims context doesn't specify resolution details" | **Over-cautious abstention.** Context clearly says "1280x800" (CP = 1.0), but the model says "does not specify." The stricter prompt may push toward refusal when the model is uncertain about Modeline details even though resolution is stated. |
| **syn021** | All pass (CP 1.0) | CP 0.25, CR 0.50 — relevant chunk buried at rank 4 | **Retrieval regression.** Larger chunks shifted embeddings; the BUDDY_USER detection-order text now loses to broader chunks in vector similarity. This is a chunking trade-off, not a prompt issue. |
| **syn022** | All pass (Faith 1.0) | Faith 0.667 — "misrepresents wait-all behavior" | **Marginal faithfulness miss.** The answer is substantively correct ("wait-all = false so stream ends") but the judge penalises the phrasing. |

Three of four regressions are generation-side (syn001, syn020, syn022) and one is retrieval
(syn021).

### Stable passes (6 samples)

syn008, syn009, syn010, syn015, syn019, syn024 passed in both baseline and post-improvements.

### Stable failures (10 samples)

syn002, syn003, syn004, syn006, syn007, syn011, syn013, syn014, syn017, syn023 failed in both.

---

## 3. Impact Attribution by Change

### Chunk size 512 → 1024

The dominant factor. Responsible for most of the 5 new passes (sections that were
previously split mid-sentence now fit in one chunk) and also the syn021 retrieval regression
(BUDDY_USER chunk lost ranking in the reshaped embedding space).

| Effect | Samples | Mechanism |
|--------|---------|-----------|
| ✅ Sections kept whole | syn012, syn016, syn018, syn025 | Full documentation section in one chunk → model can answer |
| ✅ Higher context density | syn005 | 5 × 1024 chars ≈ old 10 × 512 chars; multi-fact queries fit |
| ❌ Embedding shift | syn021 | Broader chunks compete differently in vector space; some niche queries lose |

### Query-time dedup

Hard to isolate from chunk-size effects since both were applied simultaneously. The
clearest signal is syn004: in baseline, the model abstained entirely (AR 0.0); now it
attempts an answer (Faith 1.0, AR 1.0) because dedup freed slots that previously held
4× duplicate `systemctl status` blobs. syn004 still fails on CP (0.25 — relevant chunk
at rank 4), but the answer quality improved dramatically.

### System prompt grounding

Mixed results. The "ONLY the provided context" directive improved Answer Relevancy across
the board (+0.063 mean) — answers are more focused. But it didn't eliminate faithfulness
failures: 7 samples still have Faith < 0.7, including syn007 (0.20) and syn017 (0.20)
where the model invents port numbers and udev details despite being told not to.

The prompt may also contribute to syn020's regression (over-cautious abstention: "context
does not specify" when it does). The "say what is missing rather than guessing" instruction
can tip a borderline case toward refusal.

---

## 4. Remaining Failure Buckets

| Bucket | Samples | Count | Notes |
|--------|---------|-------|-------|
| **Retrieval miss** | syn002, syn006, syn017, syn023 | 4 | Gold facts not in top-5 chunks at all |
| **Retrieval ranking** | syn004, syn011, syn021 | 3 | Relevant chunk present but buried at rank 4–5 |
| **Generation hallucination** | syn001, syn003, syn007, syn013, syn014, syn020, syn022 | 7 | Model adds/distorts claims beyond context |
| **Corpus gap** | syn023 | 1 | Openbox startup info absent from corpus entirely |

**Hallucination is now the dominant failure mode** (7 of 14 failures). The chunk-size
and dedup changes resolved most retrieval-quantity issues; what remains are ranking problems
and generation quality.

---

## 5. Recommendations

### Short-term (next eval cycle)

1. **Re-rank or MMR.** Three samples fail because the relevant chunk exists but is
   ranked 4th or 5th. A lightweight cross-encoder re-ranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) or MMR diversity selection would push relevant chunks up.

2. **Repeat evals when comparing small changes.** LLM generation and judge scoring vary
   run to run; multiple evals (or fixed decoding) give a steadier read on pass rate.

3. **Corpus gap: add openbox docs.** syn023 is a genuine corpus miss — the openbox
   startup explanation doesn't exist in any ingested document. Adding it would flip one
   stable failure.

### Medium-term

4. **Stronger grounding via citation.** The 7 faithfulness failures persist despite the
   prompt upgrade. Requiring the model to cite chunk numbers (e.g. "[1]", "[2]") for each
   claim forces it to ground every statement, making unsupported claims more obvious to
   both the judge and the developer.

5. **Per-query retrieval diagnostics.** Log the retrieved chunks alongside the generation
   for every eval sample. This would let us distinguish "chunk present but unused" (generation
   problem) from "chunk absent" (retrieval problem) without re-reading full judge reasons.

6. **Chunk-size sensitivity sweep.** The syn021 regression suggests 1024 isn't universally
   better. Testing 768 and 1024 side-by-side (on samples that still fail after re-ingest)
   would find the sweet spot for this corpus.

---

## 6. Summary

The three changes moved the pass rate from **40% → 44%** (+1 sample). The gains came
primarily from **larger chunks keeping documentation sections whole** and **dedup freeing
retrieval slots**. The grounding prompt improved answer focus (AR +0.063) but didn't
eliminate hallucination, which is now the #1 failure mode.

The next highest-leverage change is **re-ranking** to fix the 3 samples where the right
chunk is retrieved but buried, followed by **corpus coverage** for the one genuine gap
(syn023/openbox).
