# Eval Findings: Synthetic Dataset — 2026-03-24

**Run:** `artifacts/eval_runs/20260324T085932Z/single/`
**Dataset:** `data/eval_datasets/synthetic.json` (25 samples)
**Pass rate:** 32% (8/25)
**Judge & generation model:** `mistralai/devstral-small-2-2512` (same model for both)

## Aggregate Scores

| Metric | Mean | Threshold |
|--------|------|-----------|
| Faithfulness | 0.757 | 0.70 |
| Answer Relevancy | 0.822 | 0.70 |
| Contextual Precision | 0.770 | 0.60 |
| Contextual Recall | 0.657 | 0.60 |

Contextual Recall is the weakest metric and is the primary driver of failures.

---

## Part 1: Judge Quality Issues

The devstral judge is producing incorrect or self-contradictory scores in at least 8 of 25
samples. This is significant enough to invalidate the 32% pass rate as a reliable baseline.

### 1. Faithfulness of abstention answers scored inconsistently

When the generation model correctly says "the context doesn't have this info," the judge
scores it differently across samples:

| Sample | Answer gist | Faithfulness | Problem |
|--------|-------------|-------------|---------|
| syn003 | "The context does not explicitly state..." | **0.00** | Claims output "fails to acknowledge" context gap — but it literally does |
| syn004 | "The context does not contain explicit info..." | **1.00** | "No contradictions found" |

These are structurally identical abstention answers. Devstral is not reliably reasoning
about what faithfulness means for refusals.

### 2. Cross-metric contradictions within the same sample

**syn003** is the clearest case:
- **Faithfulness 0.0** — judge says answer doesn't acknowledge the context gap
- **Answer Relevancy 0.83** — judge says answer "correctly answers the main question"

A refusal answer cannot correctly answer the question AND fail to acknowledge the context
gap simultaneously. The judge gave it the worst of both worlds.

### 3. Contextual Recall missing obvious chunk matches

**syn003** — Contextual Recall = 0.0, but retrieval node 3 literally contains:

> After a successful install, `/etc/streamdeck/install.conf` is created (mode 644) with:
> STREAM_USER, BUDDY_USER, STREAM_DISPLAY, STREAM_GROUP

This is a verbatim match for the expected answer. The judge scored it 0.0 claiming "none
of the sentences in the expected output are supported by any node." Clear judge
hallucination.

### 4. Answer Relevancy penalising correct responses

**syn022** — Question asks *"what value must it be set to?"* The answer says the value is
`false`. Judge docks it to 0.67: "includes unnecessary details about setting the field to
false" — but that's exactly what was asked.

**syn008** — Answer Relevancy = 0.50. Judge says the response "incorrectly references...
the prefix 'moondeck'" but the expected answer itself says *"Logs are stored under /tmp,
prefixed moondeck..."*

### 5. Faithfulness misreading the generated text

**syn001** — Generated answer: *"The context does not explicitly state which groups the
STREAM_USER is added to."* Judge penalises faithfulness to 0.50 claiming the output
*"incorrectly implies that the retrieval context specifies which groups"* — the exact
opposite of what the answer says.

**syn017** — Faithfulness 0.67 because judge says `TAG-="uaccess"` "contradicts the
retrieval context." The judge does not understand udev syntax — `TAG-=` is the removal
operator, not a contradiction.

### Summary: Judge-suspect samples

| Sample | Suspect metric(s) | Issue type |
|--------|-------------------|-----------|
| syn001 | Faithfulness | Misread generated answer |
| syn003 | Faithfulness, Contextual Recall, cross-metric | Multiple errors |
| syn008 | Answer Relevancy | Penalises correct content |
| syn017 | Faithfulness | Domain syntax confusion (udev) |
| syn022 | Answer Relevancy | Penalises asked-for detail |
| syn025 | Faithfulness | Grep pattern terms called hallucination |

**Root cause:** DeepEval's metrics require the judge to decompose answers into claims and
verify each against the context (NLI-style reasoning). Devstral-small is a code-focused
model and underperforms on the nuanced multi-hop reasoning that faithfulness and recall
scoring demand. The cross-metric contradictions suggest the judge isn't maintaining a
coherent understanding of the same answer across different metric prompts.

---

## Part 2: Genuine RAG Issues (discount judge noise first)

### Retrieval gaps — Contextual Recall is the weakest metric (0.657 mean)

Several queries retrieve thematically adjacent chunks but miss the specific facts:

| Sample | Missing from retrieval | Likely cause |
|--------|----------------------|-------------|
| syn002 | `SKIP_DEPS=1` variable | Chunk not in top-5 |
| syn004 | Unit file `User=`, `Requires=`, `After=` directives | Retrieved 4× near-duplicate `systemctl status` blobs instead of the unit template |
| syn006 | `sunshine.conf.template` capture/encoder/log values | Chunk not surfaced |
| syn023 | Why openbox is started, `ExecStartPost` command | 0.0 precision + 0.0 recall = total miss |
| syn013 | Full wrapper script 6-step sequence | Only retrieved the SIGTERM stale-PID step |

The syn004 case is notable: retrieval returned 4 near-duplicate `systemctl status` outputs
that all confirm the service is running, but none contain the unit file directives
(`User=`, `Requires=`, `After=`). This is a chunking/dedup issue wasting the top_k budget.

### Generation model not using retrieved context

**syn003**: The correct facts were in retrieval node 3 (`/etc/streamdeck/install.conf`,
mode 644, all four variables). The model responded "I don't know." Two likely causes:
- The chunk starts mid-sentence ("id for Sunshine NVENC...") — truncated chunk boundaries
  hurt both retrieval relevance and generation comprehension
- Devstral may not read all 5 chunks carefully when the relevant one is buried

### Passing samples cluster around self-contained, well-chunked topics

The 8 passing samples (syn005, 009, 010, 011, 014, 019, 021, 024) all involve topics
where the corpus has a dedicated section with the exact facts needed. Failing samples
involve facts spread across documents or embedded in less prominent sections.

---

## Recommendations

### Priority 1: Upgrade the judge model

The largest lever for **eval trustworthiness** before any RAG investment. Options:
- Load a larger model in LM Studio (70B+) for judging only
- Use a cloud API judge (Claude, GPT-4) while keeping devstral for generation — gives a
  reliable measure of the local generation pipeline

Until the judge is reliable, the pass rate is not a meaningful signal.

### Priority 2: Chunk deduplication

syn004 demonstrates the problem: 4 near-identical `systemctl status` blobs consume 4 of
the 5 retrieval slots. Consider:
- Dedup at ingest time (cosine similarity threshold between chunks from the same document)
- Re-ranker that penalises near-duplicate results at query time

### Priority 3: Re-evaluate after judge fix

Re-run the same 25 samples with a reliable judge before changing the RAG pipeline. The
true pass rate is likely higher than 32% — several failures (syn001, syn003, syn008,
syn017, syn022, syn025) may flip to passes or near-passes. This establishes a clean
baseline.

### Priority 4: RAG improvements (only after clean baseline)

- **Retrieval gaps** (syn002, syn006, syn023): verify the facts are in the corpus at all;
  if so, investigate chunk boundary placement or embedding quality
- **top_k**: current `top_k: 5` with `chunk_size: 512` gives ~2,560 tokens of context;
  bumping to 7–8 may improve multi-fact coverage
- **Truncated chunk starts**: chunks starting mid-sentence (visible in syn003 node 3)
  hurt retrieval scoring and confuse the generation model — review splitter separator
  configuration to prefer splitting at section boundaries
