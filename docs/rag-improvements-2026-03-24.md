# RAG Pipeline Improvements Plan — 2026-03-24

**Baseline:** 40% pass rate (10/25) on synthetic dataset with GPT-4o-mini judge
(see [judge-comparison-2026-03-24.md](judge-comparison-2026-03-24.md))

**Goal:** Close the gap on the 15 failing samples by addressing root causes in the
pipeline — not by tuning the judge.

---

## 1. Chunking Strategy

### Current state

`RecursiveChunker` with `chunk_size=512`, `chunk_overlap=64`. For markdown, splits
on `## `, `### `, `\n\n`, `\n`, ` `. For code, splits on `class`, `def`, `\n\n`.

This is the standard LangChain-style recursive character splitter[^langchain-splitter]
and a reasonable phase-1 baseline. The problems it causes in the eval:

- **Mid-sentence chunk boundaries.** syn003's retrieval node 3 started mid-sentence
  because a section longer than 512 chars was split at a `\n` fallback. The generation
  model couldn't make sense of the truncated chunk and said "I don't know" despite the
  answer being present.
- **Small context density.** 512 chars ≈ 100–130 tokens per chunk. With `top_k=5`,
  the generator sees ~2,560 chars of total context. Multi-fact queries (syn002, syn006)
  need information spread across sections that may not all fit in 5 small chunks.

### What the research says about chunk size

The empirical consensus across 2025–2026 benchmarks:

- **General documents:** 512–1024 tokens optimal (Ailog RAG benchmarks[^ailog-rag];
  LangChain semi-structured eval[^langchain-bench])
- **Technical documentation:** 1024–2048 tokens with 100–200 token overlap
  (Ailog RAG benchmarks[^ailog-rag])
- **Code:** Line-based chunking with 32–64 line chunks matches syntax-aware splitting
  at small compute budgets; whole-file retrieval becomes competitive at 16K tokens
  (OpenReview code retrieval study[^openreview-code])
- **Overlap:** 10–20% of chunk size improves recall by up to 14.5% for dense retrievers
  (Firecrawl benchmarks[^firecrawl-chunking])

LangChain's own `RecursiveCharacterTextSplitter` defaults to `chunk_size=4000`
with `chunk_overlap=200`[^langchain-splitter] — nearly 8× our current setting.

**Key insight:** For our markdown documentation corpus, 512 chars is undersized. Most
documentation sections are 500–2,000 chars. A 1024-char chunk keeps more sections whole,
reducing mid-sentence splits and increasing context density per retrieval slot.

### Immediate change (this phase)

Bump markdown `chunk_size` to **1024** with `chunk_overlap` to **128** (12.5% overlap).
This is a single YAML change + re-ingest — no code changes needed.

### File-type-aware chunking libraries (future phases)

Several production-ready libraries provide smarter chunking by file type:

| Library | Approach | Languages | Notes |
|---------|----------|-----------|-------|
| **chonkie**[^chonkie] | Multi-strategy (token, sentence, recursive, semantic, code) | Python | Lightweight (9.7 MB), includes `CodeChunker` using tree-sitter AST parsing. Good candidate for a drop-in upgrade. |
| **treesitter-chunker**[^treesitter-chunker] | AST-aware semantic code chunking | 36+ languages via tree-sitter | Production-ready (95% test coverage), token-aware, exports to PostgreSQL/Neo4j. Python 3.11+. |
| **code-chunk**[^code-chunk] | AST-aware with rich contextual metadata | TS/JS, Python, Rust, Go, Java | Scope chains, imports, sibling signatures in each chunk. TypeScript library (would need a subprocess bridge or port). |
| **LlamaIndex `SimpleFileNodeParser`**[^llamaindex-filenodeparser] | Maps file types to specialised parsers | HTML, markdown, JSON, etc. | Part of a larger framework; heavy dependency for just chunking. |
| **LangChain `MarkdownHeaderTextSplitter`**[^langchain-md-splitter] | Structure-based markdown splitting on headers | Markdown only | Preserves header metadata; meant to be composed with a secondary size-based splitter. |

**Recommendation for this project:** chonkie is the strongest candidate for a future
phase upgrade. It's Python-native, lightweight, and its `CodeChunker` uses tree-sitter
under the hood — giving us AST-aware code chunking without pulling in a full framework
like LlamaIndex. Its `SemanticChunker` would also cover the "semantic chunking" phase-3
placeholder already in our config.

### Tree-sitter / LSP chunking for code (future phase)

Tree-sitter parses source code into an Abstract Syntax Tree and splits at semantic
boundaries — function definitions, class declarations, method bodies — instead of
character limits[^code-chunk-blog]. This means:

- Functions never get cut in half
- Each chunk carries its class/scope context
- Import statements can be attached to every chunk from the same file

The `treesitter-chunker` library[^treesitter-chunker] and chonkie's
`CodeChunker`[^chonkie] both implement this pattern in Python. The
OpenReview code retrieval study[^openreview-code] found that for
natural-language-to-code queries (our use case when someone asks "how does X work?"),
dense vector search with AST-aware chunks outperforms naive line-based splitting.

For our repo: we already have `DocumentType.CODE` with function/class separators. The
upgrade path is to replace those regex-based separators with actual tree-sitter parsing.
This belongs in phase 3 alongside semantic chunking.

---

## 2. Query-Time Deduplication

### Problem

syn004 returned 4 near-identical `systemctl status` chunks from different source files,
wasting 4 of 5 retrieval slots. Corpus curation (excluding `logs/`, `plan.md`) has
already been applied to address the root cause, but cross-file content duplication can
still occur in a mixed corpus (e.g., a troubleshooting guide and an install guide both
quoting the same command output).

### Solution

Add text-hash deduplication in `retriever.py` after the vector store query, before
returning results to the generator. This is the standard production
pattern[^rag-production-guide] — the store's job is to find similar vectors; the
retriever curates what the generator sees.

To compensate for discarded duplicates, over-fetch from the store (e.g., `2 * top_k`)
and return up to `top_k` unique chunks. This ensures the generator still gets a full
context window even after dedup removes near-duplicates.

### Implementation

```python
def retrieve(self, query: str) -> list[RetrievedChunk]:
    # Over-fetch to compensate for duplicates discarded by dedup
    results = self.store.query(query, top_k=self.top_k * 2)
    results = self._deduplicate(results)
    if self.score_threshold is not None:
        results = [r for r in results if r.score >= self.score_threshold]
    return results[: self.top_k]
```

---

## 3. System Prompt Upgrade

### Problem

The current prompt says "always ground your answers in the retrieved context" but doesn't
explicitly prohibit adding outside knowledge. Six samples failed faithfulness because
the model supplemented the context with pre-training knowledge (syn007, syn011, syn012,
syn017, syn018, syn025).

### Research

AWS prescriptive guidance[^aws-rag-prompts] and Stack AI's RAG prompt engineering
guide[^stackai-prompts] both recommend explicit grounding constraints:

- "Do not add information not in the context"
- "If the answer is not in the context, say you don't know"
- "No speculation; cite which source supports each claim"

Gloo's RAG generation guide[^gloo-generation] defines a faithful response as one that
"doesn't add information that wasn't in the context, doesn't contradict the context,
and represents the context's meaning accurately."

The key is keeping the grounding directive concise. Rephrase's prompt design
guide[^rephrase-prompts] warns that long, cluttered prompts dilute instructions and
degrade utilisation of retrieved information.

### New prompt

```
You are a technical assistant. Answer questions using ONLY the provided context.
Do not add information from your own knowledge — if the context does not contain
enough information, say what is missing rather than guessing.

Context:
{context}
```

Changes from the current prompt:
- **"ONLY the provided context"** — explicit prohibition vs soft "always ground"
- **"Do not add information from your own knowledge"** — targets the hallucination pattern
- **"say what is missing rather than guessing"** — guides useful abstention
- Removed `Question: {query}` from the system prompt (the query is already sent as the
  user message; having it in both places is redundant)

---

## 4. Top-K: Keep at 5

The report previously suggested bumping `top_k` to 7–8. On further analysis, this is
a band-aid for small chunks. With `chunk_size` going from 512 → 1024, each retrieval
slot carries roughly twice the information. Five slots of 1024-char chunks gives ~5,120
chars of context — equivalent to the old 10-slot budget. Re-evaluate after the chunking
change; bump only if retrieval misses persist.

---

## Summary of Changes

| Change | Type | Files |
|--------|------|-------|
| Markdown `chunk_size` 512 → 1024, overlap 64 → 128 | Config | `configs/default.yaml` |
| Query-time text-hash dedup in retriever | Code | `src/rag/retriever.py` |
| System prompt grounding upgrade | Config + code | `configs/default.yaml`, `src/rag/generator.py` |
| Phase roadmap update (chunking & dedup milestones) | Docs | `README.md` |

After these changes: re-ingest the corpus, re-run the 25-sample eval with GPT-4o-mini
judge, and compare against the 40% baseline.

---

## References

[^langchain-splitter]: [LangChain RecursiveCharacterTextSplitter](https://python.langchain.com/v0.2/api_reference/text_splitters/) — default `chunk_size=4000`, `chunk_overlap=200`
[^langchain-bench]: [LangChain Semi-structured Eval: Chunk Size Tuning](https://langchain-ai.github.io/langchain-benchmarks/notebooks/retrieval/semi_structured_benchmarking/ss_eval_chunk_sizes.html)
[^langchain-md-splitter]: [LangChain MarkdownHeaderTextSplitter](https://python.langchain.com/v0.2/api_reference/text_splitters/markdown/langchain_text_splitters.markdown.MarkdownHeaderTextSplitter.html) — structure-based splitting on markdown headers
[^ailog-rag]: [RAG Chunking Strategies 2025: Optimal Chunk Sizes & Techniques](https://app.ailog.fr/en/blog/guides/chunking-strategies) — 512–1024 tokens for general docs, 1024–2048 for technical docs
[^openreview-code]: [OpenReview: Code Retrieval Chunking Study](https://openreview.net/pdf?id=twV78Ytnve) — line-based chunking matches syntax-aware for PL→PL; dense encoders win for NL→PL
[^firecrawl-chunking]: [Best Chunking Strategies for RAG in 2026](https://www.firecrawl.dev/blog/best-chunking-strategies-rag) — 10–20% overlap improves recall by up to 14.5%
[^chonkie]: [Chonkie: Lightweight RAG Chunking Library](https://docs.chonkie.ai/getting-started/introduction) — multi-strategy chunking including `CodeChunker` (tree-sitter) and `SemanticChunker`
[^treesitter-chunker]: [treesitter-chunker on PyPI](https://pypi.org/project/treesitter-chunker/) — AST-aware code chunking, 36+ languages, Python 3.11+
[^code-chunk]: [supermemoryai/code-chunk](https://github.com/supermemoryai/code-chunk) — AST-aware chunking with scope chains and contextual metadata
[^code-chunk-blog]: [Building code-chunk: AST Aware Code Chunking](https://supermemory.ai/blog/building-code-chunk-ast-aware-code-chunking/) — five-step parse → extract → scope → chunk → enrich pipeline
[^llamaindex-filenodeparser]: [LlamaIndex File Based Node Parsers](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/FileNodeProcessors/) — `SimpleFileNodeParser` maps file types to specialised parsers
[^rag-production-guide]: [RAG Pipeline Production Guide](https://www.youngju.dev/blog/llm/2026-03-11-rag-pipeline-vector-database-production.en) — context assembly stage handles dedup and token limit management
[^aws-rag-prompts]: [Writing Best Practices to Optimize RAG Applications](https://docs.aws.amazon.com/prescriptive-guidance/latest/writing-best-practices-rag/introduction.html) — explicit grounding instructions for faithfulness
[^stackai-prompts]: [Prompt Engineering for RAG Pipelines (2026)](https://www.stack-ai.com/blog/prompt-engineering-for-rag-pipelines-the-complete-guide-to-prompt-engineering-for-retrieval-augmented-generation) — system prompt should define refusal patterns and grounding directives
[^gloo-generation]: [The Generation Step — Gloo](https://docs.gloo.com/ai-learning-center/gloo-ai-103/the-generation-step) — faithful RAG responses don't add, contradict, or embellish context
[^rephrase-prompts]: [Prompt Design for RAG Systems — Rephrase](https://rephrase-it.com/blog/prompt-design-for-rag-systems-what-goes-in-the-prompt-vs-wha) — concise grounding directives; long prompts dilute instructions
