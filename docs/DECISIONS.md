# Architecture Decision Log

## ADR-001: DeepEval over RAGAS for evaluation framework

**Date:** 2026-03-23  
**Status:** Accepted

**Context:**  
We need an evaluation framework for our local RAG system that supports custom metrics,
works with local LLM judges (Ollama), and can grow from simple RAG evals into agentic
RAG evaluation as the system evolves.

**Decision:**  
Use DeepEval as the primary evaluation framework.

**Rationale:**
- Native Ollama integration via `deepeval set-ollama` — critical for fully local operation
- Pytest-compatible test runner integrates naturally with CI/CD
- G-Eval allows custom metric criteria in plain language (needed for code/diagram eval)
- Framework-agnostic — no LangChain/LlamaIndex lock-in
- Can wrap RAGAS metrics if needed (`deepeval.metrics.ragas`)
- Roadmap includes agentic eval metrics (task completion, tool correctness)
- JSON confinement mechanisms handle weaker local LLM judges

**Consequences:**
- LLM-as-judge metrics require a capable local model (≥14B params recommended)
- Metric computation costs scale with dataset size (each metric = 1+ LLM call per sample)
- Custom prompt templates may need tuning for the specific Ollama model used

---

## ADR-002: nomic-embed-text as default embedding model

**Date:** 2026-03-23  
**Status:** Accepted

**Context:**  
Need a local embedding model that fits in 24 GB VRAM alongside a generation model.

**Decision:**  
Use nomic-embed-text (137M params, 768d, 8K context) via Ollama.

**Rationale:**
- Only ~500MB memory — leaves plenty of VRAM for generation model
- 8,192 token context handles large chunks
- Requires task-specific prefixes ("search_query:", "search_document:") for optimal results
- Good baseline; can swap to bge-m3 (568M) if multilingual or higher accuracy needed

**Alternatives considered:**
- bge-m3: Higher accuracy, multilingual, but 568M params and 1024d vectors = more storage
- mxbai-embed-large: Strong MTEB scores but 512-token context limit
- stella_en_1.5B: Highest MTEB but 1.5B params competes for VRAM

---

## ADR-003: ChromaDB as vector store

**Date:** 2026-03-23  
**Status:** Accepted

**Context:**  
Need a vector store for the prototype phase. Must work locally, persist to disk,
and be simple to set up.

**Decision:**  
Use ChromaDB with persistent storage.

**Rationale:**
- Zero infrastructure — just a Python library with on-disk persistence
- Native cosine similarity search
- Good enough for prototype; can migrate to Milvus/Qdrant/pgvector for production
- Embedding storage and metadata filtering built in

---

## ADR-004: Phased approach to RAG upgrades

**Date:** 2026-03-23  
**Status:** Accepted

**Context:**  
Many advanced RAG techniques exist (HyDE, hybrid search, Graph RAG, RAPTOR, agentic RAG).
We need to avoid premature complexity while building toward them.

**Decision:**  
Ship the simplest possible RAG pipeline first, with evaluation infrastructure
that lets us measure the impact of each upgrade.

**Rationale:**
- You can't improve what you can't measure
- Each upgrade should be a measured experiment: baseline → change → re-evaluate
- The eval infrastructure (DeepEval + UMAP viz) is the real foundation
- Simple cosine similarity retrieval is a valid baseline for comparison

**Upgrade order:**
1. Hybrid search (BM25 + dense) — highest ROI, addresses keyword matching gaps
2. Cross-encoder reranking — improves precision with minimal architecture change
3. HyDE — helps with vocabulary mismatch between queries and documents
4. Graph RAG / RAPTOR / Agentic — fundamentally different retrieval paradigms
