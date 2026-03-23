# RAG Evaluation System

A local-first RAG system with evaluation infrastructure built on DeepEval.
Designed for iterative improvement — start simple, measure everything, upgrade methodically.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RAG Pipeline                          │
│                                                         │
│  Documents ──► Chunker ──► Embedder ──► Vector Store    │
│                               │                         │
│  Query ──► Embed Query ───────┤                         │
│                               ▼                         │
│                          Retriever ──► Generator ──► Out │
└─────────────────────────────────────────────────────────┘
        │                                        │
        ▼                                        ▼
┌─────────────────────────────────────────────────────────┐
│                  Evaluation Layer                        │
│                                                         │
│  DeepEval Metrics:                                      │
│    • Faithfulness    • Answer Relevancy                 │
│    • Context Recall  • Context Precision                │
│    • Custom G-Eval metrics (your annotations)           │
│                                                         │
│  Visualization:                                         │
│    • UMAP 3D embedding plots                            │
│    • Retrieval overlap heatmaps                         │
│    • Per-metric cluster analysis (via Phoenix)          │
└─────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.11+
- 24 GB VRAM GPU (tested on RTX 3090/4090)
- Ollama (for local embedding + LLM judge)

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Pull Ollama models
ollama pull nomic-embed-text        # embedding model (~500MB)
ollama pull qwen3:14b               # generation model
ollama pull qwen3:14b               # also used as eval judge

# 3. Configure DeepEval to use local Ollama judge
deepeval set-ollama --model=qwen3:14b

# 4. Index the sample corpus
python -m src.rag.ingest --corpus-dir data/corpus

# 5. Run the eval suite
deepeval test run tests/

# 6. Launch visualization
python -m src.viz.embedding_explorer
```

## Project Structure

```
rag-eval-system/
├── src/
│   ├── rag/                  # RAG pipeline components
│   │   ├── __init__.py
│   │   ├── chunker.py        # Document splitting strategies
│   │   ├── embedder.py       # Local embedding via Ollama
│   │   ├── store.py          # ChromaDB vector store
│   │   ├── retriever.py      # Similarity search + retrieval
│   │   ├── generator.py      # LLM response generation
│   │   ├── pipeline.py       # End-to-end RAG orchestration
│   │   └── ingest.py         # Corpus ingestion CLI
│   │
│   ├── eval/                 # Evaluation infrastructure
│   │   ├── __init__.py
│   │   ├── metrics.py        # Custom DeepEval metrics
│   │   ├── datasets.py       # Eval dataset loading/management
│   │   ├── runner.py         # Evaluation orchestration
│   │   └── report.py         # Results aggregation & export
│   │
│   ├── viz/                  # Visualization tools
│   │   ├── __init__.py
│   │   ├── embedding_explorer.py  # UMAP 3D scatter plots
│   │   └── eval_dashboard.py      # Metric visualization
│   │
│   └── data/                 # Data utilities
│       ├── __init__.py
│       ├── loaders.py        # MD, code, mermaid file loaders
│       └── schemas.py        # Pydantic models for eval data
│
├── tests/                    # DeepEval test suites
│   ├── conftest.py           # Shared fixtures (pipeline, datasets)
│   ├── test_retrieval.py     # Retriever-focused evals
│   ├── test_generation.py    # Generator-focused evals
│   └── test_e2e.py           # End-to-end RAG evals
│
├── configs/
│   ├── default.yaml          # Default pipeline config
│   └── eval.yaml             # Eval-specific settings
│
├── data/
│   ├── corpus/               # Source documents (MD, code, mermaid)
│   ├── eval_datasets/        # Ground truth Q&A pairs (JSON/CSV)
│   └── embeddings/           # Cached embeddings for viz
│
├── docs/
│   └── DECISIONS.md          # Architecture decision log
│
├── scripts/
│   ├── setup_ollama.sh       # Ollama model pull script
│   └── generate_eval_data.py # Synthetic eval data generation
│
├── pyproject.toml
└── README.md
```

## Phase Roadmap

### Phase 1 — Foundation (current)
- [x] Project scaffold
- [ ] Simple chunker (recursive text splitting)
- [ ] Local embeddings via Ollama (nomic-embed-text)
- [ ] ChromaDB vector store
- [ ] Basic cosine similarity retrieval
- [ ] Ollama-based generation (qwen3:14b)
- [ ] DeepEval integration with Ollama judge
- [ ] Starter eval dataset (hand-written, ~20-30 Q&A pairs)
- [ ] UMAP embedding visualization

### Phase 2 — Evaluation Depth
- [ ] Custom G-Eval metrics for code/diagram accuracy
- [ ] Synthetic eval data generation (DeepEval synthesizer)
- [ ] Arize Phoenix integration for cluster analysis
- [ ] Annotation workflow for expanding eval datasets
- [ ] CI/CD eval pipeline (pytest + deepeval)

### Phase 3 — Retrieval Upgrades
- [ ] Hybrid search (dense + BM25 sparse)
- [ ] HyDE (Hypothetical Document Embeddings)
- [ ] Cross-encoder reranking
- [ ] Merkle tree content hashing for incremental updates

### Phase 4 — Advanced Architectures
- [ ] Graph RAG (knowledge graph extraction)
- [ ] RAPTOR (recursive abstractive processing)
- [ ] Agentic RAG (tool-use, multi-step retrieval)
- [ ] Multi-modal support (diagrams as images)
