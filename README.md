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
- [LM Studio](https://lmstudio.ai/) with the `lms` CLI (ships with the app; run the GUI at least once first)

## Quick Start

```bash
# 0. LM Studio: start server and load models (ids must match configs/default.yaml)
bash scripts/setup_lm_studio.sh
# Then load embedding + chat models, e.g.:
#   lms load text-embedding-nomic-embed-text-v1.5 -y
#   lms load qwen/qwen3-14b -y
#   lms ps

# 1. Install dependencies
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
# Prefer .venv/bin/python (and .venv/bin/pip) so commands work without activating the venv,
# including in automation where `source .venv/bin/activate` may not run.

# 2. Index the sample corpus (embedding model must be loaded in LM Studio)
.venv/bin/python -m src.rag.ingest --corpus-dir data/corpus

# 3. Run eval tests
# - generation model must be loaded in LM Studio
# - judge provider is configured in configs/eval.yaml
.venv/bin/python -m pytest -m eval

# 4. Launch visualization
.venv/bin/python -m src.viz.embedding_explorer
```

## Project Structure

```
rag-eval-system/
├── src/
│   ├── inference/            # LM Studio OpenAI-compatible client config
│   │   └── lm_studio.py
│   ├── rag/                  # RAG pipeline components
│   │   ├── __init__.py
│   │   ├── chunker.py        # Document splitting strategies
│   │   ├── embedder.py       # Embeddings via LM Studio (/v1/embeddings)
│   │   ├── store.py          # ChromaDB vector store
│   │   ├── retriever.py      # Similarity search + retrieval
│   │   ├── generator.py      # Chat completions via LM Studio (/v1/chat/completions)
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
│   ├── setup_lm_studio.sh    # LM Studio server + load-model hints
│   └── generate_eval_data.py # Synthetic eval data generation
│
├── pyproject.toml
└── README.md
```

## Judge Provider (Local or Cloud)

DeepEval judge calls are configured in `configs/eval.yaml` under `judge`.

- Cloud judge (current default):
  - `judge.provider: "openai"`
  - `judge.model: "gpt-4o-mini"`
  - `judge.openai.base_url: "https://api.openai.com/v1"`
  - `judge.openai.api_key_env: "OPENAI_API_KEY"`
- Local judge (switch back later):
  - set `judge.provider: "local"`
  - set `judge.model` to your loaded LM Studio judge model id

Judge outputs must be valid for DeepEval's native parsing path. This repo intentionally does
not apply model-specific sanitizers or JSON compatibility shims; if a judge model emits
non-compliant output, treat it as unsupported and switch to a more reliable judge model.

Example env setup for cloud judge:

```bash
export OPENAI_API_KEY="your-key"
# Batch reports call the judge many times; expect many minutes even on the starter dataset.
.venv/bin/python scripts/run_eval_report.py --dataset data/eval_datasets/starter.json
```

If your cloud judge provider rate-limits, start conservative (slower wall clock, fewer 429s):

```bash
.venv/bin/python scripts/run_eval_report.py \
  --dataset data/eval_datasets/starter.json \
  --max-concurrent 1 \
  --judge-throttle-seconds 5
```

## Phase Roadmap

### Phase 1 — Foundation ✓
- [x] Project scaffold
- [x] Simple chunker (recursive text splitting)
- [x] Local embeddings via LM Studio (Nomic embed)
- [x] ChromaDB vector store
- [x] Basic cosine similarity retrieval
- [x] LM Studio chat generation
- [x] DeepEval judge via provider-configured OpenAI-compatible endpoint (local or cloud)
- [x] Starter eval dataset (hand-written Q&A pairs)
- [x] UMAP embedding visualization

### Phase 2 — Evaluation & Pipeline Tuning (current)
- [x] Synthetic eval data generation (corpus-grounded Q&A pairs)
- [x] Judge model comparison (devstral-small vs GPT-4o-mini)
- [x] System prompt grounding upgrade (explicit faithfulness constraints)
- [x] Chunk size tuning (512 → 1024 for markdown, see `docs/rag-improvements-2026-03-24.md`)
- [x] Query-time retrieval deduplication (text-hash in retriever)
- [ ] Custom G-Eval metrics for code/diagram accuracy
- [ ] Arize Phoenix integration for cluster analysis
- [ ] Annotation workflow for expanding eval datasets
- [ ] CI/CD eval pipeline (pytest + deepeval)

### Phase 3 — Chunking & Retrieval Upgrades
- [ ] Tree-sitter AST-aware code chunking (candidates: chonkie `CodeChunker`, `treesitter-chunker`)
- [ ] Semantic chunking for prose documents (chonkie `SemanticChunker` or LlamaIndex)
- [ ] Section-aware markdown chunking (heading-preserving, sub-chunk header prepend)
- [ ] Hybrid search (dense + BM25 sparse)
- [ ] HyDE (Hypothetical Document Embeddings)
- [ ] Cross-encoder reranking
- [ ] Merkle tree content hashing for incremental updates

### Phase 4 — Advanced Architectures
- [ ] Parent-child (small-to-big) retrieval
- [ ] Graph RAG (knowledge graph extraction)
- [ ] RAPTOR (recursive abstractive processing)
- [ ] Agentic RAG (tool-use, multi-step retrieval)
- [ ] Multi-modal support (diagrams as images)
