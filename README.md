# RAG Evaluation System

A local-first RAG system with **MLflow-backed** evaluation: `mlflow.genai.evaluate()` drives batched runs, **DeepEval scorers** (judge-backed) score each sample, and **traced** retriever / LLM spans capture context for metrics and debugging.

Designed for iterative improvement — start simple, measure everything, upgrade methodically.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RAG Pipeline                          │
│                                                         │
│  Documents ──► Chunker ──► Embedder ──► Vector Store    │
│                               │                         │
│  Query ──► Embed Query ───────┤  (optional BM25 hybrid) │
│                               ▼                         │
│                          Retriever ──► Generator ──► Out │
└─────────────────────────────────────────────────────────┘
        │                                        │
        ▼                                        ▼
┌─────────────────────────────────────────────────────────┐
│           Evaluation (MLflow + DeepEval scorers)         │
│                                                         │
│  scripts/run_eval.py  ──►  mlflow.genai.evaluate()     │
│    • Faithfulness, answer relevancy                     │
│    • Contextual precision / recall                        │
│    • Logs: pass_rate, traces (RETRIEVER, LLM), UI       │
│                                                         │
│  Tracking: sqlite:///data/mlflow.db  →  mlflow ui       │
└─────────────────────────────────────────────────────────┘
```

Agent workflows (skills / commands) are mapped in [docs/ARCHITECTURE-2026-03-25T103050Z.md](docs/ARCHITECTURE-2026-03-25T103050Z.md).

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
#   lms load mistralai/devstral-small-2-2512 -y
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

# 4. Run a standalone MLflow-backed eval
.venv/bin/python scripts/run_eval.py --dataset data/eval_datasets/starter.json

# 5. Launch MLflow UI (in another terminal)
# - opens local run browser for metrics, traces, and tags
# - default URL: http://127.0.0.1:5000
.venv/bin/mlflow ui --backend-store-uri sqlite:///data/mlflow.db

```

## MLflow Tracking UI

After running `scripts/run_eval.py` or `scripts/run_ab_study.py`, open the local MLflow UI:

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///data/mlflow.db
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) and inspect:

- run-level metrics (including `pass_rate`)
- DeepEval-derived metric results (faithfulness, answer relevancy, contextual precision/recall)
- trace spans (`RETRIEVER`, `LLM`) for sample-level debugging
- tags such as `study_id`, `arm`, and retrieval mode flags

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
│   ├── eval/                 # Eval adapters for MLflow (no local report/registry)
│   │   ├── datasets.py       # JSON datasets → MLflow eval rows
│   │   ├── scorers.py        # DeepEval scorers from configs/eval.yaml
│   │   ├── eval_config.py    # Enabled dataset paths from YAML
│   │   └── ab_study_config.py # Temporary YAML overrides for A/B arms
│   └── data/                 # Data utilities
│       ├── __init__.py
│       ├── loaders.py        # MD, code, mermaid file loaders
│       └── schemas.py        # Pydantic models for eval data
│
├── tests/                    # DeepEval test suites
│   ├── conftest.py           # Shared fixtures (pipeline, datasets)
│   ├── test_retrieval.py     # Retriever-focused evals
│   ├── test_retrieval_tracing.py  # MLflow trace verification
│   ├── test_generation.py    # Generator-focused evals
│   ├── test_scorers.py       # Scorer construction tests
│   ├── test_mlflow_datasets.py    # Dataset adapter tests
│   ├── test_eval_config.py   # Eval config loading tests
│   └── test_ab_study_config.py    # A/B study YAML override tests
│
├── configs/
│   ├── default.yaml          # Default pipeline config
│   └── eval.yaml             # Eval-specific settings
│
├── data/
│   ├── corpus/               # Source documents (MD, code, mermaid)
│   ├── eval_datasets/        # Ground truth Q&A pairs (JSON/CSV)
│
├── docs/
│   ├── ARCHITECTURE-*.md     # Snapshot diagrams (incl. MLflow + .cursor skills)
│   ├── mlflow-eval-tracking.md  # MLflow tracking quick reference
│   └── DECISIONS.md          # Architecture decision log
│
├── scripts/
│   ├── setup_lm_studio.sh    # LM Studio server + load-model hints
│   ├── run_eval.py           # Standalone eval → MLflow
│   └── run_ab_study.py       # Fixed-N two-arm studies → MLflow tags
│
├── pyproject.toml
└── README.md
```

## Judge Provider (Local or Cloud)

DeepEval judge calls are configured in `configs/eval.yaml` under `judge`. The `judge.model`
field uses MLflow's model URI format — no separate `provider` field.

- Cloud judge (current default):
  - `judge.model: "openai:/gpt-4o"`
  - `judge.openai.base_url: "https://api.openai.com/v1"`
  - `judge.openai.api_key_env: "OPENAI_API_KEY"`
- Local judge (via LM Studio):
  - set `judge.model` to an MLflow URI matching your loaded model, e.g. `"openai:/mistralai/devstral-small-2-2512"`
  - set `judge.openai.base_url` to `"http://localhost:1234/v1"`
  - set `judge.openai.api_key` to `"lm-studio"`

Judge outputs must be valid for DeepEval's native parsing path. This repo intentionally does
not apply model-specific sanitizers or JSON compatibility shims; if a judge model emits
non-compliant output, treat it as unsupported and switch to a more reliable judge model.

Example env setup for cloud judge:

```bash
export OPENAI_API_KEY="your-key"
# Eval runs log directly to MLflow; expect many minutes even on the starter dataset.
.venv/bin/python scripts/run_eval.py --dataset data/eval_datasets/starter.json
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
- [ ] Visualization layer (deferred): add a GitNexus-style explorer for repo/RAG debugging ([GitNexus](https://github.com/abhigyanpatwari/GitNexus))

### Phase 2 — Evaluation & Pipeline Tuning (completed)
- [x] MLflow-native batch eval (`run_eval.py`, traces, DeepEval scorers; see `docs/mlflow-eval-tracking.md`)
- [x] Synthetic eval data generation (corpus-grounded Q&A pairs)
- [x] Judge model comparison (devstral-small vs GPT-4o-mini; upgraded default to GPT-4o)
- [x] System prompt grounding upgrade (explicit faithfulness constraints)
- [x] Chunk size tuning (512 → 1024 for markdown, see `docs/rag-improvements-2026-03-24.md`)
- [x] Query-time retrieval deduplication (text-hash in retriever)
- [x] Legacy eval registry and report pipeline removed (replaced by MLflow UI + run tags)
- [x] Fixed-N A/B study orchestration via MLflow-tagged runs (`run_ab_study.py`)
- [ ] Custom G-Eval metrics for code/diagram accuracy
- [ ] Arize Phoenix integration for cluster analysis
- [ ] Annotation workflow for expanding eval datasets
- [ ] CI/CD eval pipeline (pytest + deepeval)

### Phase 3 — Chunking & Retrieval Upgrades
- [ ] Tree-sitter AST-aware code chunking (candidates: chonkie `CodeChunker`, `treesitter-chunker`)
- [ ] Semantic chunking for prose documents (chonkie `SemanticChunker` or LlamaIndex)
- [ ] Section-aware markdown chunking (heading-preserving, sub-chunk header prepend)
- [x] Hybrid search (dense + BM25 sparse, RRF; `retrieval.hybrid` in `configs/default.yaml`)
- [ ] HyDE (Hypothetical Document Embeddings)
- [ ] Cross-encoder reranking
- [ ] Merkle tree content hashing for incremental updates

### Phase 4 — Advanced Architectures
- [ ] Parent-child (small-to-big) retrieval
- [ ] Graph RAG (knowledge graph extraction)
- [ ] RAPTOR (recursive abstractive processing)
- [ ] Agentic RAG (tool-use, multi-step retrieval)
- [ ] Multi-modal support (diagrams as images)
