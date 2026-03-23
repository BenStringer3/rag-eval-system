# RAG Evaluation System

## Overview

This project implements a local-first Retrieval-Augmented Generation (RAG) system
with a comprehensive evaluation infrastructure built on DeepEval. The system is
designed for iterative improvement: start with a simple pipeline, measure everything,
and upgrade methodically.

## Architecture

The system consists of three main layers:

### RAG Pipeline

Documents are loaded from a corpus directory, split into chunks using recursive
character splitting with document-type-aware separators, embedded using
nomic-embed-text via Ollama, and stored in a ChromaDB vector store.

When a query arrives, it is embedded with the same model (using a query-specific
prefix), and the most similar chunks are retrieved via cosine similarity search.
These chunks are then passed as context to a local LLM (qwen3:14b) which generates
a grounded response.

### Evaluation Layer

The evaluation layer uses DeepEval with a local Ollama model as the LLM judge.
It measures four core metrics:

- **Faithfulness**: Is the answer grounded in the retrieved context?
- **Answer Relevancy**: Does the answer address the question asked?
- **Contextual Precision**: Are the retrieved chunks relevant to the query?
- **Contextual Recall**: Does the retrieved context cover the information needed?

Custom G-Eval metrics can be added for domain-specific evaluation, such as
code accuracy and diagram fidelity.

### Visualization

The visualization layer provides UMAP-based 3D projections of the embedding space,
allowing developers to inspect document clusters, query-document proximity, and
coverage gaps.

## Document Types

The system handles three primary document types:

### Markdown
Standard markdown files are split on heading boundaries (##, ###) to preserve
semantic coherence within chunks. This ensures each chunk represents a complete
section or subsection.

### Code
Code files (Python, JavaScript, TypeScript, Rust, Go, etc.) are split on
class and function boundaries. This keeps function implementations intact
within individual chunks.

### Mermaid Diagrams
Mermaid diagram files are kept whole when possible (chunk size 1024) since
splitting a diagram mid-definition would lose structural meaning. When a
diagram exceeds the chunk size, it is split on double-newline boundaries.

## Embedding Model

The system uses nomic-embed-text via Ollama for local embeddings. This model
was chosen for several reasons:

- Small memory footprint: 137M parameters, only ~500MB VRAM
- Long context: 8,192 token context window handles large chunks
- 768-dimensional embeddings provide good accuracy-to-storage balance
- Requires task-specific prefixes for optimal performance:
  - `search_query: ` for queries
  - `search_document: ` for document chunks

## Vector Store

ChromaDB is used as the vector store with persistent on-disk storage.
It provides cosine similarity search, metadata filtering, and requires
zero infrastructure beyond the Python library.

## Configuration

All pipeline settings are configured via YAML files in the `configs/` directory.
The default configuration (`configs/default.yaml`) specifies the embedding model,
chunking strategy, vector store settings, and generation parameters.
