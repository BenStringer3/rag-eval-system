#!/usr/bin/env bash
# Setup script for Ollama models used by the RAG system.
# Run: bash scripts/setup_ollama.sh

set -euo pipefail

echo "=== RAG Eval System — Ollama Setup ==="
echo ""

# Check Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "ERROR: Ollama is not installed."
    echo "Install from: https://ollama.com/download"
    exit 1
fi

echo "Ollama version: $(ollama --version)"
echo ""

# Pull embedding model
echo "[1/3] Pulling embedding model: nomic-embed-text"
ollama pull nomic-embed-text

# Pull generation model
echo ""
echo "[2/3] Pulling generation model: qwen3:14b"
echo "  (This is ~8GB, may take a few minutes)"
ollama pull qwen3:14b

# Configure DeepEval to use Ollama
echo ""
echo "[3/3] Configuring DeepEval to use Ollama for evaluation"
if command -v deepeval &> /dev/null; then
    deepeval set-ollama --model=qwen3:14b
    echo "  DeepEval configured to use qwen3:14b as LLM judge"
else
    echo "  WARNING: deepeval CLI not found. Install with: pip install deepeval"
    echo "  Then run: deepeval set-ollama --model=qwen3:14b"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Verify models are available:"
echo "  ollama list"
echo ""
echo "Next steps:"
echo "  1. pip install -e '.[dev]'"
echo "  2. python -m src.rag.ingest --corpus-dir data/corpus"
echo "  3. deepeval test run tests/"
