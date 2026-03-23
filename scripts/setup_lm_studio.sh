#!/usr/bin/env bash
# Bootstrap LM Studio for this RAG project.
# Prereq: install LM Studio and open the app at least once (required for `lms` to work).
# Model ids must match configs/default.yaml (embedding + generation) and configs/eval.yaml (judge).

set -euo pipefail

echo "=== RAG Eval System — LM Studio setup ==="

if ! command -v lms &>/dev/null; then
  echo "ERROR: lms not found on PATH."
  echo "  LM Studio installs the CLI at ~/.lmstudio/bin/lms — add that directory to PATH, or run from a shell where it is already configured."
  exit 1
fi

echo "[1/3] lms version / help"
lms --help >/dev/null || true

echo "[2/3] Starting local API server (default port 1234)"
lms server start

echo "[3/3] Load models (run these if not already loaded; ids must match your install and YAML)"
echo "  Embedding (example — verify with: lms ls):"
echo "    lms load text-embedding-nomic-embed-text-v1.5 -y"
echo "  Chat / judge (example):"
echo "    lms load qwen/qwen3-14b -y"
echo "  Check loaded: lms ps"
echo ""
echo "DeepEval judge is configured in Python from configs (GPTModel → LM Studio OpenAI-compatible API)."
echo "No deepeval set-ollama step is required."
echo ""
echo "Next:"
echo "  pip install -e '.[dev]'"
echo "  python -m src.rag.ingest --corpus-dir data/corpus"
echo "  pytest -m eval   # requires both models loaded + indexed corpus"
