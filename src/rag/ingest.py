"""Corpus ingestion CLI.

Usage:
    python -m src.rag.ingest --corpus-dir data/corpus
    python -m src.rag.ingest --corpus-dir data/corpus --reset
"""

from __future__ import annotations

import argparse
import sys

from src.rag.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store")
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="data/corpus",
        help="Path to the corpus directory (default: data/corpus)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to pipeline config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing index before ingesting",
    )
    args = parser.parse_args()

    pipeline = RAGPipeline.from_config(args.config)

    if args.reset:
        print("Resetting vector store and BM25 index...")
        pipeline.reset_storage()

    n_chunks = pipeline.ingest(args.corpus_dir)
    print(f"\nIngestion complete: {n_chunks} chunks indexed.")


if __name__ == "__main__":
    main()
