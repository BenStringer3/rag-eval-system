"""Shared fixtures for DeepEval test suites.

Provides a configured RAG pipeline and evaluation dataset
that all test files can use.
"""

from __future__ import annotations

import pytest

from src.rag.pipeline import RAGPipeline


@pytest.fixture(scope="session")
def pipeline() -> RAGPipeline:
    """Session-scoped RAG pipeline fixture.

    Loads config and connects to the persisted vector store.
    Assumes `python -m src.rag.ingest` has already been run.
    """
    return RAGPipeline.from_config("configs/default.yaml")


@pytest.fixture(scope="session")
def eval_dataset_path() -> str:
    """Path to the starter evaluation dataset."""
    return "data/eval_datasets/starter.json"
