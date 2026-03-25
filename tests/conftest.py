"""Shared fixtures for RAG pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.rag.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def pipeline() -> RAGPipeline:
    """Session-scoped RAG pipeline fixture.

    Loads config and connects to the persisted vector store.
    Assumes `python -m src.rag.ingest` has already been run.

    Hybrid retrieval is turned off so threshold-locked eval tests stay stable;
    production uses ``configs/default.yaml`` (hybrid on by default).
    """
    p = RAGPipeline.from_config(str(ROOT / "configs/default.yaml"))
    p.retriever.hybrid_enabled = False
    return p
@pytest.fixture(scope="session")
def eval_dataset_path() -> str:
    """Path to the starter evaluation dataset."""
    return str(ROOT / "data/eval_datasets/starter.json")
