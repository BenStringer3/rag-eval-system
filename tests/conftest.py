"""Shared fixtures for DeepEval test suites.

Provides a configured RAG pipeline and evaluation dataset
that all test files can use.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from deepeval.models import GPTModel

from src.eval.judge import judge_model_from_config_files
from src.rag.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def pipeline() -> RAGPipeline:
    """Session-scoped RAG pipeline fixture.

    Loads config and connects to the persisted vector store.
    Assumes `python -m src.rag.ingest` has already been run.
    """
    return RAGPipeline.from_config(str(ROOT / "configs/default.yaml"))


@pytest.fixture(scope="session")
def judge_model() -> GPTModel:
    """DeepEval judge targeting LM Studio (from configs/default.yaml + configs/eval.yaml)."""
    return judge_model_from_config_files(
        ROOT / "configs/default.yaml",
        ROOT / "configs/eval.yaml",
    )


@pytest.fixture(scope="session")
def eval_dataset_path() -> str:
    """Path to the starter evaluation dataset."""
    return str(ROOT / "data/eval_datasets/starter.json")
