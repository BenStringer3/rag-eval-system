"""Evaluation dataset management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.schemas import EvalDataset, EvalSample


def load_eval_dataset(path: str | Path) -> EvalDataset:
    """Load an evaluation dataset from a JSON file.

    Expected format:
    {
        "name": "starter",
        "description": "Hand-curated Q&A pairs",
        "version": "1.0.0",
        "samples": [
            {
                "id": "q001",
                "query": "What is ...",
                "expected_answer": "...",
                "expected_context": ["..."],   // optional
                "metadata": {"category": "code"}  // optional
            }
        ]
    }
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return EvalDataset(**data)


def _sample_to_mlflow_row(dataset: EvalDataset, sample: EvalSample) -> dict[str, Any]:
    expectations: dict[str, Any] = {
        "expected_output": sample.expected_answer,
        "sample_id": sample.id,
    }
    if sample.expected_context:
        expectations["expected_context"] = sample.expected_context
    if sample.metadata:
        expectations["metadata"] = sample.metadata

    return {
        "request_id": sample.id,
        "inputs": {"question": sample.query},
        "expectations": expectations,
        "tags": {
            "dataset_name": dataset.name,
            "sample_id": sample.id,
        },
    }


def load_mlflow_eval_data(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON eval dataset and convert it into MLflow GenAI rows."""
    dataset = load_eval_dataset(path)
    return [_sample_to_mlflow_row(dataset, sample) for sample in dataset.samples]


def create_starter_dataset() -> EvalDataset:
    """Generate a minimal starter dataset for initial testing.

    These are placeholder questions that work with any corpus.
    Replace with corpus-specific Q&A pairs once you have documents.
    """
    samples = [
        EvalSample(
            id="q001",
            query="What is this project about?",
            expected_answer="This project is a RAG system with evaluation infrastructure.",
            metadata={"category": "general", "difficulty": "easy"},
        ),
        EvalSample(
            id="q002",
            query="How are documents chunked?",
            expected_answer="Documents are split using recursive character splitting with document-type-aware separators.",
            metadata={"category": "architecture", "difficulty": "medium"},
        ),
        EvalSample(
            id="q003",
            query="What embedding model is used?",
            expected_answer=(
                "The system uses a Nomic embedding model via LM Studio for local embeddings."
            ),
            metadata={"category": "architecture", "difficulty": "easy"},
        ),
    ]

    return EvalDataset(
        name="starter",
        description="Minimal starter dataset for smoke testing the eval pipeline",
        samples=samples,
        version="0.1.0",
    )


def save_dataset(dataset: EvalDataset, path: str | Path) -> None:
    """Save an EvalDataset to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dataset.model_dump(), f, indent=2)
    print(f"Saved {dataset.size} samples to {path}")
