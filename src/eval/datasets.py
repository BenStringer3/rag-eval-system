"""Evaluation dataset management.

Loads eval datasets from JSON, converts to DeepEval test cases,
and provides filtering/sampling utilities.
"""

from __future__ import annotations

import json
from pathlib import Path

from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase

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


def to_deepeval_test_cases(
    dataset: EvalDataset,
    pipeline_fn: callable,
) -> list[LLMTestCase]:
    """Convert an EvalDataset to DeepEval test cases by running the pipeline.

    Args:
        dataset: The evaluation dataset.
        pipeline_fn: A callable that takes a query string and returns
                      (answer: str, retrieval_context: list[str]).

    Returns:
        List of LLMTestCase objects ready for DeepEval evaluation.
    """
    test_cases = []

    for sample in dataset.samples:
        answer, context = pipeline_fn(sample.query)

        test_case = LLMTestCase(
            input=sample.query,
            actual_output=answer,
            expected_output=sample.expected_answer,
            retrieval_context=context,
        )
        test_cases.append(test_case)

    return test_cases


def to_deepeval_dataset(
    dataset: EvalDataset,
    pipeline_fn: callable,
) -> EvaluationDataset:
    """Convert to a full DeepEval EvaluationDataset.

    This wraps the test cases in DeepEval's dataset container,
    which provides additional utilities like push/pull from
    Confident AI (if configured).
    """
    test_cases = to_deepeval_test_cases(dataset, pipeline_fn)
    return EvaluationDataset(test_cases=test_cases)


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
