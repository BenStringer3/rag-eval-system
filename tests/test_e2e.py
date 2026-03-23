"""End-to-end RAG evaluation tests.

Runs all core metrics on each sample as a combined suite.
This is the primary test target for CI/CD: if these pass,
the pipeline meets minimum quality thresholds.
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from src.eval.datasets import load_eval_dataset
from src.eval.metrics import get_all_metrics
from src.rag.pipeline import RAGPipeline


@pytest.mark.eval
class TestEndToEnd:
    """Full pipeline evaluation combining retrieval + generation metrics."""

    def test_all_metrics(self, pipeline: RAGPipeline, eval_dataset_path: str):
        """Each sample must pass all core RAG metrics."""
        dataset = load_eval_dataset(eval_dataset_path)
        metrics = get_all_metrics(
            core_threshold=0.7,
            retrieval_threshold=0.6,
            include_custom=False,
        )

        for sample in dataset.samples:
            answer, context = pipeline.query_for_eval(sample.query)
            test_case = LLMTestCase(
                input=sample.query,
                actual_output=answer,
                expected_output=sample.expected_answer,
                retrieval_context=context,
            )
            assert_test(test_case, metrics)


@pytest.mark.eval
@pytest.mark.slow
class TestEndToEndWithCustom:
    """Full pipeline evaluation including Phase 2 custom metrics.

    Marked as slow because custom G-Eval metrics require additional
    LLM judge calls. Run with: pytest -m "eval and slow"
    """

    def test_all_metrics_including_custom(
        self, pipeline: RAGPipeline, eval_dataset_path: str
    ):
        """Each sample must pass all metrics including custom ones."""
        dataset = load_eval_dataset(eval_dataset_path)
        metrics = get_all_metrics(
            core_threshold=0.7,
            retrieval_threshold=0.6,
            include_custom=True,
        )

        for sample in dataset.samples:
            answer, context = pipeline.query_for_eval(sample.query)
            test_case = LLMTestCase(
                input=sample.query,
                actual_output=answer,
                expected_output=sample.expected_answer,
                retrieval_context=context,
            )
            assert_test(test_case, metrics)
