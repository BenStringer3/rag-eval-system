"""Retrieval-focused evaluation tests.

Tests whether the retriever surfaces relevant context for
queries in the eval dataset. Uses DeepEval's contextual
precision and recall metrics.
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase

from src.eval.datasets import load_eval_dataset
from src.rag.pipeline import RAGPipeline


@pytest.mark.eval
class TestRetrieval:
    """Retrieval quality tests."""

    def test_contextual_precision(
        self, pipeline: RAGPipeline, eval_dataset_path: str, judge_model: GPTModel
    ):
        """Retrieved chunks should be relevant to the query."""
        dataset = load_eval_dataset(eval_dataset_path)
        metric = ContextualPrecisionMetric(threshold=0.6, model=judge_model)

        for sample in dataset.samples:
            answer, context = pipeline.query_for_eval(sample.query)
            test_case = LLMTestCase(
                input=sample.query,
                actual_output=answer,
                expected_output=sample.expected_answer,
                retrieval_context=context,
            )
            assert_test(test_case, [metric])

    def test_contextual_recall(
        self, pipeline: RAGPipeline, eval_dataset_path: str, judge_model: GPTModel
    ):
        """Retrieved context should cover the information needed to answer."""
        dataset = load_eval_dataset(eval_dataset_path)
        metric = ContextualRecallMetric(threshold=0.6, model=judge_model)

        for sample in dataset.samples:
            answer, context = pipeline.query_for_eval(sample.query)
            test_case = LLMTestCase(
                input=sample.query,
                actual_output=answer,
                expected_output=sample.expected_answer,
                retrieval_context=context,
            )
            assert_test(test_case, [metric])
