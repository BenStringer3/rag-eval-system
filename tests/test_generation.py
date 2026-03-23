"""Generation-focused evaluation tests.

Tests whether the generator produces faithful, relevant answers
grounded in the retrieved context. Uses DeepEval's faithfulness
and answer relevancy metrics.
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase

from src.eval.datasets import load_eval_dataset
from src.rag.pipeline import RAGPipeline


@pytest.mark.eval
class TestGeneration:
    """Generation quality tests."""

    def test_faithfulness(
        self, pipeline: RAGPipeline, eval_dataset_path: str, judge_model: GPTModel
    ):
        """Generated answers should be grounded in retrieved context."""
        dataset = load_eval_dataset(eval_dataset_path)
        metric = FaithfulnessMetric(threshold=0.7, model=judge_model)

        for sample in dataset.samples:
            answer, context = pipeline.query_for_eval(sample.query)
            test_case = LLMTestCase(
                input=sample.query,
                actual_output=answer,
                retrieval_context=context,
            )
            assert_test(test_case, [metric])

    def test_answer_relevancy(
        self, pipeline: RAGPipeline, eval_dataset_path: str, judge_model: GPTModel
    ):
        """Generated answers should address the question asked."""
        dataset = load_eval_dataset(eval_dataset_path)
        metric = AnswerRelevancyMetric(threshold=0.7, model=judge_model)

        for sample in dataset.samples:
            answer, context = pipeline.query_for_eval(sample.query)
            test_case = LLMTestCase(
                input=sample.query,
                actual_output=answer,
                retrieval_context=context,
            )
            assert_test(test_case, [metric])
