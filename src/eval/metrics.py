"""Custom evaluation metrics built on DeepEval.

Phase 1: Standard RAG metrics (faithfulness, relevancy, context).
Phase 2+: Custom G-Eval metrics for code accuracy and diagram fidelity.

Metrics use a DeepEval GPTModel pointed at LM Studio's OpenAI-compatible API.
"""

from __future__ import annotations

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCaseParams


def get_core_metrics(judge_model: GPTModel, threshold: float = 0.7) -> list:
    """Return the standard RAG evaluation metrics."""
    return [
        FaithfulnessMetric(threshold=threshold, model=judge_model),
        AnswerRelevancyMetric(threshold=threshold, model=judge_model),
    ]


def get_retrieval_metrics(judge_model: GPTModel, threshold: float = 0.6) -> list:
    """Return retrieval-focused metrics.

    These require `expected_output` in the test case to compute
    contextual recall.
    """
    return [
        ContextualPrecisionMetric(threshold=threshold, model=judge_model),
        ContextualRecallMetric(threshold=threshold, model=judge_model),
    ]


# ---------------------------------------------------------------------------
# Phase 2: Custom G-Eval metrics (disabled by default)
# ---------------------------------------------------------------------------


def get_code_accuracy_metric(judge_model: GPTModel, threshold: float = 0.8) -> GEval:
    """Custom metric: does the response contain correct code?"""
    return GEval(
        name="Code Accuracy",
        criteria=(
            "Evaluate whether the code in the 'actual output' is syntactically correct "
            "and accurately addresses the coding question in the 'input'. "
            "Consider: correct syntax, appropriate use of APIs/libraries mentioned "
            "in the context, and whether the code would produce the expected behavior."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=judge_model,
        threshold=threshold,
    )


def get_diagram_fidelity_metric(judge_model: GPTModel, threshold: float = 0.7) -> GEval:
    """Custom metric: does the response accurately describe a diagram?"""
    return GEval(
        name="Diagram Fidelity",
        criteria=(
            "Evaluate whether the 'actual output' accurately describes the structure "
            "and relationships shown in the diagram content within the 'retrieval context'. "
            "Consider: correct identification of nodes/entities, accurate description "
            "of connections/flows, and proper understanding of the diagram's semantics "
            "(e.g., sequence diagrams show temporal order, flowcharts show decision logic)."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=judge_model,
        threshold=threshold,
    )


def get_all_metrics(
    judge_model: GPTModel,
    core_threshold: float = 0.7,
    retrieval_threshold: float = 0.6,
    include_custom: bool = False,
) -> list:
    """Return all configured metrics.

    Args:
        judge_model: DeepEval model (LM Studio via OpenAI-compatible API).
        core_threshold: Threshold for faithfulness + answer relevancy.
        retrieval_threshold: Threshold for contextual precision + recall.
        include_custom: Whether to include Phase 2 custom metrics.
    """
    metrics = get_core_metrics(judge_model, core_threshold) + get_retrieval_metrics(
        judge_model, retrieval_threshold
    )

    if include_custom:
        metrics.append(get_code_accuracy_metric(judge_model))
        metrics.append(get_diagram_fidelity_metric(judge_model))

    return metrics
