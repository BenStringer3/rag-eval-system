"""Custom evaluation metrics built on DeepEval.

Phase 1: Standard RAG metrics (faithfulness, relevancy, context).
Phase 2+: Custom G-Eval metrics for code accuracy and diagram fidelity.

All metrics work with local Ollama models as the LLM judge.
"""

from __future__ import annotations

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCaseParams


def get_core_metrics(threshold: float = 0.7) -> list:
    """Return the standard RAG evaluation metrics.

    These are the Phase 1 metrics that work out of the box with
    DeepEval + Ollama. No custom prompts needed.
    """
    return [
        FaithfulnessMetric(threshold=threshold),
        AnswerRelevancyMetric(threshold=threshold),
    ]


def get_retrieval_metrics(threshold: float = 0.6) -> list:
    """Return retrieval-focused metrics.

    These require `expected_output` in the test case to compute
    contextual recall.
    """
    return [
        ContextualPrecisionMetric(threshold=threshold),
        ContextualRecallMetric(threshold=threshold),
    ]


# ---------------------------------------------------------------------------
# Phase 2: Custom G-Eval metrics (disabled by default)
# ---------------------------------------------------------------------------

def get_code_accuracy_metric(threshold: float = 0.8) -> GEval:
    """Custom metric: does the response contain correct code?

    Evaluates whether code snippets in the response are syntactically
    correct and semantically aligned with the query's intent.
    """
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
        threshold=threshold,
    )


def get_diagram_fidelity_metric(threshold: float = 0.7) -> GEval:
    """Custom metric: does the response accurately describe a diagram?

    Evaluates whether the response correctly captures the structure,
    relationships, and flow depicted in a mermaid diagram.
    """
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
        threshold=threshold,
    )


def get_all_metrics(
    core_threshold: float = 0.7,
    retrieval_threshold: float = 0.6,
    include_custom: bool = False,
) -> list:
    """Return all configured metrics.

    Args:
        core_threshold: Threshold for faithfulness + answer relevancy.
        retrieval_threshold: Threshold for contextual precision + recall.
        include_custom: Whether to include Phase 2 custom metrics.
    """
    metrics = get_core_metrics(core_threshold) + get_retrieval_metrics(retrieval_threshold)

    if include_custom:
        metrics.append(get_code_accuracy_metric())
        metrics.append(get_diagram_fidelity_metric())

    return metrics
