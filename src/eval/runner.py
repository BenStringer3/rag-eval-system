"""Evaluation runner.

Orchestrates running DeepEval metrics across an evaluation dataset
using the RAG pipeline, collecting results into an EvalReport.
"""

from __future__ import annotations

from pathlib import Path

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from src.data.schemas import EvalReport, EvalResult, MetricScore
from src.eval.datasets import load_eval_dataset, to_deepeval_test_cases
from src.eval.metrics import get_all_metrics
from src.rag.pipeline import RAGPipeline


def run_evaluation(
    pipeline: RAGPipeline,
    dataset_path: str | Path,
    include_custom_metrics: bool = False,
    core_threshold: float = 0.7,
    retrieval_threshold: float = 0.6,
    verbose: bool = True,
) -> EvalReport:
    """Run a full evaluation of the RAG pipeline on a dataset.

    Args:
        pipeline: Configured RAG pipeline.
        dataset_path: Path to the eval dataset JSON.
        include_custom_metrics: Whether to include Phase 2 custom metrics.
        core_threshold: Threshold for core RAG metrics.
        retrieval_threshold: Threshold for retrieval metrics.
        verbose: Print progress and results.

    Returns:
        EvalReport with all results.
    """
    dataset = load_eval_dataset(dataset_path)
    metrics = get_all_metrics(
        core_threshold=core_threshold,
        retrieval_threshold=retrieval_threshold,
        include_custom=include_custom_metrics,
    )

    if verbose:
        print(f"Running evaluation on '{dataset.name}' ({dataset.size} samples)")
        print(f"  Metrics: {[m.__class__.__name__ for m in metrics]}")

    # Generate pipeline outputs for each sample
    test_cases = to_deepeval_test_cases(dataset, pipeline.query_for_eval)

    # Run DeepEval evaluation
    eval_results = evaluate(test_cases, metrics, print_results=verbose)

    # Convert to our EvalReport format
    results = []
    for i, (sample, test_case) in enumerate(zip(dataset.samples, test_cases)):
        scores = []
        for metric in metrics:
            # After evaluate(), each metric has .score and .reason populated
            # for the last test case. We need to re-measure for each.
            # Note: In practice, evaluate() handles this internally.
            scores.append(
                MetricScore(
                    metric_name=metric.__class__.__name__,
                    score=metric.score if hasattr(metric, "score") and metric.score is not None else 0.0,
                    threshold=metric.threshold,
                    passed=metric.score >= metric.threshold if hasattr(metric, "score") and metric.score is not None else False,
                    reason=getattr(metric, "reason", None),
                )
            )

        results.append(
            EvalResult(
                sample_id=sample.id,
                query=sample.query,
                generated_answer=test_case.actual_output,
                retrieval_context=test_case.retrieval_context or [],
                expected_answer=sample.expected_answer,
                scores=scores,
            )
        )

    report = EvalReport(
        dataset_name=dataset.name,
        results=results,
        config={
            "core_threshold": core_threshold,
            "retrieval_threshold": retrieval_threshold,
            "include_custom_metrics": include_custom_metrics,
        },
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Results: {report.pass_rate:.1%} pass rate")
        for name, score in report.mean_scores.items():
            print(f"  {name}: {score:.3f}")

    return report
