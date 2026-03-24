"""Evaluation runner.

Orchestrates running DeepEval metrics across an evaluation dataset
using the RAG pipeline, collecting results into an EvalReport.
"""

from __future__ import annotations

from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig

from src.data.schemas import EvalReport, EvalResult, MetricScore
from src.eval.datasets import load_eval_dataset, to_deepeval_test_cases
from src.eval.judge import judge_model_from_config_files
from src.eval.metrics import get_all_metrics
from src.rag.pipeline import RAGPipeline


def run_evaluation(
    pipeline: RAGPipeline,
    dataset_path: str | Path,
    include_custom_metrics: bool = False,
    core_threshold: float = 0.7,
    retrieval_threshold: float = 0.6,
    verbose: bool = True,
    default_config_path: str | Path = "configs/default.yaml",
    eval_config_path: str | Path = "configs/eval.yaml",
    max_concurrent: int = 5,
    judge_throttle_seconds: int = 1,
) -> EvalReport:
    """Run a full evaluation of the RAG pipeline on a dataset.

    Args:
        pipeline: Configured RAG pipeline.
        dataset_path: Path to the eval dataset JSON.
        include_custom_metrics: Whether to include Phase 2 custom metrics.
        core_threshold: Threshold for core RAG metrics.
        retrieval_threshold: Threshold for retrieval metrics.
        verbose: Print progress and results.
        default_config_path: Pipeline YAML (provides lm_studio base_url / api_key).
        eval_config_path: Eval YAML (provides judge model name / temperature).
        judge_throttle_seconds: Delay between async judge tasks to reduce rate limits.

    Returns:
        EvalReport with all results.
    """
    dataset = load_eval_dataset(dataset_path)
    judge_model = judge_model_from_config_files(default_config_path, eval_config_path)
    metrics = get_all_metrics(
        judge_model,
        core_threshold=core_threshold,
        retrieval_threshold=retrieval_threshold,
        include_custom=include_custom_metrics,
    )

    if verbose:
        print(f"Running evaluation on '{dataset.name}' ({dataset.size} samples)")
        print(f"  Metrics: {[m.__class__.__name__ for m in metrics]}")

    # Generate pipeline outputs for each sample
    test_cases = to_deepeval_test_cases(dataset, pipeline.query_for_eval)

    evaluation_result = evaluate(
        test_cases,
        metrics,
        display_config=DisplayConfig(print_results=verbose),
        async_config=AsyncConfig(
            max_concurrent=max_concurrent,
            throttle_value=judge_throttle_seconds,
        ),
        error_config=ErrorConfig(ignore_errors=False),
    )

    if len(evaluation_result.test_results) != len(dataset.samples):
        raise RuntimeError(
            "DeepEval result count mismatch: "
            f"{len(evaluation_result.test_results)} vs {len(dataset.samples)} samples"
        )

    results = []
    for sample, test_case, test_result in zip(
        dataset.samples, test_cases, evaluation_result.test_results
    ):
        scores = []
        for md in test_result.metrics_data or []:
            mscore = md.score if md.score is not None else 0.0
            reason = md.reason or md.error
            scores.append(
                MetricScore(
                    metric_name=md.name,
                    score=mscore,
                    threshold=md.threshold,
                    passed=md.success,
                    reason=reason,
                )
            )

        results.append(
            EvalResult(
                sample_id=sample.id,
                query=sample.query,
                generated_answer=test_case.actual_output or "",
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
