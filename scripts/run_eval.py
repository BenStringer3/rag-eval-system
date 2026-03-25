#!/usr/bin/env python3
"""Run MLflow-native GenAI evaluation over one or more datasets."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.scorers import metric_pass_mask

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run mlflow.genai.evaluate() for the configured RAG app.")
    p.add_argument("--dataset", type=Path, default=None, help="Single dataset JSON path.")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    p.add_argument("--eval-config", type=Path, default=ROOT / "configs" / "eval.yaml")
    p.add_argument("--ingest", action="store_true")
    p.add_argument("--corpus-dir", type=Path, default=ROOT / "data" / "corpus")
    return p.parse_args()


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _predict_fn(pipeline):
    def predict(question: str) -> str:
        return pipeline.query(question).answer

    return predict


def _summarize_results(
    result_df: pd.DataFrame | None,
    thresholds: dict[str, float],
) -> tuple[float, dict[str, float]]:
    if result_df is None or result_df.empty:
        return 0.0, {}

    per_metric: dict[str, float] = {}
    passed_columns: list[pd.Series] = []
    for metric_name, threshold in thresholds.items():
        column = f"{metric_name}/value"
        if column not in result_df.columns:
            continue
        passed = metric_pass_mask(result_df[column], threshold)
        per_metric[f"{metric_name}_pass_rate"] = float(passed.mean())
        passed_columns.append(passed)

    if not passed_columns:
        return 0.0, per_metric

    overall = passed_columns[0]
    for passed in passed_columns[1:]:
        overall = overall & passed
    return float(overall.mean()), per_metric


def main() -> int:
    args = _parse_args()

    from src.eval.datasets import load_eval_dataset, load_mlflow_eval_data
    from src.eval.eval_config import enabled_dataset_paths
    from src.eval.scorers import (
        build_scorers,
        load_eval_config,
        metric_thresholds,
    )
    from src.rag.pipeline import RAGPipeline

    eval_cfg = load_eval_config(args.eval_config)
    tracking_cfg = eval_cfg.get("tracking") or {}
    tracking_uri = tracking_cfg.get("uri")
    experiment_name = tracking_cfg.get("experiment", "rag-eval-system")
    if not tracking_uri:
        raise ValueError(f"tracking.uri is required in {args.eval_config}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    scorers = build_scorers(args.eval_config)
    thresholds = metric_thresholds(eval_cfg)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    pipeline = RAGPipeline.from_config(str(args.config))
    if args.ingest:
        pipeline.ingest(str(args.corpus_dir))

    if args.dataset is not None:
        datasets = [("single", args.dataset.resolve())]
    else:
        datasets = enabled_dataset_paths(args.eval_config, project_root=ROOT)

    rag_cfg = _load_yaml(args.config)
    gen_cfg = rag_cfg.get("generation") or {}
    retrieval_cfg = rag_cfg.get("retrieval") or {}
    hybrid_cfg = retrieval_cfg.get("hybrid") or {}

    for dataset_key, dataset_path in datasets:
        dataset = load_eval_dataset(dataset_path)
        eval_rows = load_mlflow_eval_data(dataset_path)
        run_name = f"eval-{dataset_key}-{timestamp}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(
                {
                    "eval.dataset_key": dataset_key,
                    "eval.dataset_name": dataset.name,
                    "eval.dataset_path": str(dataset_path),
                    "eval.kind": "standalone",
                    "rag.config_path": str(args.config.resolve()),
                    "rag.eval_config_path": str(args.eval_config.resolve()),
                    "rag.retrieval.hybrid_enabled": str(bool(hybrid_cfg.get("enabled", False))).lower(),
                }
            )
            mlflow.log_params(
                {
                    "generation_model": gen_cfg.get("model"),
                    "judge_model": (eval_cfg.get("judge") or {}).get("model"),
                    "retrieval_top_k": retrieval_cfg.get("top_k"),
                }
            )
            result = mlflow.genai.evaluate(
                data=eval_rows,
                predict_fn=_predict_fn(pipeline),
                scorers=scorers,
            )
            pass_rate, per_metric_pass_rates = _summarize_results(result.result_df, thresholds)
            mlflow.log_metric("pass_rate", pass_rate)
            for metric_name, value in per_metric_pass_rates.items():
                mlflow.log_metric(metric_name, value)

            print(
                f"{dataset_key}: run_id={run.info.run_id} pass_rate={pass_rate:.3f} "
                f"rows={len(result.result_df) if result.result_df is not None else 0}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
