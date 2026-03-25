#!/usr/bin/env python3
"""Run a fixed-N MLflow-backed A/B study for two RAG config arms."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.ab_study_config import write_yaml_with_overrides  # noqa: E402
from src.eval.datasets import load_eval_dataset, load_mlflow_eval_data  # noqa: E402
from src.eval.scorers import (  # noqa: E402
    build_scorers,
    load_eval_config,
    metric_pass_mask,
    metric_thresholds,
)
from src.rag.pipeline import RAGPipeline  # noqa: E402


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed-N in-process MLflow A/B evaluation.")
    p.add_argument("--study-id", type=str, default=None)
    p.add_argument("--base-config", type=Path, default=ROOT / "configs" / "default.yaml")
    p.add_argument("--eval-config", type=Path, default=ROOT / "configs" / "eval.yaml")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--arm-a-label", type=str, default="arm_a")
    p.add_argument("--arm-b-label", type=str, default="arm_b")
    p.add_argument("--arm-a-overrides-json", type=str, required=True)
    p.add_argument("--arm-b-overrides-json", type=str, required=True)
    p.add_argument("--pairing", choices=("none", "paired"), default="paired")
    p.add_argument("--n-per-arm", type=int, required=True)
    p.add_argument("--ingest", action="store_true")
    p.add_argument("--corpus-dir", type=Path, default=ROOT / "data" / "corpus")
    return p.parse_args()


def _parse_overrides_json(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise SystemExit("Overrides JSON must decode to an object.")
    return value


def _predict_fn(pipeline: RAGPipeline):
    def predict(question: str) -> str:
        return pipeline.query(question).answer

    return predict


def _summarize_results(result_df: pd.DataFrame | None, thresholds: dict[str, float]) -> float:
    if result_df is None or result_df.empty:
        return 0.0
    passed_columns: list[pd.Series] = []
    for metric_name, threshold in thresholds.items():
        column = f"{metric_name}/value"
        if column in result_df.columns:
            passed_columns.append(metric_pass_mask(result_df[column], threshold))
    if not passed_columns:
        return 0.0
    overall = passed_columns[0]
    for passed in passed_columns[1:]:
        overall = overall & passed
    return float(overall.mean())


def _run_arm(
    *,
    arm_key: str,
    arm_label: str,
    overrides: dict[str, Any],
    block: int | None,
    study_id: str,
    args: argparse.Namespace,
    eval_rows: list[dict[str, Any]],
    dataset_name: str,
    scorers,
    thresholds: dict[str, float],
) -> tuple[str, float]:
    cfg_path = ROOT / "artifacts" / "ab_study" / study_id / "configs" / f"{arm_key}.yaml"
    write_yaml_with_overrides(args.base_config, cfg_path, overrides=overrides)
    pipeline = RAGPipeline.from_config(str(cfg_path))
    if args.ingest:
        pipeline.ingest(str(args.corpus_dir))

    with mlflow.start_run(
        run_name=f"{study_id}-{arm_key}-{block if block is not None else 'na'}"
    ) as run:
        mlflow.set_tags(
            {
                "study_id": study_id,
                "arm": arm_key,
                "arm_label": arm_label,
                "block": "none" if block is None else str(block),
                "pairing": args.pairing,
                "eval.kind": "ab_study",
                "eval.dataset_name": dataset_name,
                "eval.dataset_path": str(args.dataset.resolve()),
                "rag.config_path": str(cfg_path.resolve()),
            }
        )
        result = mlflow.genai.evaluate(
            data=eval_rows,
            predict_fn=_predict_fn(pipeline),
            scorers=scorers,
        )
        pass_rate = _summarize_results(result.result_df, thresholds)
        mlflow.log_metric("pass_rate", pass_rate)
        return run.info.run_id, pass_rate


def main() -> int:
    args = _parse_args()
    if args.n_per_arm < 1:
        raise SystemExit("--n-per-arm must be >= 1")

    study_id = args.study_id or _utc_stamp()
    study_dir = ROOT / "artifacts" / "ab_study" / study_id
    study_dir.mkdir(parents=True, exist_ok=True)

    arm_a_overrides = _parse_overrides_json(args.arm_a_overrides_json)
    arm_b_overrides = _parse_overrides_json(args.arm_b_overrides_json)

    eval_cfg = load_eval_config(args.eval_config)
    tracking_cfg = eval_cfg.get("tracking") or {}
    tracking_uri = tracking_cfg.get("uri")
    experiment_name = tracking_cfg.get("experiment", "rag-eval-system")
    if not tracking_uri:
        raise ValueError(f"tracking.uri is required in {args.eval_config}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    dataset = load_eval_dataset(args.dataset)
    eval_rows = load_mlflow_eval_data(args.dataset)
    scorers = build_scorers(args.eval_config)
    thresholds = metric_thresholds(eval_cfg)

    manifest: list[dict[str, Any]] = []
    if args.pairing == "paired":
        for block in range(args.n_per_arm):
            for arm_key, arm_label, overrides in (
                ("a", args.arm_a_label, arm_a_overrides),
                ("b", args.arm_b_label, arm_b_overrides),
            ):
                run_id, pass_rate = _run_arm(
                    arm_key=arm_key,
                    arm_label=arm_label,
                    overrides=overrides,
                    block=block,
                    study_id=study_id,
                    args=args,
                    eval_rows=eval_rows,
                    dataset_name=dataset.name,
                    scorers=scorers,
                    thresholds=thresholds,
                )
                manifest.append(
                    {"block": block, "arm": arm_key, "run_id": run_id, "pass_rate": pass_rate}
                )
                print(f"block={block} arm={arm_key} run_id={run_id} pass_rate={pass_rate:.3f}")
    else:
        for arm_key, arm_label, overrides in (
            ("a", args.arm_a_label, arm_a_overrides),
            ("b", args.arm_b_label, arm_b_overrides),
        ):
            for _ in range(args.n_per_arm):
                run_id, pass_rate = _run_arm(
                    arm_key=arm_key,
                    arm_label=arm_label,
                    overrides=overrides,
                    block=None,
                    study_id=study_id,
                    args=args,
                    eval_rows=eval_rows,
                    dataset_name=dataset.name,
                    scorers=scorers,
                    thresholds=thresholds,
                )
                manifest.append({"arm": arm_key, "run_id": run_id, "pass_rate": pass_rate})
                print(f"arm={arm_key} run_id={run_id} pass_rate={pass_rate:.3f}")

    (study_dir / "study_meta.json").write_text(
        json.dumps(
            {
                "study_id": study_id,
                "pairing": args.pairing,
                "n_per_arm": args.n_per_arm,
                "dataset": str(args.dataset.resolve()),
                "arm_a_label": args.arm_a_label,
                "arm_b_label": args.arm_b_label,
                "arm_a_overrides": arm_a_overrides,
                "arm_b_overrides": arm_b_overrides,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (study_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"study_id={study_id}")
    print(f"manifest={study_dir / 'manifest.json'}")
    print(f"mlflow_filter=tags.study_id = '{study_id}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
