#!/usr/bin/env python3
"""Run batch RAG evaluation and write markdown, JSON, and CSV reports.

Run from the repository root with the package installed (``pip install -e .``).
Generation models come from ``configs/default.yaml``. Judge provider/model come from
``configs/eval.yaml`` (local LM Studio or cloud OpenAI-compatible endpoint).

Example::

    python scripts/run_eval_report.py
    python scripts/run_eval_report.py --ingest --output-dir artifacts/eval_runs
    python scripts/run_eval_report.py --include-custom
    python scripts/run_eval_report.py --dataset data/eval_datasets/starter.json  # single dataset only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _git_short_sha(cwd: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return r.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run DeepEval-backed evaluation and export reports under a timestamped folder.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "eval_runs",
        help="Base directory; a UTC timestamp subfolder is created for each run (default: artifacts/eval_runs).",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Eval dataset JSON path. If omitted, runs all enabled datasets from --eval-config.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
        help="RAG pipeline YAML.",
    )
    p.add_argument(
        "--eval-config",
        type=Path,
        default=ROOT / "configs" / "eval.yaml",
        help="Eval / judge YAML.",
    )
    p.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest corpus into the vector store before evaluating.",
    )
    p.add_argument(
        "--corpus-dir",
        type=Path,
        default=ROOT / "data" / "corpus",
        help="Corpus directory when --ingest is set (default: data/corpus).",
    )
    p.add_argument(
        "--include-custom",
        action="store_true",
        help="Include Phase-2 custom G-Eval metrics (extra judge calls).",
    )
    p.add_argument(
        "--core-threshold",
        type=float,
        default=0.7,
        help="Pass threshold for core generation metrics.",
    )
    p.add_argument(
        "--retrieval-threshold",
        type=float,
        default=0.6,
        help="Pass threshold for retrieval metrics.",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Max concurrent judge requests sent by DeepEval (default: 3). Lower if you see timeouts/rate limits.",
    )
    p.add_argument(
        "--judge-throttle-seconds",
        type=int,
        default=3,
        help="Seconds DeepEval waits between judge tasks (default: 3). Increase to reduce provider rate limits.",
    )
    p.add_argument(
        "--per-task-timeout",
        type=int,
        default=500,
        help="Per-task timeout in seconds for DeepEval (default: 500). Raise if your judge is consistently slower.",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Less console output from DeepEval.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    sys.path.insert(0, str(ROOT))

    # Set DeepEval timeout env vars before any deepeval import (settings read at import time).
    import os
    os.environ.setdefault(
        "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", str(args.per_task_timeout)
    )

    from src.eval.eval_config import enabled_dataset_paths
    from src.eval.report import to_json, to_markdown, to_csv
    from src.eval.runner import run_evaluation
    from src.rag.pipeline import RAGPipeline

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir: Path = args.output_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    # Judge wrapper reads these env vars to write run-scoped JSONL traces.
    os.environ["RAG_EVAL_RUN_ID"] = stamp
    os.environ["RAG_EVAL_JUDGE_TRACE_PATH"] = str(
        (run_dir / "judge_length_errors.jsonl").resolve()
    )

    if args.dataset is not None:
        to_run = [("single", args.dataset.resolve())]
    else:
        to_run = enabled_dataset_paths(args.eval_config, project_root=ROOT)

    pipeline = RAGPipeline.from_config(str(args.config))
    if args.ingest:
        pipeline.ingest(str(args.corpus_dir))

    per_dataset_meta: list[dict] = []
    for ds_key, ds_path in to_run:
        if not ds_path.is_file():
            raise FileNotFoundError(f"Eval dataset not found: {ds_path}")
        os.environ["RAG_EVAL_DATASET_KEY"] = ds_key
        os.environ["RAG_EVAL_DATASET_PATH"] = str(ds_path)

        report = run_evaluation(
            pipeline,
            ds_path,
            include_custom_metrics=args.include_custom,
            core_threshold=args.core_threshold,
            retrieval_threshold=args.retrieval_threshold,
            verbose=not args.quiet,
            default_config_path=args.config,
            eval_config_path=args.eval_config,
            max_concurrent=args.max_concurrent,
            judge_throttle_seconds=args.judge_throttle_seconds,
        )

        sub = run_dir / ds_key
        sub.mkdir(parents=True, exist_ok=True)
        md_path = sub / "report.md"
        md_path.write_text(to_markdown(report), encoding="utf-8")
        print(f"Wrote {md_path}")
        to_json(report, sub / "report.json")
        to_csv(report, sub / "report.csv")

        per_dataset_meta.append(
            {
                "key": ds_key,
                "path": str(ds_path),
                "pass_rate": report.pass_rate,
                "mean_scores": report.mean_scores,
            }
        )

    with open(args.config, encoding="utf-8") as f:
        default_cfg = yaml.safe_load(f)
    with open(args.eval_config, encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)
    judge_cfg = eval_cfg["judge"]
    gen_cfg = default_cfg["generation"]

    retrieval = default_cfg.get("retrieval") or {}
    hybrid = retrieval.get("hybrid") or {}
    meta = {
        "run_id": stamp,
        "datasets": per_dataset_meta,
        "config": str(args.config.resolve()),
        "eval_config": str(args.eval_config.resolve()),
        "include_custom_metrics": args.include_custom,
        "core_threshold": args.core_threshold,
        "retrieval_threshold": args.retrieval_threshold,
        "eval_context": {
            "generation_model": gen_cfg["model"],
            "generation_temperature": gen_cfg["temperature"],
            "judge_model": judge_cfg["model"],
            "judge_temperature": judge_cfg.get("temperature"),
        },
        "git_commit": _git_short_sha(ROOT),
        "retrieval_features": {
            "hybrid_enabled": bool(hybrid.get("enabled", False)),
        },
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nRun directory: {run_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
