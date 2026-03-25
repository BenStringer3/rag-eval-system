#!/usr/bin/env python3
"""Run repeated evals for an A/B comparison (fixed-N or exploratory sequential).

Writes ``artifacts/ab_study/<study_id>/manifest.jsonl`` and optionally ingests
successful runs into ``data/eval_registry.db``.

Example (paired fixed-N, single dataset)::

    .venv/bin/python scripts/run_ab_study.py \\
      --dataset data/eval_datasets/synthetic.json \\
      --arm-a-label "control" --arm-b-label "treatment" \\
      --arm-a-overrides-json '{"retrieval.hybrid.enabled": false}' \\
      --arm-b-overrides-json '{"retrieval.hybrid.enabled": true}' \\
      --pairing paired --n-per-arm 3 \\
      --max-concurrent 1 --judge-throttle-seconds 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.ab_study_config import (  # noqa: E402
    eval_run_dir_complete,
    paired_run_ids_from_manifest_records,
    write_yaml_with_overrides,
)
from src.eval.registry_db import connect, init_schema  # noqa: E402
from src.eval.registry_ingest import ingest_run_dir  # noqa: E402
from src.eval.registry_stats import load_series, two_sample_test  # noqa: E402

ArmKey = Literal["a", "b"]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A/B eval study orchestrator.")
    p.add_argument(
        "--study-id",
        type=str,
        default=None,
        help="Folder name under artifacts/ab_study (default: UTC stamp).",
    )
    p.add_argument("--base-config", type=Path, default=ROOT / "configs" / "default.yaml")
    p.add_argument("--eval-config", type=Path, default=ROOT / "configs" / "eval.yaml")
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="If set, same as run_eval_report --dataset (dataset key 'single').",
    )
    p.add_argument("--registry-db", type=Path, default=ROOT / "data" / "eval_registry.db")
    p.add_argument("--no-ingest", action="store_true", help="Do not update SQLite registry after successful runs.")

    p.add_argument("--arm-a-label", type=str, default="arm_a")
    p.add_argument("--arm-b-label", type=str, default="arm_b")
    p.add_argument(
        "--arm-a-overrides-json",
        type=str,
        required=True,
        help="JSON object mapping dotted paths to values.",
    )
    p.add_argument(
        "--arm-b-overrides-json",
        type=str,
        required=True,
        help="JSON object mapping dotted paths to values.",
    )

    p.add_argument(
        "--pairing",
        choices=("none", "paired"),
        default="paired",
        help="paired: each block is A then B. none: run all A then all B (fixed-N only).",
    )
    p.add_argument("--n-per-arm", type=int, default=None, help="Fixed-N: runs per arm (paired: number of blocks).")
    p.add_argument("--exploratory", action="store_true", help="Sequential stopping (label results exploratory).")
    p.add_argument("--min-per-arm", type=int, default=2, help="Exploratory: min completed pairs or rounds per arm.")
    p.add_argument("--max-per-arm", type=int, default=12, help="Exploratory: max pairs (paired) or rounds (none).")
    p.add_argument("--alpha", type=float, default=0.05, help="Exploratory: stop if p-value < alpha after min reached.")
    p.add_argument(
        "--primary-metric",
        type=str,
        default="pass_rate",
        help="Exploratory stopping uses this metric (pass_rate or mean_scores key, e.g. Faithfulness).",
    )

    p.add_argument("--retries", type=int, default=0, help="Retry failed eval subprocess per attempt (same arm).")
    p.add_argument("--retry-backoff-seconds", type=float, default=30.0)
    p.add_argument("--max-concurrent", type=int, default=3)
    p.add_argument("--judge-throttle-seconds", type=int, default=3)
    p.add_argument("--per-task-timeout", type=int, default=500)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _append_manifest(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _dataset_keys_for_run(args: argparse.Namespace) -> list[str]:
    if args.dataset is not None:
        return ["single"]
    from src.eval.eval_config import enabled_dataset_paths

    pairs = enabled_dataset_paths(args.eval_config, project_root=ROOT)
    return [k for k, _ in pairs]


def _run_eval_subprocess(*, config_path: Path, eval_output_dir: Path, args: argparse.Namespace) -> tuple[int, str]:
    cmd: list[str] = [
        sys.executable,
        str(ROOT / "scripts" / "run_eval_report.py"),
        "--config",
        str(config_path),
        "--eval-config",
        str(args.eval_config),
        "--output-dir",
        str(eval_output_dir),
        "--max-concurrent",
        str(args.max_concurrent),
        "--judge-throttle-seconds",
        str(args.judge_throttle_seconds),
        "--per-task-timeout",
        str(args.per_task_timeout),
    ]
    if args.quiet:
        cmd.append("--quiet")
    if args.dataset is not None:
        cmd.extend(["--dataset", str(args.dataset.resolve())])
    r = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=None,
    )
    tail = (r.stderr or "")[-4000:] + (r.stdout or "")[-4000:]
    return r.returncode, tail


def _discover_new_run_dir(eval_output_dir: Path, before: set[str]) -> Path | None:
    if not eval_output_dir.is_dir():
        return None
    names = {p.name for p in eval_output_dir.iterdir() if p.is_dir()}
    new = names - before
    if len(new) == 1:
        return eval_output_dir / next(iter(new))
    return None


def _attempt_eval(
    *,
    arm: ArmKey,
    block: int | None,
    overrides: dict[str, Any],
    study_dir: Path,
    eval_output_dir: Path,
    base_config: Path,
    args: argparse.Namespace,
    manifest_path: Path,
    study_id: str,
) -> str | None:
    cfg_dir = study_dir / "configs"
    out_cfg = cfg_dir / f"default_arm_{arm}.yaml"
    write_yaml_with_overrides(base_config, out_cfg, overrides=overrides)

    retries = max(0, int(args.retries))
    last_rc = -1
    last_tail = ""
    for attempt in range(retries + 1):
        before = {p.name for p in eval_output_dir.iterdir() if p.is_dir()} if eval_output_dir.is_dir() else set()
        rc, tail = _run_eval_subprocess(config_path=out_cfg, eval_output_dir=eval_output_dir, args=args)
        last_rc, last_tail = rc, tail
        if rc == 0:
            run_dir = _discover_new_run_dir(eval_output_dir, before)
            if run_dir is None:
                _append_manifest(
                    manifest_path,
                    {
                        "ts": _utc_stamp(),
                        "study_id": study_id,
                        "status": "failed",
                        "arm": arm,
                        "block": block,
                        "error_hint": "no new run directory after exit 0",
                        "exit_code": 0,
                    },
                )
                return None
            if not eval_run_dir_complete(run_dir):
                _append_manifest(
                    manifest_path,
                    {
                        "ts": _utc_stamp(),
                        "study_id": study_id,
                        "status": "failed",
                        "arm": arm,
                        "block": block,
                        "run_id": run_dir.name,
                        "error_hint": "incomplete meta/report.json",
                        "exit_code": 0,
                    },
                )
                return None
            rid = run_dir.name
            _append_manifest(
                manifest_path,
                {
                    "ts": _utc_stamp(),
                    "study_id": study_id,
                    "status": "success",
                    "arm": arm,
                    "block": block,
                    "run_id": rid,
                    "config_path": str(out_cfg.resolve()),
                    "exit_code": 0,
                },
            )
            if not args.no_ingest:
                conn = connect(args.registry_db)
                init_schema(conn)
                ingest_run_dir(conn, run_dir)
                conn.close()
            return rid
        if attempt < retries:
            time.sleep(args.retry_backoff_seconds)

    _append_manifest(
        manifest_path,
        {
            "ts": _utc_stamp(),
            "study_id": study_id,
            "status": "failed",
            "arm": arm,
            "block": block,
            "error_hint": last_tail[:2000],
            "exit_code": last_rc,
        },
    )
    return None


def _load_manifest_runs(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _successful_ids_by_arm(records: list[dict[str, Any]], arm: ArmKey) -> list[str]:
    xs = [r for r in records if r.get("status") == "success" and r.get("arm") == arm]
    xs.sort(key=lambda r: (r.get("block") is None, r.get("block", 0), r.get("run_id", "")))
    return [r["run_id"] for r in xs if r.get("run_id")]


def _exploratory_stop(
    *,
    conn,
    dataset_key: str,
    primary: str,
    a_ids: list[str],
    b_ids: list[str],
    paired: bool,
    alpha: float,
) -> tuple[bool, float | None]:
    if len(a_ids) < 2 or len(b_ids) < 2:
        return False, None
    if paired and len(a_ids) != len(b_ids):
        return False, None
    try:
        a = load_series(conn, dataset_key, primary, a_ids)
        b = load_series(conn, dataset_key, primary, b_ids)
        res = two_sample_test(b, a, paired=paired)  # treatment=B, control=A by convention
        return res.pvalue < alpha, res.pvalue
    except (KeyError, ValueError):
        return False, None


def _parse_overrides_json(s: str) -> dict[str, Any]:
    v = json.loads(s)
    if not isinstance(v, dict):
        raise SystemExit("Overrides JSON must decode to an object/dict.")
    return v


def main() -> int:
    args = _parse_args()
    if args.exploratory:
        if args.n_per_arm is not None:
            print("Do not pass --n-per-arm with --exploratory.", file=sys.stderr)
            return 1
    else:
        if args.n_per_arm is None:
            print("Pass --n-per-arm (fixed-N) or use --exploratory.", file=sys.stderr)
            return 1
        if args.n_per_arm < 1:
            print("--n-per-arm must be >= 1.", file=sys.stderr)
            return 1

    a_overrides = _parse_overrides_json(args.arm_a_overrides_json)
    b_overrides = _parse_overrides_json(args.arm_b_overrides_json)

    study_id = args.study_id or _utc_stamp()
    study_dir = ROOT / "artifacts" / "ab_study" / study_id
    eval_output_dir = study_dir / "eval_runs"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = study_dir / "manifest.jsonl"

    dataset_keys = _dataset_keys_for_run(args)
    if len(dataset_keys) != 1:
        print(
            "A/B study currently supports exactly one dataset key per run "
            f"(use --dataset for synthetic/starter). Got: {dataset_keys}.",
            file=sys.stderr,
        )
        return 1
    dataset_key = dataset_keys[0]

    study_meta = {
        "study_id": study_id,
        "pairing": args.pairing,
        "exploratory": args.exploratory,
        "n_per_arm": args.n_per_arm,
        "min_per_arm": args.min_per_arm,
        "max_per_arm": args.max_per_arm,
        "alpha": args.alpha,
        "primary_metric": args.primary_metric,
        "dataset_key": dataset_key,
        "arm_a_label": args.arm_a_label,
        "arm_b_label": args.arm_b_label,
        "arm_a_overrides": a_overrides,
        "arm_b_overrides": b_overrides,
    }
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "study_meta.json").write_text(json.dumps(study_meta, indent=2), encoding="utf-8")

    print(f"Study directory: {study_dir.resolve()}")
    print(f"Manifest: {manifest_path}")
    print(f"Eval output: {eval_output_dir}")

    if not args.exploratory:
        n = args.n_per_arm
        assert n is not None
        if args.pairing == "paired":
            for b in range(n):
                _attempt_eval(
                    arm="a",
                    block=b,
                    overrides=a_overrides,
                    study_dir=study_dir,
                    eval_output_dir=eval_output_dir,
                    base_config=args.base_config,
                    args=args,
                    manifest_path=manifest_path,
                    study_id=study_id,
                )
                _attempt_eval(
                    arm="b",
                    block=b,
                    overrides=b_overrides,
                    study_dir=study_dir,
                    eval_output_dir=eval_output_dir,
                    base_config=args.base_config,
                    args=args,
                    manifest_path=manifest_path,
                    study_id=study_id,
                )
        else:
            for _ in range(n):
                _attempt_eval(
                    arm="a",
                    block=None,
                    overrides=a_overrides,
                    study_dir=study_dir,
                    eval_output_dir=eval_output_dir,
                    base_config=args.base_config,
                    args=args,
                    manifest_path=manifest_path,
                    study_id=study_id,
                )
            for _ in range(n):
                _attempt_eval(
                    arm="b",
                    block=None,
                    overrides=b_overrides,
                    study_dir=study_dir,
                    eval_output_dir=eval_output_dir,
                    base_config=args.base_config,
                    args=args,
                    manifest_path=manifest_path,
                    study_id=study_id,
                )
        print("Fixed-N study complete. Render report with scripts/render_ab_study_report.py")
        return 0

    # Exploratory
    rounds = 0
    conn = connect(args.registry_db)
    init_schema(conn)
    conn.close()
    while rounds < args.max_per_arm:
        if args.pairing == "paired":
            _attempt_eval(
                arm="a",
                block=rounds,
                overrides=a_overrides,
                study_dir=study_dir,
                eval_output_dir=eval_output_dir,
                base_config=args.base_config,
                args=args,
                manifest_path=manifest_path,
                study_id=study_id,
            )
            _attempt_eval(
                arm="b",
                block=rounds,
                overrides=b_overrides,
                study_dir=study_dir,
                eval_output_dir=eval_output_dir,
                base_config=args.base_config,
                args=args,
                manifest_path=manifest_path,
                study_id=study_id,
            )
        else:
            _attempt_eval(
                arm="a",
                block=None,
                overrides=a_overrides,
                study_dir=study_dir,
                eval_output_dir=eval_output_dir,
                base_config=args.base_config,
                args=args,
                manifest_path=manifest_path,
                study_id=study_id,
            )
            _attempt_eval(
                arm="b",
                block=None,
                overrides=b_overrides,
                study_dir=study_dir,
                eval_output_dir=eval_output_dir,
                base_config=args.base_config,
                args=args,
                manifest_path=manifest_path,
                study_id=study_id,
            )
        rounds += 1
        rec = _load_manifest_runs(manifest_path)
        if args.pairing == "paired":
            a_ids, b_ids = paired_run_ids_from_manifest_records(rec, arm_a="a", arm_b="b")
        else:
            a_ids = _successful_ids_by_arm(rec, "a")
            b_ids = _successful_ids_by_arm(rec, "b")
        if rounds >= args.min_per_arm:
            conn = connect(args.registry_db)
            try:
                stop, pval = _exploratory_stop(
                    conn=conn,
                    dataset_key=dataset_key,
                    primary=args.primary_metric,
                    a_ids=a_ids,
                    b_ids=b_ids,
                    paired=(args.pairing == "paired") and len(a_ids) == len(b_ids) and len(a_ids) >= 2,
                    alpha=args.alpha,
                )
            finally:
                conn.close()
            if stop:
                print(
                    f"Exploratory stop: primary={args.primary_metric!r} p={pval:.6f} < alpha={args.alpha} "
                    f"after {rounds} round(s). (Label analysis exploratory.)"
                )
                return 0

    print("Exploratory stop: reached --max-per-arm without significance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

