"""Load eval run folders into the SQLite registry."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import yaml


def hybrid_enabled_from_meta(meta: dict) -> bool:
    """Prefer meta['retrieval_features']['hybrid_enabled']; else parse default.yaml from meta['config']."""
    rf = meta.get("retrieval_features")
    if isinstance(rf, dict) and "hybrid_enabled" in rf:
        return bool(rf["hybrid_enabled"])
    cfg_path = meta.get("config")
    if not cfg_path:
        raise ValueError("meta.json missing config path and retrieval_features")
    path = Path(cfg_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found for hybrid parse: {path}")
    with open(path, encoding="utf-8") as f:
        default_cfg = yaml.safe_load(f)
    retrieval = default_cfg.get("retrieval") or {}
    hybrid = retrieval.get("hybrid") or {}
    return bool(hybrid.get("enabled", False))


def ingest_run_dir(conn: sqlite3.Connection, run_dir: Path) -> None:
    """Upsert one artifacts/eval_runs/<UTC>/ folder (meta.json + each dataset report.json)."""
    run_dir = run_dir.resolve()
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    run_id = meta.get("run_id", run_dir.name)
    hybrid = hybrid_enabled_from_meta(meta)

    for ds in meta.get("datasets", []):
        ds_key = ds["key"]
        report_path = run_dir / ds_key / "report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"Missing report.json: {report_path}")

        with open(report_path, encoding="utf-8") as f:
            report: dict[str, Any] = json.load(f)

        git_commit = meta.get("git_commit")
        if git_commit is not None:
            git_commit = str(git_commit)

        mean_scores = ds.get("mean_scores") or {}
        eval_ctx = meta.get("eval_context") or {}

        cur = conn.execute(
            "SELECT id FROM runs WHERE run_id = ? AND dataset_key = ?",
            (run_id, ds_key),
        )
        row = cur.fetchone()
        if row:
            run_row_id = row[0]
            conn.execute(
                """UPDATE runs SET dataset_path = ?, pass_rate = ?, mean_scores_json = ?,
                core_threshold = ?, retrieval_threshold = ?, eval_context_json = ?,
                git_commit = ?, hybrid_enabled = ?
                WHERE id = ?""",
                (
                    ds["path"],
                    float(ds["pass_rate"]),
                    json.dumps(mean_scores),
                    float(meta["core_threshold"]),
                    float(meta["retrieval_threshold"]),
                    json.dumps(eval_ctx),
                    git_commit,
                    1 if hybrid else 0,
                    run_row_id,
                ),
            )
            conn.execute("DELETE FROM sample_results WHERE run_row_id = ?", (run_row_id,))
        else:
            conn.execute(
                """INSERT INTO runs (
                    run_id, dataset_key, dataset_path, pass_rate, mean_scores_json,
                    core_threshold, retrieval_threshold, eval_context_json, git_commit, hybrid_enabled
                ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    ds_key,
                    ds["path"],
                    float(ds["pass_rate"]),
                    json.dumps(mean_scores),
                    float(meta["core_threshold"]),
                    float(meta["retrieval_threshold"]),
                    json.dumps(eval_ctx),
                    git_commit,
                    1 if hybrid else 0,
                ),
            )
            run_row_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

        for res in report.get("results", []):
            sample_id = res["sample_id"]
            scores_list = res.get("scores", [])
            if res.get("passed_all") is not None:
                passed_all = 1 if res["passed_all"] else 0
            else:
                passed_all = 1 if all(s.get("passed") for s in scores_list) else 0
            metrics = [
                {
                    "metric_name": s["metric_name"],
                    "score": s["score"],
                    "threshold": s["threshold"],
                    "passed": s["passed"],
                }
                for s in scores_list
            ]
            conn.execute(
                """INSERT INTO sample_results (run_row_id, sample_id, passed_all, metrics_json)
                VALUES (?,?,?,?)""",
                (run_row_id, sample_id, passed_all, json.dumps(metrics)),
            )

    conn.commit()


def iter_run_dirs(base: Path) -> list[Path]:
    """Return sorted subdirs of base that contain meta.json (UTC stamp folders)."""
    if not base.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and (p / "meta.json").is_file():
            out.append(p)
    return out
