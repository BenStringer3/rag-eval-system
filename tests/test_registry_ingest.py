"""Tests for eval registry ingest."""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.registry_db import connect, init_schema
from src.eval.registry_ingest import hybrid_enabled_from_meta, ingest_run_dir


def test_hybrid_from_retrieval_features() -> None:
    assert hybrid_enabled_from_meta({"retrieval_features": {"hybrid_enabled": True}}) is True
    assert hybrid_enabled_from_meta({"retrieval_features": {"hybrid_enabled": False}}) is False


def test_hybrid_from_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "default.yaml"
    cfg.write_text(
        "retrieval:\n  hybrid:\n    enabled: true\n",
        encoding="utf-8",
    )
    meta = {"config": str(cfg)}
    assert hybrid_enabled_from_meta(meta) is True


def test_ingest_minimal_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260101T000000Z"
    run_dir.mkdir()
    (run_dir / "single").mkdir()
    meta = {
        "run_id": "20260101T000000Z",
        "datasets": [
            {
                "key": "single",
                "path": str(tmp_path / "synthetic.json"),
                "pass_rate": 0.5,
                "mean_scores": {"Faithfulness": 0.8},
            }
        ],
        "core_threshold": 0.7,
        "retrieval_threshold": 0.6,
        "eval_context": {
            "generation_model": "m",
            "generation_temperature": 0.0,
            "judge_model": "j",
            "judge_temperature": 0,
        },
        "retrieval_features": {"hybrid_enabled": True},
        "git_commit": "abc1234",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    report = {
        "dataset_name": "synthetic",
        "results": [
            {
                "sample_id": "syn001",
                "scores": [
                    {
                        "metric_name": "Faithfulness",
                        "score": 1.0,
                        "threshold": 0.7,
                        "passed": True,
                    }
                ],
            }
        ],
        "config": {},
    }
    (run_dir / "single" / "report.json").write_text(json.dumps(report), encoding="utf-8")

    db_path = tmp_path / "reg.db"
    conn = connect(db_path)
    init_schema(conn)
    ingest_run_dir(conn, run_dir)
    conn.close()

    conn = connect(db_path)
    r = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    s = conn.execute("SELECT COUNT(*) FROM sample_results").fetchone()[0]
    h = conn.execute("SELECT hybrid_enabled FROM runs").fetchone()[0]
    conn.close()
    assert r == 1
    assert s == 1
    assert h == 1
