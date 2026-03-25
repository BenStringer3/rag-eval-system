"""Tests for registry_stats."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.registry_db import connect, init_schema
from src.eval.registry_stats import load_metric_values, load_series, two_sample_test


def _insert_run(
    conn,
    *,
    run_id: str,
    dataset_key: str,
    hybrid: int,
    pass_rate: float,
    mean_scores: dict,
) -> None:
    conn.execute(
        """INSERT INTO runs (
            run_id, dataset_key, dataset_path, pass_rate, mean_scores_json,
            core_threshold, retrieval_threshold, eval_context_json, git_commit, hybrid_enabled
        ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            run_id,
            dataset_key,
            "/tmp/x.json",
            pass_rate,
            json.dumps(mean_scores),
            0.7,
            0.6,
            "{}",
            None,
            hybrid,
        ),
    )


def test_two_sample_welch(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    db = tmp_path / "r.db"
    conn = connect(db)
    init_schema(conn)
    for i, pr in enumerate([0.4, 0.45, 0.42]):
        _insert_run(
            conn,
            run_id=f"d{i}",
            dataset_key="single",
            hybrid=0,
            pass_rate=pr,
            mean_scores={"Faithfulness": 0.8},
        )
    for i, pr in enumerate([0.5, 0.55, 0.52]):
        _insert_run(
            conn,
            run_id=f"h{i}",
            dataset_key="single",
            hybrid=1,
            pass_rate=pr,
            mean_scores={"Faithfulness": 0.85},
        )
    conn.commit()
    conn.close()

    conn = connect(db)
    d = load_series(conn, "single", "pass_rate", ["d0", "d1", "d2"])
    h = load_series(conn, "single", "pass_rate", ["h0", "h1", "h2"])
    res = two_sample_test(h, d, paired=False)
    assert res.n_a == 3 and res.n_b == 3
    assert res.pvalue >= 0.0
    f = load_metric_values(conn, "single", "Faithfulness", ["d0"])
    assert f == [0.8]
    conn.close()


def test_two_sample_paired(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    db = tmp_path / "r.db"
    conn = connect(db)
    init_schema(conn)
    for rid, pr in [("d0", 0.4), ("h0", 0.5), ("d1", 0.41), ("h1", 0.51)]:
        _insert_run(
            conn,
            run_id=rid,
            dataset_key="single",
            hybrid=1 if rid.startswith("h") else 0,
            pass_rate=pr,
            mean_scores={},
        )
    conn.commit()
    conn.close()
    conn = connect(db)
    d = load_series(conn, "single", "pass_rate", ["d0", "d1"])
    h = load_series(conn, "single", "pass_rate", ["h0", "h1"])
    res = two_sample_test(h, d, paired=True)
    assert res.test_name == "paired t-test"
    conn.close()
