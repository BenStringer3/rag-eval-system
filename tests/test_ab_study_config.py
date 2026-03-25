"""Tests for A/B study config helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.eval.ab_study_config import (
    eval_run_dir_complete,
    paired_run_ids_from_manifest_records,
    write_yaml_with_overrides,
)


def test_write_yaml_with_overrides(tmp_path: Path) -> None:
    base = tmp_path / "default.yaml"
    base.write_text(
        "retrieval:\n  hybrid:\n    enabled: true\n  top_k: 5\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.yaml"
    write_yaml_with_overrides(base, out, overrides={"retrieval.hybrid.enabled": False})
    with open(out, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["retrieval"]["hybrid"]["enabled"] is False


def test_paired_run_ids_from_manifest_records() -> None:
    rec = [
        {"status": "success", "arm": "a", "block": 1, "run_id": "a1"},
        {"status": "success", "arm": "b", "block": 1, "run_id": "b1"},
        {"status": "success", "arm": "a", "block": 0, "run_id": "a0"},
        {"status": "success", "arm": "b", "block": 0, "run_id": "b0"},
    ]
    a, b = paired_run_ids_from_manifest_records(rec, arm_a="a", arm_b="b")
    assert a == ["a0", "a1"]
    assert b == ["b0", "b1"]


def test_set_by_path_rejects_non_dict(tmp_path: Path) -> None:
    base = tmp_path / "default.yaml"
    base.write_text("retrieval: 5\n", encoding="utf-8")
    out = tmp_path / "out.yaml"
    with pytest.raises(ValueError):
        write_yaml_with_overrides(base, out, overrides={"retrieval.hybrid.enabled": True})


def test_eval_run_dir_complete(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260101T000000Z"
    run_dir.mkdir()
    (run_dir / "single").mkdir()
    assert not eval_run_dir_complete(run_dir)
    meta = {
        "run_id": "20260101T000000Z",
        "datasets": [{"key": "single", "path": "/x.json", "pass_rate": 1.0, "mean_scores": {}}],
    }
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert not eval_run_dir_complete(run_dir)
    (run_dir / "single" / "report.json").write_text("{}", encoding="utf-8")
    assert eval_run_dir_complete(run_dir)

