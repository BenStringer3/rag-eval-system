"""Eval YAML dataset registry."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.eval.eval_config import enabled_dataset_paths

ROOT = Path(__file__).resolve().parents[1]


def test_enabled_dataset_paths_includes_starter_and_synthetic():
    paths = enabled_dataset_paths(ROOT / "configs/eval.yaml", project_root=ROOT)
    by_key = dict(paths)
    assert "starter" in by_key and "synthetic" in by_key
    assert by_key["starter"] == ROOT / "data/eval_datasets/starter.json"
    assert by_key["synthetic"] == ROOT / "data/eval_datasets/synthetic.json"


def test_enabled_dataset_paths_respects_enabled_false(tmp_path: Path):
    cfg = {
        "datasets": {
            "a": {"enabled": True, "path": "data/x.json"},
            "b": {"enabled": False, "path": "data/y.json"},
        }
    }
    p = tmp_path / "eval.yaml"
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.json").write_text('{"name":"x","samples":[]}', encoding="utf-8")
    (tmp_path / "data" / "y.json").write_text('{"name":"y","samples":[]}', encoding="utf-8")

    out = enabled_dataset_paths(p, project_root=tmp_path)
    assert [k for k, _ in out] == ["a"]
