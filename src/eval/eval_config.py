"""Parse evaluation YAML for dataset registry (paths, enable flags)."""

from __future__ import annotations

from pathlib import Path

import yaml


def enabled_dataset_paths(
    eval_config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> list[tuple[str, Path]]:
    """Return ``(dataset_key, absolute_json_path)`` for each enabled dataset.

    A dataset is skipped only when ``enabled: false`` is set explicitly.
    Paths in YAML are relative to ``project_root`` (repo root); defaults to
    the parent of the eval config's directory's parent (``../..`` from
    ``configs/eval.yaml`` → repo root).
    """
    eval_config_path = Path(eval_config_path).resolve()
    if project_root is None:
        project_root = eval_config_path.parent.parent
    else:
        project_root = Path(project_root).resolve()

    with open(eval_config_path) as f:
        cfg = yaml.safe_load(f)

    block = cfg.get("datasets") or {}
    out: list[tuple[str, Path]] = []
    for key, spec in block.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("enabled") is False:
            continue
        rel = spec.get("path")
        if not rel:
            raise ValueError(f"datasets.{key} missing 'path' in {eval_config_path}")
        path = (project_root / rel).resolve()
        out.append((key, path))

    if not out:
        raise ValueError(f"No enabled datasets in {eval_config_path}")

    return out
