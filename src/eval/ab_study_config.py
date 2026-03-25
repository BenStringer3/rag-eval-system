"""Helpers for running and validating A/B studies over repeated eval runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def paired_run_ids_from_manifest_records(
    records: list[dict[str, Any]],
    *,
    arm_a: str = "a",
    arm_b: str = "b",
) -> tuple[list[str], list[str]]:
    """Run IDs for A/B arms ordered by block index (successful rows only)."""
    by_block: dict[int, dict[str, str]] = {}
    for rec in records:
        if rec.get("status") != "success" or not rec.get("run_id"):
            continue
        b = rec.get("block")
        if b is None:
            continue
        arm = rec.get("arm")
        if arm in (arm_a, arm_b):
            by_block.setdefault(int(b), {})[arm] = rec["run_id"]

    a_list: list[str] = []
    b_list: list[str] = []
    for b in sorted(by_block):
        blk = by_block[b]
        if arm_a in blk and arm_b in blk:
            a_list.append(blk[arm_a])
            b_list.append(blk[arm_b])
    return a_list, b_list


def _set_by_dotted_path(obj: dict[str, Any], dotted: str, value: Any) -> None:
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        raise ValueError("empty dotted path")
    cur: dict[str, Any] = obj
    for p in parts[:-1]:
        nxt = cur.get(p)
        if nxt is None:
            nxt = {}
            cur[p] = nxt
        if not isinstance(nxt, dict):
            raise ValueError(f"Cannot set {dotted!r}: non-dict at {p!r}")
        cur = nxt
    cur[parts[-1]] = value


def write_yaml_with_overrides(
    base_config: Path,
    out_path: Path,
    *,
    overrides: dict[str, Any],
) -> None:
    """Copy YAML from disk and apply dotted-key overrides (e.g. ``retrieval.hybrid.enabled``)."""
    with open(base_config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping at YAML root: {base_config}")
    for k, v in overrides.items():
        _set_by_dotted_path(cfg, str(k), v)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def eval_run_dir_complete(run_dir: Path) -> bool:
    """True when ``meta.json`` exists and every dataset has ``<key>/report.json``."""
    run_dir = run_dir.resolve()
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        return False
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    for ds in meta.get("datasets", []):
        key = ds.get("key")
        if not key:
            return False
        if not (run_dir / key / "report.json").is_file():
            return False
    return bool(meta.get("datasets"))

