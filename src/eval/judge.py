"""DeepEval judge model wired to LM Studio (OpenAI-compatible API)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from deepeval.models import GPTModel


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_judge_model(default_cfg: dict[str, Any], eval_cfg: dict[str, Any]) -> GPTModel:
    """Build GPTModel for DeepEval metrics from merged pipeline + eval config."""
    lm = default_cfg["lm_studio"]
    judge = eval_cfg["judge"]
    return GPTModel(
        model=judge["model"],
        base_url=lm["base_url"],
        api_key=lm["api_key"],
        temperature=judge.get("temperature", 0.0),
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
    )


def judge_model_from_config_files(
    default_path: str | Path = "configs/default.yaml",
    eval_path: str | Path = "configs/eval.yaml",
) -> GPTModel:
    """Load configs from disk and return a DeepEval judge targeting LM Studio."""
    dp = Path(default_path)
    ep = Path(eval_path)
    if not dp.exists():
        raise FileNotFoundError(f"Pipeline config not found: {dp}")
    if not ep.exists():
        raise FileNotFoundError(f"Eval config not found: {ep}")
    return build_judge_model(_load_yaml(dp), _load_yaml(ep))
