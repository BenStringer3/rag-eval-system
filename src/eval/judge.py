"""DeepEval judge model wired to LM Studio (OpenAI-compatible API)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from deepeval.models import GPTModel

# DeepEval forwards unknown GPTModel kwargs to OpenAI(); only client ctor args belong there.


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_judge_model(default_cfg: dict[str, Any], eval_cfg: dict[str, Any]) -> GPTModel:
    """Build GPTModel for DeepEval metrics from merged pipeline + eval config."""
    judge = eval_cfg["judge"]
    provider = judge["provider"]
    if provider == "local":
        local = default_cfg["lm_studio"]
        base_url = local["base_url"]
        api_key = local["api_key"]
    elif provider == "openai":
        openai_cfg = judge["openai"]
        base_url = openai_cfg["base_url"]
        api_key_env = openai_cfg["api_key_env"]
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing required environment variable for openai judge provider: {api_key_env}"
            )
    else:
        raise ValueError(
            f"Unsupported judge provider '{provider}'. Expected one of: local, openai."
        )

    gen_kw = judge.get("generation_kwargs")
    model = GPTModel(
        model=judge["model"],
        base_url=base_url,
        api_key=api_key,
        temperature=judge.get("temperature", 0.0),
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        generation_kwargs=dict(gen_kw) if gen_kw else None,
    )
    return model


def judge_model_from_config_files(
    default_path: str | Path = "configs/default.yaml",
    eval_path: str | Path = "configs/eval.yaml",
) -> GPTModel:
    """Load configs from disk and return a provider-configured DeepEval judge."""
    dp = Path(default_path)
    ep = Path(eval_path)
    if not dp.exists():
        raise FileNotFoundError(f"Pipeline config not found: {dp}")
    if not ep.exists():
        raise FileNotFoundError(f"Eval config not found: {ep}")
    return build_judge_model(_load_yaml(dp), _load_yaml(ep))
