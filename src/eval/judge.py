"""DeepEval judge model wired to LM Studio (OpenAI-compatible API)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from deepeval.models import GPTModel

# DeepEval forwards unknown GPTModel kwargs to OpenAI(); only client ctor args belong there.
# Qwen "disable thinking" belongs on each chat.completions request (generation_kwargs / extra_body).

_THINK_PATCHED = False

# Qwen3-style reasoning fences and common typos; remove so trim_and_load_json sees real JSON.
_THINK_BLOCK = re.compile(
    r"<\s*(?:think|thinking|thnking)\s*>.*?</\s*(?:think|thinking|thnking)\s*>",
    re.DOTALL | re.IGNORECASE,
)


def _strip_thinking_fences(text: str) -> str:
    if not text:
        return text
    cleaned = _THINK_BLOCK.sub("", text)
    return cleaned.strip()


def _patch_deepeval_trim_for_thinking() -> None:
    """Wrap DeepEval's JSON extractor once so Qwen-style  blocks don't break metrics."""
    global _THINK_PATCHED
    if _THINK_PATCHED:
        return
    from deepeval.models.llms import utils as llm_utils

    _orig = llm_utils.trim_and_load_json

    def _trim_and_load_json_stripped(input_string: str):
        return _orig(_strip_thinking_fences(input_string))

    llm_utils.trim_and_load_json = _trim_and_load_json_stripped
    _THINK_PATCHED = True


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_judge_model(default_cfg: dict[str, Any], eval_cfg: dict[str, Any]) -> GPTModel:
    """Build GPTModel for DeepEval metrics from merged pipeline + eval config."""
    _patch_deepeval_trim_for_thinking()
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
    return GPTModel(
        model=judge["model"],
        base_url=base_url,
        api_key=api_key,
        temperature=judge.get("temperature", 0.0),
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        generation_kwargs=dict(gen_kw) if gen_kw else None,
    )


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
