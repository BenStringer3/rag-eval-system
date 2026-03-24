"""DeepEval judge model wired to LM Studio (OpenAI-compatible API)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import yaml
from deepeval.models import GPTModel

# DeepEval forwards unknown GPTModel kwargs to OpenAI(); only client ctor args belong there.


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _looks_like_length_finish_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "lengthfinishreasonerror" in msg or (
        "finish_reason" in msg and "length" in msg
    )


def _approx_tokens(text: str) -> int:
    # Fast heuristic: ~4 characters per token for English-heavy text.
    return max(1, (len(text) + 3) // 4)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


class TracedGPTModel(GPTModel):
    """GPTModel wrapper that logs length-finish judge errors to JSONL."""

    def __init__(
        self,
        *args: Any,
        judge_provider: str,
        judge_model_name: str,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._judge_provider = judge_provider
        self._judge_model_name = judge_model_name

    def _length_error_log_path(self) -> Path:
        explicit = os.environ.get("RAG_EVAL_JUDGE_TRACE_PATH")
        if explicit:
            return Path(explicit)
        return Path("artifacts/eval_runs/judge_length_errors.jsonl")

    def _log_length_finish_error(self, prompt: str, exc: Exception) -> None:
        max_completion_tokens = self.generation_kwargs.get("max_completion_tokens")
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "judge_length_finish_error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "run_id": os.environ.get("RAG_EVAL_RUN_ID"),
            "dataset_key": os.environ.get("RAG_EVAL_DATASET_KEY"),
            "dataset_path": os.environ.get("RAG_EVAL_DATASET_PATH"),
            "judge_provider": self._judge_provider,
            "judge_model": self._judge_model_name,
            "max_completion_tokens": max_completion_tokens,
            "prompt_chars": len(prompt),
            "prompt_est_tokens": _approx_tokens(prompt),
        }
        _append_jsonl(self._length_error_log_path(), payload)

    def generate(self, prompt: str, schema: Any | None = None):  # type: ignore[override]
        try:
            return super().generate(prompt=prompt, schema=schema)
        except Exception as exc:
            if _looks_like_length_finish_error(exc):
                self._log_length_finish_error(prompt=prompt, exc=exc)
            raise

    async def a_generate(self, prompt: str, schema: Any | None = None):  # type: ignore[override]
        try:
            return await super().a_generate(prompt=prompt, schema=schema)
        except Exception as exc:
            if _looks_like_length_finish_error(exc):
                self._log_length_finish_error(prompt=prompt, exc=exc)
            raise


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
    model = TracedGPTModel(
        model=judge["model"],
        base_url=base_url,
        api_key=api_key,
        temperature=judge.get("temperature", 0.0),
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        generation_kwargs=dict(gen_kw) if gen_kw else None,
        judge_provider=provider,
        judge_model_name=judge["model"],
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
