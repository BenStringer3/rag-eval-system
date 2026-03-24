"""DeepEval judge model wired to LM Studio (OpenAI-compatible API)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from deepeval.models import GPTModel
from openai import LengthFinishReasonError

# DeepEval forwards unknown GPTModel kwargs to OpenAI(); only client ctor args belong there.

# Cap assistant text in JSONL traces (structured parse failures can be huge).
_MAX_ASSISTANT_PREVIEW_CHARS = 4000


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _approx_tokens(text: str) -> int:
    # Fast heuristic: ~4 characters per token for English-heavy text.
    return max(1, (len(text) + 3) // 4)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _is_length_finish_error(exc: BaseException, _depth: int = 0) -> bool:
    """True when the API hit max output tokens mid-response (structured parse fails)."""
    if isinstance(exc, LengthFinishReasonError):
        return True
    if _depth >= 8:
        return False
    cause = exc.__cause__
    if cause is not None and _is_length_finish_error(cause, _depth + 1):
        return True
    # Rare: re-wrapped errors; OpenAI's user-facing message is stable.
    return "length limit was reached" in str(exc).lower()


def _openai_length_finish_details(exc: LengthFinishReasonError) -> dict[str, Any]:
    out: dict[str, Any] = {}
    comp = exc.completion
    usage = comp.usage
    if usage is not None:
        out["usage"] = usage.model_dump()
    choices = comp.choices or []
    if not choices:
        return out
    ch0 = choices[0]
    fr = getattr(ch0, "finish_reason", None)
    if fr is not None:
        out["finish_reason"] = fr
    msg = getattr(ch0, "message", None)
    content = getattr(msg, "content", None) if msg is not None else None
    if content:
        cap = _MAX_ASSISTANT_PREVIEW_CHARS
        out["assistant_content_preview"] = content[:cap]
        if len(content) > cap:
            out["assistant_content_preview_truncated"] = True
    return out


def judge_length_finish_trace_record(
    *,
    prompt: str,
    exc: BaseException,
    judge_provider: str,
    judge_model: str,
    max_completion_tokens: int | None,
    run_id: str | None,
    dataset_key: str | None,
    dataset_path: str | None,
) -> dict[str, Any]:
    """Structured log line for a judge call that stopped at the output token cap."""
    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "judge_length_finish_error",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "run_id": run_id,
        "dataset_key": dataset_key,
        "dataset_path": dataset_path,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "max_completion_tokens": max_completion_tokens,
        "prompt_chars": len(prompt),
        "prompt_est_tokens": _approx_tokens(prompt),
    }
    if isinstance(exc, LengthFinishReasonError):
        payload["openai"] = _openai_length_finish_details(exc)
    return payload


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

    def _maybe_log_length_finish(self, prompt: str, exc: BaseException) -> None:
        if not _is_length_finish_error(exc):
            return
        gen_kw = self.generation_kwargs or {}
        record = judge_length_finish_trace_record(
            prompt=prompt,
            exc=exc,
            judge_provider=self._judge_provider,
            judge_model=self._judge_model_name,
            max_completion_tokens=gen_kw.get("max_completion_tokens"),
            run_id=os.environ.get("RAG_EVAL_RUN_ID"),
            dataset_key=os.environ.get("RAG_EVAL_DATASET_KEY"),
            dataset_path=os.environ.get("RAG_EVAL_DATASET_PATH"),
        )
        _append_jsonl(self._length_error_log_path(), record)

    def generate(self, prompt: str, schema: Any | None = None):  # type: ignore[override]
        try:
            return super().generate(prompt=prompt, schema=schema)
        except Exception as exc:
            self._maybe_log_length_finish(prompt, exc)
            raise

    async def a_generate(self, prompt: str, schema: Any | None = None):  # type: ignore[override]
        try:
            return await super().a_generate(prompt=prompt, schema=schema)
        except Exception as exc:
            self._maybe_log_length_finish(prompt, exc)
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
