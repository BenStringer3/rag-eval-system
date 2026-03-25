"""MLflow-native evaluation scorers and threshold helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.deepeval import (
    AnswerRelevancy,
    ContextualPrecision,
    ContextualRecall,
    Faithfulness,
)

_SCORER_FACTORIES = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "contextual_precision": ContextualPrecision,
    "contextual_recall": ContextualRecall,
}


def load_eval_config(eval_config_path: str | Path) -> dict[str, Any]:
    with open(eval_config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def configure_judge_environment(eval_config: dict[str, Any]) -> None:
    judge_cfg = eval_config.get("judge") or {}
    openai_cfg = judge_cfg.get("openai") or {}

    base_url = openai_cfg.get("base_url")
    if base_url:
        os.environ["OPENAI_API_BASE"] = str(base_url)

    api_key = None
    api_key_env = openai_cfg.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(str(api_key_env))
    if api_key is None:
        api_key = openai_cfg.get("api_key")
    if api_key:
        os.environ["OPENAI_API_KEY"] = str(api_key)


def metric_thresholds(eval_config: dict[str, Any]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for metric_name, spec in (eval_config.get("metrics") or {}).items():
        if not isinstance(spec, dict) or spec.get("enabled") is False:
            continue
        if metric_name not in _SCORER_FACTORIES:
            continue
        thresholds[metric_name] = float(spec["threshold"])
    return thresholds


def metric_pass_mask(series: pd.Series, threshold: float) -> pd.Series:
    """Row-wise pass for one `{metric}/value` column from mlflow.genai.evaluate().

    DeepEval scorers set Feedback.value to yes/no; other scorers may log numeric scores.
    """
    lowered = series.astype(str).str.strip().str.lower()
    if len(series) > 0 and bool(lowered.isin(["yes", "no"]).all()):
        return lowered == "yes"
    vals = pd.to_numeric(series, errors="coerce")
    return vals >= float(threshold)


def build_scorers(eval_config_path: str | Path) -> list[Scorer]:
    eval_config = load_eval_config(eval_config_path)
    configure_judge_environment(eval_config)

    model_uri = (eval_config.get("judge") or {}).get("model")
    if not model_uri:
        raise ValueError(f"judge.model is required in {eval_config_path}")

    thresholds = metric_thresholds(eval_config)
    scorers: list[Scorer] = []
    for metric_name, threshold in thresholds.items():
        scorer_cls = _SCORER_FACTORIES[metric_name]
        scorer = scorer_cls(model=model_uri, threshold=threshold)
        scorer.name = metric_name
        scorers.append(scorer)
    if not scorers:
        raise ValueError(f"No enabled MLflow scorers found in {eval_config_path}")
    return scorers
