"""Tests for MLflow scorer configuration."""

from __future__ import annotations

import os
from pathlib import Path

from src.eval.scorers import build_scorers, metric_thresholds


def test_build_scorers_and_thresholds(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    eval_config = tmp_path / "eval.yaml"
    eval_config.write_text(
        """
judge:
  model: "openai:/gpt-4o-mini"
  openai:
    base_url: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"
metrics:
  faithfulness:
    enabled: true
    threshold: 0.7
  answer_relevancy:
    enabled: false
    threshold: 0.7
  contextual_precision:
    enabled: true
    threshold: 0.6
tracking:
  uri: "sqlite:///data/mlflow.db"
""".strip(),
        encoding="utf-8",
    )

    scorers = build_scorers(eval_config)
    thresholds = metric_thresholds(
        {
            "metrics": {
                "faithfulness": {"enabled": True, "threshold": 0.7},
                "answer_relevancy": {"enabled": False, "threshold": 0.7},
                "contextual_precision": {"enabled": True, "threshold": 0.6},
            }
        }
    )

    assert [scorer.name for scorer in scorers] == ["faithfulness", "contextual_precision"]
    assert thresholds == {"faithfulness": 0.7, "contextual_precision": 0.6}
    assert os.environ["OPENAI_API_BASE"] == "https://api.openai.com/v1"
    assert os.environ["OPENAI_API_KEY"] == "test-key"
