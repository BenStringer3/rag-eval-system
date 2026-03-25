"""Tests for MLflow dataset adapters."""

from __future__ import annotations

from pathlib import Path

from src.eval.datasets import load_mlflow_eval_data


def test_load_mlflow_eval_data_maps_expected_fields(tmp_path: Path) -> None:
    dataset_path = tmp_path / "eval.json"
    dataset_path.write_text(
        """
{
  "name": "synthetic",
  "samples": [
    {
      "id": "syn001",
      "query": "What is MLflow?",
      "expected_answer": "A tracking system.",
      "expected_context": ["MLflow tracks experiments."],
      "metadata": {"category": "docs"}
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    rows = load_mlflow_eval_data(dataset_path)

    assert rows == [
        {
            "request_id": "syn001",
            "inputs": {"question": "What is MLflow?"},
            "expectations": {
                "expected_output": "A tracking system.",
                "sample_id": "syn001",
                "expected_context": ["MLflow tracks experiments."],
                "metadata": {"category": "docs"},
            },
            "tags": {"dataset_name": "synthetic", "sample_id": "syn001"},
        }
    ]
