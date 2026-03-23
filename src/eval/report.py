"""Evaluation report generation and export.

Converts EvalReport into human-readable formats: markdown summary,
JSON export, and CSV for spreadsheet analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data.schemas import EvalReport


def to_markdown(report: EvalReport) -> str:
    """Render an EvalReport as a markdown summary."""
    lines = [
        f"# Evaluation Report: {report.dataset_name}",
        "",
        f"**Pass rate:** {report.pass_rate:.1%}  ",
        f"**Samples:** {len(report.results)}  ",
        "",
        "## Metric Averages",
        "",
        "| Metric | Mean Score |",
        "|--------|-----------|",
    ]

    for name, score in report.mean_scores.items():
        lines.append(f"| {name} | {score:.3f} |")

    lines.extend(["", "## Per-Sample Results", ""])

    for result in report.results:
        status = "✅" if result.passed_all else "❌"
        lines.append(f"### {status} {result.sample_id}")
        lines.append(f"**Query:** {result.query}  ")
        lines.append(f"**Answer:** {result.generated_answer[:200]}{'...' if len(result.generated_answer) > 200 else ''}  ")
        lines.append("")
        for score in result.scores:
            passed = "✅" if score.passed else "❌"
            reason = f" — {score.reason}" if score.reason else ""
            lines.append(f"- {passed} {score.metric_name}: {score.score:.3f} (threshold: {score.threshold}){reason}")
        lines.append("")

    return "\n".join(lines)


def to_json(report: EvalReport, path: str | Path) -> None:
    """Export an EvalReport as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)
    print(f"Saved JSON report to {path}")


def to_csv(report: EvalReport, path: str | Path) -> None:
    """Export per-sample scores as a CSV for spreadsheet analysis."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all metric names
    metric_names = list(report.mean_scores.keys())

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["sample_id", "query", "passed_all"] + metric_names
        writer.writerow(header)

        for result in report.results:
            score_map = {s.metric_name: s.score for s in result.scores}
            row = [
                result.sample_id,
                result.query,
                result.passed_all,
            ] + [score_map.get(name, "") for name in metric_names]
            writer.writerow(row)

    print(f"Saved CSV report to {path}")
