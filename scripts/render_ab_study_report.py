#!/usr/bin/env python3
"""Build markdown + Plotly HTML figures for an A/B study folder.

Reads ``artifacts/ab_study/<id>/study_meta.json`` and ``manifest.jsonl`` and pulls
run aggregates from the eval registry SQLite DB.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.ab_study_config import paired_run_ids_from_manifest_records  # noqa: E402
from src.eval.registry_stats import (  # noqa: E402
    bootstrap_ci_mean_diff,
    fetch_run_rows_for_study,
    load_series,
    two_sample_test,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render A/B study report under docs/.")
    p.add_argument(
        "--study-dir",
        type=Path,
        required=True,
        help="Path to artifacts/ab_study/<study_id> (contains manifest.jsonl).",
    )
    p.add_argument("--db", type=Path, default=ROOT / "data" / "eval_registry.db")
    p.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Output markdown (default: docs/ab-study-<study_id>.md).",
    )
    return p.parse_args()


def _load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _counts_by_arm(manifest: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    ok: dict[str, int] = defaultdict(int)
    bad: dict[str, int] = defaultdict(int)
    for r in manifest:
        arm = r.get("arm", "?")
        if r.get("status") == "success":
            ok[arm] += 1
        else:
            bad[arm] += 1
    return dict(ok), dict(bad)


def _run_ids_by_arm(manifest: list[dict]) -> tuple[list[str], list[str]]:
    a: list[str] = []
    b: list[str] = []
    for rec in manifest:
        if rec.get("status") != "success" or not rec.get("run_id"):
            continue
        if rec.get("arm") == "a":
            a.append(rec["run_id"])
        elif rec.get("arm") == "b":
            b.append(rec["run_id"])
    return a, b


def main() -> int:
    args = _parse_args()
    study_dir = args.study_dir.resolve()
    manifest_path = study_dir / "manifest.jsonl"
    meta_path = study_dir / "study_meta.json"
    if not manifest_path.is_file():
        print(f"Missing {manifest_path}", file=sys.stderr)
        return 1
    if not args.db.is_file():
        print(f"Registry DB not found: {args.db}", file=sys.stderr)
        return 1

    study_meta: dict = {}
    if meta_path.is_file():
        study_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    study_id = study_meta.get("study_id", study_dir.name)
    dataset_key = study_meta.get("dataset_key", "single")

    label_a = study_meta.get("arm_a_label", "arm_a")
    label_b = study_meta.get("arm_b_label", "arm_b")

    manifest = _load_manifest(manifest_path)
    ok_counts, fail_counts = _counts_by_arm(manifest)
    a_ids, b_ids = _run_ids_by_arm(manifest)
    paired_a, paired_b = paired_run_ids_from_manifest_records(manifest, arm_a="a", arm_b="b")

    out_md = args.out_md or (ROOT / "docs" / f"ab-study-{study_id}.md")
    fig_dir = ROOT / "docs" / "_figures" / f"ab-study-{study_id}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rel_fig = f"_figures/ab-study-{study_id}"

    conn = sqlite3.connect(str(args.db))
    try:
        rows_a = fetch_run_rows_for_study(conn, dataset_key, a_ids)
        rows_b = fetch_run_rows_for_study(conn, dataset_key, b_ids)
    finally:
        conn.close()

    metric_names: list[str] = []
    if rows_a:
        metric_names = sorted(rows_a[0]["mean_scores"].keys())
    elif rows_b:
        metric_names = sorted(rows_b[0]["mean_scores"].keys())

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Install plotly.", file=sys.stderr)
        return 1

    pr_a = [r["pass_rate"] for r in rows_a]
    pr_b = [r["pass_rate"] for r in rows_b]
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Box(y=pr_a, name=label_a, boxmean="sd"))
    fig_pr.add_trace(go.Box(y=pr_b, name=label_b, boxmean="sd"))
    fig_pr.update_layout(title_text="Pass rate by arm", yaxis_title="pass_rate")
    pr_html = fig_dir / "pass_rate.html"
    fig_pr.write_html(str(pr_html), include_plotlyjs="cdn")

    if metric_names:
        ncols = 2
        nrows = (len(metric_names) + ncols - 1) // ncols
        fig_m = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=metric_names,
            vertical_spacing=0.12,
        )
        for i, name in enumerate(metric_names):
            row_i, col_i = i // ncols + 1, i % ncols + 1
            va = [float(row["mean_scores"].get(name, 0)) for row in rows_a]
            vb = [float(row["mean_scores"].get(name, 0)) for row in rows_b]
            fig_m.add_trace(
                go.Box(y=va, name=label_a, legendgroup="a", showlegend=(i == 0)),
                row=row_i,
                col=col_i,
            )
            fig_m.add_trace(
                go.Box(y=vb, name=label_b, legendgroup="b", showlegend=(i == 0)),
                row=row_i,
                col=col_i,
            )
        fig_m.update_layout(height=260 * nrows, title_text="Mean metric scores by arm")
        m_html = fig_dir / "metrics.html"
        fig_m.write_html(str(m_html), include_plotlyjs="cdn")
    else:
        m_html = None

    lines_stats: list[str] = []
    use_paired = study_meta.get("pairing") == "paired" and len(paired_a) >= 2 and len(paired_a) == len(paired_b)
    conn = sqlite3.connect(str(args.db))
    try:
        series_to_test = ["pass_rate", *metric_names]
        for sname in series_to_test:
            try:
                if use_paired:
                    a = load_series(conn, dataset_key, sname, paired_a)
                    b = load_series(conn, dataset_key, sname, paired_b)
                else:
                    a = load_series(conn, dataset_key, sname, a_ids)
                    b = load_series(conn, dataset_key, sname, b_ids)
                if len(a) < 2 or len(b) < 2:
                    continue
                res = two_sample_test(b, a, paired=use_paired)  # B minus A by convention
                try:
                    _, lo, hi = bootstrap_ci_mean_diff(b, a, n_bootstrap=2000)
                    ci_txt = f", boot 95% CI Δ(B−A) [{lo:.4f}, {hi:.4f}]"
                except Exception:
                    ci_txt = ""
                lines_stats.append(
                    f"| {sname} | {res.test_name} | n={res.n_a} | "
                    f"mean_A={res.mean_b:.4f} mean_B={res.mean_a:.4f} | "
                    f"t={res.statistic:.4f} p={res.pvalue:.6f}{ci_txt} |"
                )
            except (KeyError, ValueError) as e:
                lines_stats.append(f"| {sname} | (skip) | | | {e} |")
    finally:
        conn.close()

    design = "exploratory sequential" if study_meta.get("exploratory") else "fixed-N"
    disclaimer = (
        "Exploratory stopping inflates Type I error; treat p-values as heuristic unless "
        "a fixed-N preregistered protocol was used."
        if study_meta.get("exploratory")
        else "Fixed-N design: interpret p-values at face value for the pre-specified sample size."
    )

    md: list[str] = [
        f"# A/B study `{study_id}`",
        "",
        "## Methods",
        "",
        f"- **Design:** {design}",
        f"- **Pairing:** {study_meta.get('pairing', '?')}",
        f"- **Dataset key:** `{dataset_key}`",
        f"- **Arm A:** {label_a}",
        f"- **Arm B:** {label_b}",
        "",
        disclaimer,
        "",
        "## Attempt counts",
        "",
        f"- Successful by arm: `{ok_counts!r}`",
        f"- Failed by arm: `{fail_counts!r}`",
        "",
        "## Figures",
        "",
        f"- [Pass rate (interactive)]({rel_fig}/pass_rate.html)",
    ]
    if m_html:
        md.append(f"- [Metrics (interactive)]({rel_fig}/metrics.html)")
    md.extend(
        [
            "",
            "## Statistical summaries (B minus A)",
            "",
            "| Series | Test | n | means | t / p |",
            "|--------|------|---|-------|-------|",
            *lines_stats,
            "",
            f"- Study artifacts: `{study_dir}`",
            f"- Manifest: `{manifest_path.name}`",
            "",
        ]
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

