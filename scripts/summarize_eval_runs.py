#!/usr/bin/env python3
"""Aggregate pass_rate and mean_scores across several eval run directories (meta.json only).

Example::

    for i in 1 2 3; do .venv/bin/python scripts/run_eval_report.py --max-concurrent 1; done
    .venv/bin/python scripts/summarize_eval_runs.py \\
        artifacts/eval_runs/20260324T175312Z \\
        artifacts/eval_runs/20260324T180009Z \\
        artifacts/eval_runs/20260324T181000Z
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def _stats(values: list[float]) -> tuple[float, float, float, float | None]:
    """min, max, median, stdev (None if len < 3)."""
    if not values:
        raise ValueError("empty values")
    lo, hi = min(values), max(values)
    med = float(statistics.median(values))
    stdev: float | None
    if len(values) >= 3:
        stdev = float(statistics.stdev(values))
    else:
        stdev = None
    return lo, hi, med, stdev


def _load_meta(run_dir: Path) -> dict:
    path = run_dir / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing meta.json: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Summarize pass_rate and mean_scores from multiple eval run folders.",
    )
    p.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Paths to artifacts/eval_runs/<UTC>/ (each must contain meta.json).",
    )
    args = p.parse_args()
    run_dirs = [d.resolve() for d in args.run_dirs]

    by_key: dict[str, dict[str, list]] = {}
    for rd in run_dirs:
        meta = _load_meta(rd)
        run_id = meta.get("run_id", rd.name)
        for ds in meta.get("datasets", []):
            key = ds["key"]
            bucket = by_key.setdefault(key, {"pass_rates": [], "mean_score_lists": []})
            bucket["pass_rates"].append((run_id, float(ds["pass_rate"])))
            bucket["mean_score_lists"].append((run_id, dict(ds["mean_scores"])))

    if not by_key:
        print("No datasets found in given meta.json files.", file=sys.stderr)
        return 1

    for key in sorted(by_key.keys()):
        b = by_key[key]
        pr_vals = [v for _, v in b["pass_rates"]]
        lo, hi, med, sd = _stats(pr_vals)
        print(f"\n## dataset: {key}  (n={len(pr_vals)} runs)")
        print(f"  pass_rate:  min={lo:.4f}  max={hi:.4f}  median={med:.4f}", end="")
        if sd is not None:
            print(f"  stdev={sd:.4f}")
        else:
            print()

        metric_names: set[str] = set()
        for _, ms in b["mean_score_lists"]:
            metric_names.update(ms.keys())
        for m in sorted(metric_names):
            vals = []
            for rid, ms in b["mean_score_lists"]:
                if m not in ms:
                    print(f"  WARNING: metric {m!r} missing in run {rid}", file=sys.stderr)
                    continue
                vals.append(float(ms[m]))
            if len(vals) != len(b["mean_score_lists"]):
                continue
            lo, hi, med, sd = _stats(vals)
            line = f"  {m}:  min={lo:.4f}  max={hi:.4f}  median={med:.4f}"
            if sd is not None:
                line += f"  stdev={sd:.4f}"
            print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
