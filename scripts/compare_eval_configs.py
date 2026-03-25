#!/usr/bin/env python3
"""Compare mean metric values across two groups of runs (unpaired or paired t-test).

Uses scipy.stats; install dev extras if needed: pip install -e '.[dev]' (scipy in dev).

Groups are lists of run_id values (same dataset). Metric names match mean_scores_json keys
(e.g. Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall).

Example::

    .venv/bin/python scripts/compare_eval_configs.py \\
      --dataset single \\
      --metric Faithfulness \\
      --group-a 20260324T175312Z,20260324T180009Z \\
      --group-b 20260324T195433Z
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
from src.eval.registry_stats import load_metric_values  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-sample t-test on run-level mean metric values from the eval registry.",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=ROOT / "data" / "eval_registry.db",
        help="SQLite path (default: data/eval_registry.db).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="dataset_key (e.g. single).",
    )
    p.add_argument(
        "--metric",
        type=str,
        required=True,
        help="Key in mean_scores_json (e.g. Faithfulness).",
    )
    p.add_argument(
        "--group-a",
        type=str,
        required=True,
        help="Comma-separated run_id list for group A.",
    )
    p.add_argument(
        "--group-b",
        type=str,
        required=True,
        help="Comma-separated run_id list for group B.",
    )
    p.add_argument(
        "--paired",
        action="store_true",
        help="Use paired t-test (groups must have equal length; order matters).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        from scipy import stats
    except ImportError:
        print("Install scipy: pip install scipy  (or pip install -e '.[dev]')", file=sys.stderr)
        return 1

    if not args.db.is_file():
        print(f"Database not found: {args.db}", file=sys.stderr)
        return 1

    group_a = [x.strip() for x in args.group_a.split(",") if x.strip()]
    group_b = [x.strip() for x in args.group_b.split(",") if x.strip()]
    if len(group_a) < 2 or len(group_b) < 2:
        print("Each group should have at least 2 runs for a meaningful t-test.", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(args.db))
    try:
        try:
            va = load_metric_values(conn, args.dataset, args.metric, group_a)
            vb = load_metric_values(conn, args.dataset, args.metric, group_b)
        except KeyError as e:
            raise SystemExit(str(e)) from e
    finally:
        conn.close()

    if args.paired:
        if len(va) != len(vb):
            print("Paired mode requires equal group sizes.", file=sys.stderr)
            return 1
        res = stats.ttest_rel(va, vb)
        test_name = "paired t-test"
    else:
        res = stats.ttest_ind(va, vb, equal_var=False)
        test_name = "Welch's t-test (unequal variance)"

    print(f"Metric: {args.metric}")
    print(f"Dataset: {args.dataset}")
    print(f"Test: {test_name}")
    print(f"Group A (n={len(va)}): mean={sum(va)/len(va):.6f} values={va}")
    print(f"Group B (n={len(vb)}): mean={sum(vb)/len(vb):.6f} values={vb}")
    print(f"statistic={res.statistic:.6f} pvalue={res.pvalue:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
