"""Load run-level aggregates from the eval registry and run simple two-sample tests."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TwoSampleResult:
    test_name: str
    statistic: float
    pvalue: float
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float


def load_series(
    conn: sqlite3.Connection,
    dataset_key: str,
    name: str,
    run_ids: list[str],
) -> list[float]:
    """Load one scalar per run: ``pass_rate`` column or key in ``mean_scores_json``."""
    if name == "pass_rate":
        col = "pass_rate"
        out: list[float] = []
        for rid in run_ids:
            row = conn.execute(
                f"SELECT {col} FROM runs WHERE run_id = ? AND dataset_key = ?",
                (rid.strip(), dataset_key),
            ).fetchone()
            if not row:
                raise KeyError(f"No run row for run_id={rid!r} dataset_key={dataset_key!r}")
            out.append(float(row[0]))
        return out
    return load_metric_values(conn, dataset_key, name, run_ids)


def load_metric_values(
    conn: sqlite3.Connection,
    dataset_key: str,
    metric: str,
    run_ids: list[str],
) -> list[float]:
    """Mean metric values for each run_id (order matches ``run_ids``)."""
    out: list[float] = []
    for rid in run_ids:
        row = conn.execute(
            "SELECT mean_scores_json FROM runs WHERE run_id = ? AND dataset_key = ?",
            (rid.strip(), dataset_key),
        ).fetchone()
        if not row:
            raise KeyError(f"No run row for run_id={rid!r} dataset_key={dataset_key!r}")
        ms = json.loads(row[0])
        if metric not in ms:
            raise KeyError(
                f"Metric {metric!r} not in mean_scores for run {rid}: {list(ms.keys())}"
            )
        out.append(float(ms[metric]))
    return out


def fetch_run_rows_for_study(
    conn: sqlite3.Connection,
    dataset_key: str,
    run_ids: list[str],
) -> list[dict[str, Any]]:
    """Rows for reporting (pass_rate, mean_scores, hybrid, eval_context)."""
    rows: list[dict[str, Any]] = []
    for rid in run_ids:
        r = conn.execute(
            """SELECT run_id, pass_rate, mean_scores_json, hybrid_enabled, eval_context_json
               FROM runs WHERE run_id = ? AND dataset_key = ?""",
            (rid.strip(), dataset_key),
        ).fetchone()
        if not r:
            continue
        rows.append(
            {
                "run_id": r[0],
                "pass_rate": r[1],
                "mean_scores": json.loads(r[2]),
                "hybrid_enabled": bool(r[3]),
                "eval_context": json.loads(r[4]),
            }
        )
    return rows


def two_sample_test(
    values_a: list[float],
    values_b: list[float],
    *,
    paired: bool,
) -> TwoSampleResult:
    from scipy import stats

    if len(values_a) < 2 or len(values_b) < 2:
        raise ValueError("Each group needs at least 2 observations for t-test.")
    if paired:
        if len(values_a) != len(values_b):
            raise ValueError("Paired mode requires equal-length groups.")
        res = stats.ttest_rel(values_a, values_b)
        name = "paired t-test"
    else:
        res = stats.ttest_ind(values_a, values_b, equal_var=False)
        name = "Welch t-test"
    ma = float(np.mean(values_a))
    mb = float(np.mean(values_b))
    return TwoSampleResult(
        test_name=name,
        statistic=float(res.statistic),
        pvalue=float(res.pvalue),
        n_a=len(values_a),
        n_b=len(values_b),
        mean_a=ma,
        mean_b=mb,
    )


def bootstrap_ci_mean(
    values: list[float],
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (point_mean, low, high) percentile bootstrap CI for the mean."""
    if not values:
        raise ValueError("empty values")
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    n = len(arr)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(arr[idx]))
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(np.mean(arr)), lo, hi


def bootstrap_ci_mean_diff(
    values_a: list[float],
    values_b: list[float],
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Independent bootstrap: CI for mean(A) - mean(B)."""
    rng = np.random.default_rng(seed)
    a = np.array(values_a, dtype=float)
    b = np.array(values_b, dtype=float)
    na, nb = len(a), len(b)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        ia = rng.integers(0, na, size=na)
        ib = rng.integers(0, nb, size=nb)
        diffs[i] = float(np.mean(a[ia]) - np.mean(b[ib]))
    point = float(np.mean(a) - np.mean(b))
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return point, lo, hi
