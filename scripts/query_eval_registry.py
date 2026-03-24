#!/usr/bin/env python3
"""Query the eval registry SQLite DB (canned listings or raw SQL).

Example::

    .venv/bin/python scripts/query_eval_registry.py last -n 10
    .venv/bin/python scripts/query_eval_registry.py last --dataset single
    .venv/bin/python scripts/query_eval_registry.py sql "SELECT run_id, pass_rate FROM runs LIMIT 5"
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Query data/eval_registry.db.")
    p.add_argument(
        "--db",
        type=Path,
        default=ROOT / "data" / "eval_registry.db",
        help="SQLite path (default: data/eval_registry.db).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    last = sub.add_parser("last", help="List recent runs (by row id).")
    last.add_argument("-n", type=int, default=20, help="Max rows (default: 20).")
    last.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Filter by dataset_key (e.g. single).",
    )

    sub.add_parser("stats", help="Print row counts for runs and sample_results.")

    raw = sub.add_parser("sql", help="Run a read-only SQL query.")
    raw.add_argument("query", type=str, help="SQL SELECT statement.")

    return p.parse_args()


def _check_readonly_sql(q: str) -> None:
    t = q.strip().lower()
    if not t.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")


def main() -> int:
    args = _parse_args()
    if not args.db.is_file():
        print(f"Database not found: {args.db} (run ingest_eval_registry.py first)", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row

    if args.cmd == "stats":
        r = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        s = conn.execute("SELECT COUNT(*) FROM sample_results").fetchone()[0]
        print(f"runs: {r}")
        print(f"sample_results: {s}")
        conn.close()
        return 0

    if args.cmd == "sql":
        _check_readonly_sql(args.query)
        rows = conn.execute(args.query).fetchall()
        if not rows:
            print("(no rows)")
            conn.close()
            return 0
        cols = rows[0].keys()
        print("\t".join(cols))
        for row in rows:
            print("\t".join(str(row[c]) for c in cols))
        conn.close()
        return 0

    # last
    q = (
        "SELECT id, run_id, dataset_key, pass_rate, hybrid_enabled, git_commit "
        "FROM runs ORDER BY id DESC LIMIT ?"
    )
    params: list = [args.n]
    if args.dataset:
        q = (
            "SELECT id, run_id, dataset_key, pass_rate, hybrid_enabled, git_commit "
            "FROM runs WHERE dataset_key = ? ORDER BY id DESC LIMIT ?"
        )
        params = [args.dataset, args.n]
    rows = conn.execute(q, params).fetchall()
    if not rows:
        print("(no rows)")
        conn.close()
        return 0
    cols = ["id", "run_id", "dataset_key", "pass_rate", "hybrid_enabled", "git_commit"]
    print("\t".join(cols))
    for row in rows:
        print("\t".join(str(row[c]) for c in cols))
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
