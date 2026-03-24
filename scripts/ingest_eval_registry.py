#!/usr/bin/env python3
"""Ingest artifacts/eval_runs/<UTC>/ folders into SQLite (data/eval_registry.db by default).

Reads meta.json + each dataset's report.json; idempotent upsert on (run_id, dataset_key).

Example::

    .venv/bin/python scripts/ingest_eval_registry.py artifacts/eval_runs/20260324T195433Z
    .venv/bin/python scripts/ingest_eval_registry.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest eval run folders into the eval registry DB.")
    p.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        help="Paths to artifacts/eval_runs/<UTC>/ (each must contain meta.json).",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Ingest every subfolder of --base that contains meta.json.",
    )
    p.add_argument(
        "--base",
        type=Path,
        default=ROOT / "artifacts" / "eval_runs",
        help="Base directory for --all (default: artifacts/eval_runs).",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=ROOT / "data" / "eval_registry.db",
        help="SQLite database path (default: data/eval_registry.db).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    sys.path.insert(0, str(ROOT))

    from src.eval.registry_db import connect, init_schema
    from src.eval.registry_ingest import ingest_run_dir, iter_run_dirs

    if args.all:
        dirs = iter_run_dirs(args.base.resolve())
        if not dirs:
            print(f"No run directories with meta.json under {args.base}", file=sys.stderr)
            return 1
    else:
        if not args.run_dirs:
            print("Provide run_dirs or --all.", file=sys.stderr)
            return 1
        dirs = [d.resolve() for d in args.run_dirs]

    conn = connect(args.db)
    init_schema(conn)
    for d in dirs:
        ingest_run_dir(conn, d)
        print(f"Ingested {d}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
