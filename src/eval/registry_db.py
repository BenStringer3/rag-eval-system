"""SQLite schema and default path for the eval run registry (queryable index over meta.json + report.json)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Default registry path (gitignored); single file, stdlib sqlite3 only.
DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data" / "eval_registry.db"

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  dataset_key TEXT NOT NULL,
  dataset_path TEXT NOT NULL,
  pass_rate REAL NOT NULL,
  mean_scores_json TEXT NOT NULL,
  core_threshold REAL NOT NULL,
  retrieval_threshold REAL NOT NULL,
  eval_context_json TEXT NOT NULL,
  git_commit TEXT,
  hybrid_enabled INTEGER NOT NULL,
  UNIQUE(run_id, dataset_key)
);

CREATE INDEX IF NOT EXISTS idx_runs_run_id ON runs(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_hybrid ON runs(hybrid_enabled);

CREATE TABLE IF NOT EXISTS sample_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_row_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  sample_id TEXT NOT NULL,
  passed_all INTEGER NOT NULL,
  metrics_json TEXT NOT NULL,
  UNIQUE(run_row_id, sample_id)
);

CREATE INDEX IF NOT EXISTS idx_sample_results_run ON sample_results(run_row_id);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
