# Eval registry (SQLite)

The eval registry is a **queryable index** over completed runs under `artifacts/eval_runs/<UTC>/`. It does **not** replace `meta.json` or `report.json`; those remain the source of truth for full detail.

- **Database path:** `data/eval_registry.db` (gitignored). Create it by ingesting runs.
- **Schema:** [`src/eval/registry_db.py`](../src/eval/registry_db.py) — tables `runs` and `sample_results`.

## Ingest

After one or more eval batches exist:

```bash
.venv/bin/python scripts/ingest_eval_registry.py artifacts/eval_runs/20260324T195433Z
```

Backfill everything under `artifacts/eval_runs/` that has a `meta.json`:

```bash
.venv/bin/python scripts/ingest_eval_registry.py --all
```

Ingest is **idempotent** on `(run_id, dataset_key)`: re-ingesting updates aggregates and replaces per-sample rows for that run.

**Hybrid flag:** New runs written by `scripts/run_eval_report.py` include `retrieval_features.hybrid_enabled` in `meta.json`. For older runs without that field, ingest reads `retrieval.hybrid.enabled` from the `config` path stored in `meta.json` (fail-fast if the file is missing).

## Query

```bash
.venv/bin/python scripts/query_eval_registry.py last -n 10
.venv/bin/python scripts/query_eval_registry.py last --dataset single -n 20
.venv/bin/python scripts/query_eval_registry.py stats
.venv/bin/python scripts/query_eval_registry.py sql "SELECT run_id, pass_rate, hybrid_enabled FROM runs ORDER BY id DESC LIMIT 5"
```

## Compare groups (t-test)

Requires **scipy** (`pip install -e '.[dev]'`). Compares **run-level** means for a named metric (keys match `mean_scores` in reports, e.g. `Faithfulness`, `Answer Relevancy`).

```bash
.venv/bin/python scripts/compare_eval_configs.py \
  --dataset single \
  --metric Faithfulness \
  --group-a 20260324T175312Z,20260324T180009Z \
  --group-b 20260324T195433Z
```

Use `--paired` only when runs are intentionally paired in order (same length). Default is Welch’s t-test (unpaired, unequal variance).

## A/B study (orchestrated)

Agents: see [`.cursor/skills/science-loop/SKILL.md`](../.cursor/skills/science-loop/SKILL.md) (science loop).

Automated A/B runs write under `artifacts/ab_study/<study_id>/` (`study_meta.json`, `manifest.jsonl`, `eval_runs/<UTC>/`). Successful runs are ingested into the registry unless `--no-ingest`.

**Fixed-N (recommended for interpretable p-values):**

```bash
.venv/bin/python scripts/run_ab_study.py \
  --dataset data/eval_datasets/synthetic.json \
  --arm-a-label "dense (hybrid off)" --arm-b-label "hybrid on" \
  --arm-a-overrides-json '{"retrieval.hybrid.enabled": false}' \
  --arm-b-overrides-json '{"retrieval.hybrid.enabled": true}' \
  --pairing paired --n-per-arm 5 \
  --max-concurrent 1 --judge-throttle-seconds 5
```

**Exploratory sequential** (peek after each block; inflates Type I error—label analysis exploratory):

```bash
.venv/bin/python scripts/run_ab_study.py \
  --dataset data/eval_datasets/starter.json \
  --arm-a-label "dense (hybrid off)" --arm-b-label "hybrid on" \
  --arm-a-overrides-json '{"retrieval.hybrid.enabled": false}' \
  --arm-b-overrides-json '{"retrieval.hybrid.enabled": true}' \
  --pairing paired --exploratory \
  --min-per-arm 2 --max-per-arm 10 --alpha 0.05 \
  --primary-metric pass_rate
```

**Report** (Plotly HTML under `docs/_figures/ab-study-<study_id>/`, markdown in `docs/`):

```bash
.venv/bin/python scripts/render_ab_study_report.py \
  --study-dir artifacts/ab_study/<study_id>
```

Failed eval subprocesses are logged in `manifest.jsonl` with `status: failed`; optional `--retries` repeats an arm. Do not ingest incomplete run folders.

## Variance and multi-run workflow

Judge and generator noise still apply; the registry helps **filter and aggregate**, not remove variance. For medians without SQL, see [`scripts/summarize_eval_runs.py`](../scripts/summarize_eval_runs.py) and [determinism-2026-03-24.md](determinism-2026-03-24.md).

## Cursor skill

Agents: see [`.cursor/skills/eval-registry/SKILL.md`](../.cursor/skills/eval-registry/SKILL.md).
