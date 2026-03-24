# Run eval report (`scripts/run_eval_report.py`)

**Follow the skill:** read and apply `.claude/skills/run-eval-report/SKILL.md` (same content as `.cursor/skills/run-eval-report/SKILL.md` via symlink). That file is the source of truth for flags, runtime expectations, and reporting.

## What to do

1. Work from the repo root. Run Python as **`.venv/bin/python`** (do not rely on `source .venv/bin/activate` in automation).
2. **Dataset choice** (user preference or ask once):
   - **Fast-er / smaller:** `data/eval_datasets/starter.json`
   - **Larger / slower:** `data/eval_datasets/synthetic.json`
   - **All enabled in config:** omit `--dataset` (see `configs/eval.yaml`).
3. **Expect long wall clock:** many minutes for starter; much longer for synthetic. Use generous timeouts or run in background and monitor output; stalled-looking progress is often normal.
4. After completion, report the **`artifacts/eval_runs/<UTC>/`** path, the dataset subfolder (`single/`, `starter/`, or `synthetic/`), and that `report.md`, `report.json`, `report.csv`, and `meta.json` exist.

## Minimal invocation examples

```bash
.venv/bin/python scripts/run_eval_report.py --dataset data/eval_datasets/starter.json
```

```bash
.venv/bin/python scripts/run_eval_report.py --dataset data/eval_datasets/synthetic.json
```

If the cloud judge rate-limits, use `--max-concurrent 1` and `--judge-throttle-seconds 5` (see the skill).
