# Judge comparison: GPT-4o vs GPT-4o-mini (hybrid vs dense A/B)

This note compares **the same hybrid-vs-dense synthetic eval design** under two judge backends: **`openai:/gpt-4o`** vs **`openai:/gpt-4o-mini-2024-07-18`**. It complements the full A/B write-up in [`ab-study-20260325T154618Z-hybrid-vs-dense.md`](ab-study-20260325T154618Z-hybrid-vs-dense.md), which diagnosed **GPT-4o-mini as the dominant variance source** (especially faithfulness and contextual precision flip rates across repeated blocks).

## Data sources

| Field | GPT-4o-mini study | GPT-4o study |
|--------|-------------------|--------------|
| Tracking URI | `sqlite:///data/mlflow.db` (from `configs/eval.yaml`) | same |
| Experiment | `rag-eval-system` | same |
| `study_id` | `20260325T154618Z` | `20260325T180234Z` |
| `eval.kind` | `ab_study` | `ab_study` |
| Dataset | `synthetic` (25 items) | `eval.dataset_name=synthetic`, path `data/eval_datasets/synthetic.json` |
| Judge | Documented as `openai:/gpt-4o-mini-2024-07-18` in the prior report; those runs **pre-date** `judge_model` logging and have **no** `judge_model` param row in MLflow | `params.judge_model = openai:/gpt-4o` |
| Blocks completed | **10** pairs (`block` 0–9) | **5** finished pairs (`block` 0–4); `20260325T180234Z-a-5` was **RUNNING** when this report was written (no `b-5` yet) |

**MLflow filters**

- Mini (reference): `tags.eval.kind = 'ab_study' AND tags.study_id = '20260325T154618Z'`
- GPT-4o: `tags.eval.kind = 'ab_study' AND tags.study_id = '20260325T180234Z'` (and in the UI you can cross-check `params.judge_model = openai:/gpt-4o`)

**Important caveat:** the GPT-4o replication is **half the finished paired block count** of the mini study. Headline comparisons below emphasize **blocks 0–4** (direct overlap) and **block 0** (where the prior doc isolated a judge failure). Do not treat five GPT-4o blocks as a full replacement for ten mini blocks without more runs.

## Headline outcome

**Yes — judge behavior is materially better behaved with GPT-4o than in the documented GPT-4o-mini run, on the evidence we have.**

1. **The block-0 “hybrid catastrophe” under mini does not reproduce under GPT-4o.** With mini, hybrid `pass_rate` was **0.20** vs dense **0.68** on block 0, driven by a **faithfulness mean of 0.60** on hybrid. With GPT-4o on the same block index, hybrid **0.68** vs dense **0.76**, and **faithfulness/mean is 0.96 for both arms** — aligned with the prior report’s conclusion that block 0 was largely **judge inconsistency**, not a real retrieval regression.

2. **On overlapping blocks 0–4, the sign of the mean paired `pass_rate` delta (hybrid − dense) flips** from **−0.04** (mini) to **+0.048** (GPT-4o). That does **not** prove hybrid is better overall — sample size is small — but it shows the **mini judge was strong enough to invert apparent arm ordering** on this slice via the block-0 outlier.

3. **Run-level metric levels under GPT-4o are higher and less pathological on faithfulness/relevancy** than typical mini runs in the prior doc: e.g. GPT-4o dense **faithfulness/mean** averages **0.944** over five blocks vs **0.836** over ten blocks for mini (different n; still directionally consistent with a stricter-but-less-random judge).

## Side-by-side: block 0 (the prior “smoking gun”)

| Metric | Mini (154618Z) dense | Mini hybrid | GPT-4o (180234Z) dense | GPT-4o hybrid |
|--------|---------------------:|------------:|-------------------------:|--------------:|
| `pass_rate` | 0.68 | **0.20** | 0.76 | 0.68 |
| `faithfulness/mean` | 0.96 | **0.60** | 0.96 | 0.96 |
| `answer_relevancy/mean` | 0.96 | 0.88 | 0.96 | 0.96 |
| `contextual_precision/mean` | 0.76 | 0.60 | 0.80 | 0.76 |
| `contextual_recall/mean` | 0.96 | 1.00 | 1.00 | 1.00 |

Under GPT-4o, hybrid block 0 is **not** an extreme low-`pass_rate` outlier relative to dense; the mini run’s **0.20** hybrid `pass_rate` is the discrepancy.

## Overlap blocks 0–4: paired deltas (hybrid − dense)

Mean of per-block deltas:

| Metric | Mini (154618Z), blocks 0–4 | GPT-4o (180234Z), blocks 0–4 |
|--------|---------------------------:|-------------------------------:|
| `pass_rate` | −0.040 | **+0.048** |
| `faithfulness/mean` | −0.048 | −0.040 |
| `answer_relevancy/mean` | 0.000 | +0.008 |
| `contextual_precision/mean` | +0.048 | **+0.120** |
| `contextual_recall/mean` | +0.040 | −0.008 |

Per-block `pass_rate` values for transparency:

| block | Mini Δ `pass_rate` | GPT-4o Δ `pass_rate` |
|------:|-------------------:|-----------------------:|
| 0 | **−0.48** | −0.08 |
| 1 | +0.08 | +0.16 |
| 2 | +0.20 | +0.16 |
| 3 | +0.04 | −0.20 |
| 4 | −0.04 | +0.20 |

Almost all of the mini **−0.48** at block 0 is absent under GPT-4o.

## GPT-4o study: per-run table (all logged runs)

Arm **a** = dense (hybrid off), **b** = hybrid on. Values from `latest_metrics`.

| block | run (arm) | `pass_rate` | `faithfulness/mean` | `answer_relevancy/mean` | `contextual_precision/mean` | `contextual_recall/mean` |
|------:|-----------|------------:|--------------------:|------------------------:|----------------------------:|-------------------------:|
| 0 | `20260325T180234Z-a-0` | 0.76 | 0.96 | 0.96 | 0.80 | 1.00 |
| 1 | `20260325T180234Z-a-1` | 0.48 | 1.00 | 0.96 | 0.48 | 0.88 |
| 2 | `20260325T180234Z-a-2` | 0.44 | 0.88 | 1.00 | 0.68 | 0.88 |
| 3 | `20260325T180234Z-a-3` | 0.76 | 0.96 | 0.96 | 0.92 | 0.92 |
| 4 | `20260325T180234Z-a-4` | 0.52 | 0.92 | 0.92 | 0.60 | 0.92 |
| 0 | `20260325T180234Z-b-0` | 0.68 | 0.96 | 0.96 | 0.76 | 1.00 |
| 1 | `20260325T180234Z-b-1` | 0.64 | 0.92 | 0.92 | 0.80 | 0.92 |
| 2 | `20260325T180234Z-b-2` | 0.60 | 0.80 | 0.96 | 0.80 | 0.92 |
| 3 | `20260325T180234Z-b-3` | 0.56 | 0.92 | 1.00 | 0.80 | 0.84 |
| 4 | `20260325T180234Z-b-4` | 0.72 | 0.92 | 1.00 | 0.92 | 0.88 |

**Unpaired means ± population std dev across these five blocks**

| Arm | `pass_rate` | `faithfulness/mean` | `answer_relevancy/mean` | `contextual_precision/mean` | `contextual_recall/mean` |
|-----|------------:|--------------------:|------------------------:|----------------------------:|-------------------------:|
| dense (a) | 0.592 ± 0.139 | 0.944 ± 0.041 | 0.960 ± 0.025 | 0.696 ± 0.153 | 0.920 ± 0.044 |
| hybrid (b) | 0.640 ± 0.057 | 0.904 ± 0.054 | 0.968 ± 0.030 | 0.816 ± 0.054 | 0.912 ± 0.053 |

Hybrid **`pass_rate` across blocks is less dispersed** than in mini blocks 0–4 (mini hybrid std ~0.178 vs GPT-4o ~0.057), largely because **mini block 0 hybrid `pass_rate` = 0.20** inflates cross-block spread. That pattern is consistent with **lower judge-induced swing** on the composite pass gate, but a dedicated **per-sample flip-rate** analysis like the one in the mini report would require **more GPT-4o replicates** (or trace-level exports), not just five blocks.

## Interpretation (eval triage)

Using the **rag-eval-runs** pattern: when **the same arm and block index** imply the **same or near-identical RAG outputs**, large metric swings point to the **Judge / format** bucket, not retrieval.

- The prior doc showed **contradictory rationales** for the same hybrid answer across blocks (e.g. `syn003`) under GPT-4o-mini — classic judge variance.
- GPT-4o on block 0 **does not** reproduce the mini hybrid **faithfulness** collapse, which is exactly what you would expect if the dominant failure mode was **unreliable entailment judging** rather than hybrid retrieval.

**What GPT-4o does *not* eliminate:** generation noise (LM Studio), thresholded yes/no metrics, and small-$n$ block counts. Contextual precision remains volatile arm-to-arm (e.g. dense block 1 `contextual_precision/mean` = 0.48) — some of that may be real retrieval variation, some judge sensitivity.

## Recommendations (judge-specific)

1. **Keep logging `params.judge_model`** on every eval so future SQL/UI filters do not rely on study IDs alone.
2. **Re-run the full 10-block paired design with GPT-4o** (or another fixed strong judge) to match the statistical footprint of `20260325T154618Z`, then recompute flip rates if you still need per-sample stability numbers.
3. Treat **`pass_rate` as a judge-gated composite** — when comparing judges, report **per-metric means** alongside `pass_rate`, as in the mini report.

## Related

- Full mini study + flip-rate methodology: [`ab-study-20260325T154618Z-hybrid-vs-dense.md`](ab-study-20260325T154618Z-hybrid-vs-dense.md)
- MLflow query recipes: `.cursor/skills/mlflow/SKILL.md`
- Metric meanings / triage: `.cursor/skills/rag-eval-runs/SKILL.md`
