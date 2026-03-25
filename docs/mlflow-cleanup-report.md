# Replacing vibe-coded ML infrastructure with battle-tested OSS

**MLflow is the gravitational center of the OSS ML experiment ecosystem, and as of early 2026, its native integration with DeepEval and RAGAS means you can build a statistically rigorous RAG A/B testing stack that is 100% Apache 2.0, runs entirely on local SQLite, and requires zero cloud services.** The critical gap across every tool in this space is the same: none of them ship built-in statistical significance testing for comparing experiment run groups. That layer — paired t-tests, bootstrap confidence intervals, power analysis — must be assembled from scipy, pingouin, and mlxtend. This report maps the entire landscape, identifies what's genuinely OSS versus marketing, and provides concrete integration patterns for a Python-based RAG eval system already using DeepEval and SQLite.

---

## MLflow is the only experiment tracker that matters for OSS

MLflow (Apache 2.0, **~20k GitHub stars**, 30M+ monthly PyPI downloads, 900+ contributors) is the unambiguous industry standard. Its data model is straightforward: **Experiments** are named containers, **Runs** are individual executions within them, and each run stores parameters (string key-value pairs), metrics (numeric values with optional step history), artifacts (files), and tags (arbitrary metadata). The Python API is minimal — `mlflow.start_run()`, `log_param()`, `log_metric()`, `log_artifact()` — and `mlflow.autolog()` provides zero-code tracking for sklearn, PyTorch, XGBoost, and others.

Run grouping is handled through nested runs (`mlflow.start_run(nested=True)` for parent-child relationships) and arbitrary tags for filtering. The `search_runs()` API returns pandas DataFrames and supports SQL-like filter strings (`params.model = 'gpt-4o' AND metrics.faithfulness > 0.8`). The UI provides sortable run tables, metric overlay charts, parallel coordinates plots, and side-by-side run comparison — functional if not beautiful.

The transformative recent development is **MLflow 3.8+**, which added native `get_judge()` wrappers for both DeepEval and RAGAS metrics:

```python
from mlflow.metrics.genai import get_judge
faithfulness = get_judge("deepeval", "faithfulness")
ragas_precision = get_judge("ragas", "context_precision")
results = mlflow.evaluate(model=your_rag_fn, data=eval_dataset, 
                          scorers=[faithfulness, ragas_precision])
```

This single API unifies 20+ DeepEval metrics and RAGAS metrics as MLflow scorers, with results automatically tracked in MLflow's backend. MLflow 3.7+ also added trace comparison for GenAI applications and multi-turn evaluation support.

**What MLflow does NOT do**: statistical comparison of run groups. There are no built-in t-tests, bootstrap CIs, or ANOVA. You must export results via `search_runs()` and handle statistics yourself. This is the single biggest gap in the ecosystem, shared by every tracking tool.

The other contenders break down cleanly. **DVC** (Apache 2.0, ~14k stars) is excellent for data versioning and pipeline DAGs but its experiment tracking feels secondary — it's Git-based with no rich self-hosted web UI (DVC Studio is SaaS). **Evidently AI** (Apache 2.0, ~5.5k stars) provides 20+ statistical tests but for *data drift detection*, not experiment comparison — it's complementary to MLflow, not a replacement. **Optuna** (MIT, ~11k stars) is purpose-built for hyperparameter optimization with a trial/study model and excellent dashboard, but it's not a general experiment tracker. **ClearML** has impressive auto-logging but its server uses SSPL (not true OSS). **Weights & Biases** has the best UI in the space but the server is proprietary — the client SDK being Apache 2.0 is misleading. **Neptune.ai** is fully proprietary SaaS (being acquired by OpenAI). **Aim** (Apache 2.0, ~5.9k stars) offers the best self-hosted UI for metric exploration but has an uncertain future after its AimOS pivot.

---

## Statistical rigor requires assembling four libraries yourself

Professional ML A/B testing differs fundamentally from web experimentation: both models are evaluated on the **same test examples**, making **paired tests** the standard. This eliminates inter-example variance and dramatically reduces required sample sizes. The canonical reference remains Dietterich (1998), which demonstrated that two commonly used tests — difference-of-proportions and naive paired t-tests on random splits — have unacceptably high Type I error rates.

The decision tree for choosing a statistical test is straightforward:

- **Binary outcomes** (correct/incorrect per example): **McNemar's test** via `mlxtend.evaluate.mcnemar()`. This tests whether two models' disagreements are asymmetric on a 2×2 contingency table.
- **Continuous scores** (faithfulness, relevancy, etc.) with approximately normal differences: **Paired t-test** via `scipy.stats.ttest_rel()`.
- **Non-normal continuous scores**: **Wilcoxon signed-rank test** via `scipy.stats.wilcoxon()`.
- **Algorithm comparison via cross-validation**: **5×2cv combined F-test** via `mlxtend.evaluate.combined_ftest_5x2cv()` — more robust than the t-test variant.
- **Three or more models**: **Cochran's Q** (omnibus) → post-hoc pairwise McNemar with **Holm-Bonferroni correction** via `statsmodels.stats.multitest.multipletests(p_values, method='holm')`.

**Always report bootstrap confidence intervals** alongside p-values. The BCa (bias-corrected and accelerated) method is recommended:

```python
from scipy.stats import bootstrap
res = bootstrap((scores_a, scores_b), statistic=lambda a, b, axis=0: 
    np.mean(a - b, axis=axis), n_resamples=10000, paired=True, 
    confidence_level=0.95, method='BCa')
```

For **power analysis**, use `statsmodels.stats.power.TTestPower` or pingouin's `pg.power_ttest()`. Typical effect sizes in ML: small improvements (prompt tuning) yield Cohen's d ≈ 0.1–0.3 requiring **500–1000+ eval samples**; meaningful model upgrades yield d ≈ 0.3–0.5 requiring **100–400 samples**; generation jumps yield d ≈ 0.5+ requiring **30–100 samples**.

**Pingouin** (v0.6.0, actively maintained) deserves special attention — a single `pg.ttest(scores_a, scores_b, paired=True)` call returns the t-statistic, p-value, Cohen's d, 95% CI, statistical power, *and* Bayes Factor simultaneously. No other library matches this density of output.

For **Bayesian model comparison**, the `baycomp` library implements the Benavoli et al. (2017) framework with ROPE (Region of Practical Equivalence) support, returning `P(A better)`, `P(equivalent)`, `P(B better)` — far more interpretable than p-values. It's functional but classified as inactive (last meaningful update Feb 2025).

**Handling LLM judge non-determinism** is a solved problem in practice: run each evaluation **3–5 times** and average scores (for continuous metrics) or use majority voting (for binary judgments). Even at `temperature=0`, GPU floating-point non-associativity and dynamic batching produce variations — a 2024 study measured accuracy variations up to **15%** across identical runs. For pairwise comparisons, always run twice with positions swapped to mitigate position bias (GPT-4 shows ~40% inconsistency when pairs are reversed). Report inter-run variance alongside aggregated scores.

**Sequential testing** (SPRT) is worth considering when LLM API calls are expensive — it can reduce required samples by **50%+** when effects are clear by stopping early. For fixed offline benchmarks, classical fixed-N testing is simpler and sufficient.

---

## Overnight experiment running: from simple loops to the Karpathy pattern

The overnight experiment runner landscape spans from dead-simple Python loops to AI agents that design their own experiments. For RAG pipeline evaluation, start simple and add sophistication only when needed.

**Tier 1 (start here)**: A Python script with `concurrent.futures.ProcessPoolExecutor`, a list of config dicts, results logged to SQLite, and a Slack webhook notification on completion. This is ~100–200 lines and handles 80% of use cases. Add retry with exponential backoff for API rate limits:

```python
def run_with_retry(config, max_retries=3):
    for attempt in range(max_retries):
        try:
            return run_experiment(config)
        except RateLimitError:
            sleep(60 * (2 ** attempt) + random.uniform(0, 10))
    return {"status": "failed", "config": config}
```

**Tier 2 (smart selection)**: **Optuna** (MIT, v4.7, Jan 2026) transforms a sequential experiment sweep into intelligent Bayesian search. Its define-by-run API wraps arbitrary evaluation logic, studies persist to SQLite for pause/resume, and the `optuna-dashboard` command launches a real-time web UI showing optimization history and parameter importance. For config management, **Hydra** (MIT, 10.3k stars) provides hierarchical YAML composition with parameter sweep support via `--multirun`. Hydra's Optuna sweeper plugin combines both tools. Note that Hydra's development has slowed and some practitioners now prefer plain YAML + Pydantic for simplicity.

**Tier 3 (AI-driven)**: The **autoresearch** pattern, released by Karpathy in March 2026 (MIT, **42k stars in two weeks**), represents a paradigm shift. In 630 lines of Python, an AI agent (Claude Code or similar) reads a `program.md` instruction file, modifies experiment code, runs training/eval for a fixed time budget, and either git-commits (if metrics improve) or git-resets (if not). Karpathy's first run produced 89 experiments overnight with 15 keepers and an 11% efficiency gain. The pattern works because git serves as persistent memory and the fixed-budget constraint prevents runaway experiments. **Risks**: Goodhart's Law (agents aggressively optimize metrics that don't perfectly measure what you care about), diminishing returns after ~100 experiments, and context pollution from repeated log output.

**Meta's Ax** (MIT, v1.0 released Nov 2025) occupies the high-end niche — its Gaussian process-based Bayesian optimization maximizes sample efficiency when each experiment is expensive, with multi-objective optimization and outcome constraints. It's more sophisticated than Optuna but also more complex. **Ray Tune** (Apache 2.0) is overkill for single-machine RAG evaluation — its value lies in multi-GPU parallelism and distributed scheduling.

For notifications, **ntfy.sh** (free, OSS push notifications) or a simple Slack webhook POST provides morning-ready experiment reports with zero infrastructure.

---

## RAG evaluation tools: four frameworks, one clear integration path

**RAGAS** (Apache 2.0, **~12.9k stars**, monthly releases) is the de facto standard for RAG-specific metrics — faithfulness, answer relevancy, context precision, context recall — all reference-free using LLM-as-judge. It's a metrics library, not an experiment platform, which is exactly right: metrics should be decoupled from tracking.

**DeepEval** (Apache 2.0, ~7k stars, 400k+ monthly downloads) is the most developer-friendly framework, treating LLM evaluation as Pytest-native unit testing. It offers **50+ metrics** spanning RAG, agents, conversations, and safety, plus an `ArenaGEval` metric that directly compares test cases A/B-style. The important distinction: DeepEval the OSS library runs entirely locally; **Confident AI** is the commercial SaaS dashboard by the same team. For team collaboration features (shareable reports, annotation queues), you need Confident AI. For everything else, DeepEval + MLflow replaces it.

**Arize Phoenix** (~8.5k stars) provides the best visualization in the space — particularly UMAP-based embedding clustering for diagnosing retrieval quality — but uses **Elastic License 2.0**, which is source-available, not OSI-approved open source. It supports experiment comparison in its UI and integrates bidirectionally with RAGAS and DeepEval. Worth using for debugging but not as a foundation.

**TruLens** (MIT, ~2.3k stars, now under Snowflake) offers a genuine all-in-one: evaluation, tracing, and a built-in Streamlit dashboard with version leaderboards. Its "RAG Triad" (context relevance, groundedness, answer relevance) was an influential contribution. The Snowflake backing ensures longevity but may push toward Snowflake-specific features. No A/B statistical testing built in.

**None of these frameworks include built-in statistical A/B comparison.** This is the universal gap. Every tool stops at showing you metric values and leaves significance testing as an exercise for the user.

---

## The concrete integration architecture for your existing system

Given an existing Python-based RAG eval system using DeepEval with SQLite storage, the migration path is:

**Step 1: Add MLflow with SQLite backend** — one line of config:
```python
mlflow.set_tracking_uri("sqlite:///mlflow_evals.db")
```
MLflow creates its own schema in SQLite via SQLAlchemy. Your existing custom SQLite stays as a read-only archive.

**Step 2: Use MLflow 3.8+ native judge wrappers** — replace direct DeepEval calls with `mlflow.evaluate()` using `get_judge("deepeval", metric_name)`. This gives you automatic experiment tracking with zero additional logging code. You can mix DeepEval and RAGAS metrics in a single evaluation call.

**Step 3: Add a statistical comparison module** — a ~50-line Python module that pulls run groups via `mlflow.search_runs()`, computes paired t-tests or bootstrap CIs using scipy/pingouin, and returns structured comparison results. This is the layer no OSS tool provides.

**Step 4: Dashboard** — `mlflow server` gives you immediate run comparison. For custom statistical views, a **Streamlit app** querying MLflow's API adds ~100 lines. For terminal monitoring during overnight runs, Rich progress bars work well.

**Step 5: Overnight automation** — wrap your eval in Optuna's `study.optimize()` with SQLite storage for intelligent experiment selection, or use the simple job queue pattern with retry logic. Add a Slack webhook for morning notifications.

The resulting stack is: **DeepEval + RAGAS → MLflow 3.8+ (SQLite) → scipy/pingouin → MLflow UI + optional Streamlit**. Every component is Apache 2.0 or MIT, self-hosted, and used by professional ML engineers in production.

---

## Conclusion

The OSS ML experimentation ecosystem has a clear architecture: **MLflow for tracking, DeepEval + RAGAS for metrics, scipy/pingouin/mlxtend for statistics, and simple Python patterns for orchestration.** The most important insight is that the "statistical rigor" layer is conspicuously absent from every commercial and open-source experiment platform — this ~50-line scipy module is the single highest-leverage piece of custom code to write. Everything else has a battle-tested OSS solution. The Karpathy autoresearch pattern represents the emerging frontier where AI agents autonomously run and iterate on experiments overnight, but it requires clear, ungameable metrics to be safe. For RAG evaluation specifically, the MLflow 3.8+ native integration with DeepEval and RAGAS judges eliminates what was previously the most tedious glue code, making the migration from custom infrastructure a matter of days rather than weeks.