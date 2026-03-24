---
name: Cloud Judge Provider Switch
overview: Add a provider-selectable DeepEval judge configuration so evaluations can run against OpenAI (`gpt-4o-mini`) now and be switched back to local LM Studio later without code edits.
todos:
  - id: design-provider-config
    content: Add provider-aware judge schema to eval.yaml with comments and default to OpenAI gpt-4o-mini.
    status: completed
  - id: implement-judge-routing
    content: Refactor src/eval/judge.py to build GPTModel from selected provider and OPENCODE_ZEN_API_KEY.
    status: completed
  - id: align-cli-and-docs
    content: Update run_eval_report help text and README instructions for cloud/local judge switching.
    status: completed
  - id: smoke-validate
    content: Validate Zen cloud judge connectivity first, then run a small eval and lint checks on modified files.
    status: completed
isProject: false
---

# Add Cloud Judge Provider Toggle

## Goal

Enable the judge model to run on either local LM Studio or OpenAI, selected via eval config, with API keys sourced from environment variables for safety.

## Proposed Changes

- Update `[/home/ben/repos/rag-eval-system/configs/eval.yaml](/home/ben/repos/rag-eval-system/configs/eval.yaml)`:
  - Add explicit judge provider settings (e.g. `provider: openai|local`).
  - Add provider-specific connection fields with concise comments (including how local re-enable works).
  - Set initial cloud target to `gpt-4o-mini`.
- Refactor `[/home/ben/repos/rag-eval-system/src/eval/judge.py](/home/ben/repos/rag-eval-system/src/eval/judge.py)`:
  - Replace LM-Studio-only wiring with provider-aware `GPTModel` construction.
  - Resolve API key from `OPENCODE_ZEN_API_KEY` for OpenAI provider.
  - Keep fail-fast validation (missing provider fields/env var should raise clear errors).
  - Preserve existing `generation_kwargs` and JSON-trimming patch behavior.
- Update CLI/help text in `[/home/ben/repos/rag-eval-system/scripts/run_eval_report.py](/home/ben/repos/rag-eval-system/scripts/run_eval_report.py)` where it currently implies LM Studio-only judge behavior.
- Update user docs in `[/home/ben/repos/rag-eval-system/README.md](/home/ben/repos/rag-eval-system/README.md)`:
  - Document how to run judge on OpenAI now.
  - Document one-line switch back to local.
  - Document required env var (`OPENCODE_ZEN_API_KEY`).

## Validation

- Run a direct cloud-judge connectivity check (single minimal OpenAI-compatible request) using `OPENCODE_ZEN_API_KEY` to confirm auth + endpoint reachability before running DeepEval.
- Run a focused eval smoke run with current dataset using updated config path.
- Verify judge requests succeed with OpenAI provider and produce `report.json`/`report.md`.
- Run lints on touched files and fix any introduced issues.

## Notes

- No fallback chain will be added; provider config is explicit and strict.
- Local judge path remains available by flipping provider/model fields in `eval.yaml`.

