# Refactor Notes

## Goal

Make the repository easier to understand and maintain without changing notebook-defined behavior.

## What Changed

- Extracted duplicated RQ1 workflow logic into `rq1/prefill_workflows/`
- Kept all existing RQ1 script entry points in place as thin wrappers
- Added a notebook-parity audit and a test that reads notebook sources directly
- Rewrote the README around the actual notebook-backed workflow

## Mismatches Found During Refactor

### 1. `rq1/judge.py` had drifted from `rq1_judge.ipynb`

Found:

- judge model had been changed away from the notebook-defined `openai/gpt-4.1`
- API call changed from `responses.create(...)` to `chat.completions.create(...)`
- inline comment claimed the notebook model did not exist

Resolution:

- restored the notebook-defined judge model
- restored notebook-style `responses.create(...)` judging behavior
- kept the same file list and prompt templates from the notebook

### 2. `rq1/(sglang)prefix_static.py` had notebook-default drift

Found:

- notebook cell default active set: `gemma`
- script default active set: `llama`
- notebook default dataset: `harmbench`
- script default dataset: `socialharmbench`

Resolution:

- aligned wrapper defaults back to the notebook cell used for the static SGLang workflow

### 3. `rq1/(sglang)advprefix.py` had notebook-default drift

Found:

- notebook cell used `http://127.0.0.1:30000/v1`
- script used `http://127.0.0.1:30001/v1`
- notebook cell model-set selections differed from the script defaults

Resolution:

- aligned the wrapper config back to the notebook cell that defines the SGLang adversarial-prefix workflow

## What Was Intentionally Kept Identical

- output file naming
- prompt text
- follow-up text
- generation parameter values
- judge prompts
- dataset names and loading behavior
- the hardcoded HarmBench CSV path
- SGLang/OpenRouter endpoint defaults taken from the notebooks
- `rq1_runs/` as the output directory

## Validation Added

- `tools/write_parity_report.py`
- `docs/notebook_parity_audit.md`
- `tests/test_notebook_parity.py`

These checks verify notebook parity for configuration, prompts, model names, output filenames, and record schemas.
