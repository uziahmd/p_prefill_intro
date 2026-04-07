from __future__ import annotations

import ast
import json
from pathlib import Path

from .notebook_configs import (
    ADVPREFIX_GENERATOR,
    FOLLOWUPS,
    HARM_BENCH_CSV,
    JUDGE,
    OPENROUTER_ADVPREFIX,
    OPENROUTER_STATIC,
    SGLANG_ADVPREFIX,
    SGLANG_STATIC,
    STATIC_PREFILLS,
)

ROOT = Path(__file__).resolve().parents[2]

OPENROUTER_STATIC_RECORD_FIELDS = (
    "prompt_id",
    "category",
    "sub_topic",
    "type",
    "prompt_text",
    "model_name",
    "condition",
    "prefill_id",
    "prefill_text",
    "initial_response",
    "followup_id",
    "followup_text",
    "followup_response",
    "gen_kwargs",
    "initial_openrouter_response",
    "followup_openrouter_response",
)

SGLANG_STATIC_RECORD_FIELDS = (
    "prompt_id",
    "category",
    "sub_topic",
    "type",
    "prompt_text",
    "model_name",
    "condition",
    "prefill_id",
    "prefill_text",
    "initial_response",
    "followup_id",
    "followup_text",
    "followup_response",
    "gen_kwargs_initial",
    "gen_kwargs_followup",
)

ADVPREFIX_RECORD_FIELDS = (
    "prompt_id",
    "category",
    "sub_topic",
    "type",
    "prompt_text",
    "model_name",
    "condition",
    "prefill_id",
    "prefill_text",
    "followup_id",
    "followup_text",
    "initial_response",
    "followup_response",
    "gen_kwargs",
)

JUDGED_RECORD_FIELDS = ("judge", "judge_model")


def _load_notebook_cell(rel_path: str, cell_index: int) -> str:
    notebook = json.loads((ROOT / rel_path).read_text(encoding="utf-8"))
    return "".join(notebook["cells"][cell_index].get("source", []))


def _assignments(source: str) -> dict[str, ast.AST]:
    tree = ast.parse(source)
    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
    return assignments


def _literal(node: ast.AST):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path":
        return ast.literal_eval(node.args[0])
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        return {
            keyword.arg: _literal(keyword.value)
            for keyword in node.keywords
            if keyword.arg is not None
        }
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "getenv"
        and len(node.args) >= 2
    ):
        return _literal(node.args[1])
    if isinstance(node, ast.Subscript):
        return None
    return ast.literal_eval(node)


def _compare(label: str, actual, expected) -> dict:
    return {
        "label": label,
        "status": "match" if actual == expected else "mismatch",
        "actual": actual,
        "expected": expected,
    }


def _static_output_paths(model_name: str, dataset_name: str) -> tuple[str, str]:
    suffix = "_harmbench" if dataset_name.lower() == "harmbench" else "_socialharmbench"
    safe_model = model_name.replace("/", "__")
    return f"gen_{safe_model}{suffix}.jsonl", f"gen_{safe_model}{suffix}_f2.jsonl"


def _adv_output_paths(model_name: str, dataset_name: str) -> tuple[str, str]:
    safe_model = model_name.replace("/", "__")
    return f"gen_adv_{safe_model}_{dataset_name}.jsonl", f"gen_adv_{safe_model}_{dataset_name}_f2.jsonl"


def audit_notebook_parity() -> list[dict]:
    checks: list[dict] = []

    openrouter_static_cells = _assignments(_load_notebook_cell("rq1/(openrouter)rq1_prefill.ipynb", 0))
    checks.extend(
        [
            _compare("openrouter_static.target_models", tuple(_literal(openrouter_static_cells["TARGET_MODELS"])), OPENROUTER_STATIC["target_models"]),
            _compare("openrouter_static.prefills", _literal(openrouter_static_cells["PREFILLS"]), STATIC_PREFILLS),
            _compare("openrouter_static.followups", {
                _literal(openrouter_static_cells["F1_ID"]): _literal(openrouter_static_cells["F1_TEXT"]),
                _literal(openrouter_static_cells["F2_ID"]): _literal(openrouter_static_cells["F2_TEXT"]),
            }, FOLLOWUPS),
            _compare("openrouter_static.gen_kwargs", _literal(openrouter_static_cells["GEN_KWARGS"]), OPENROUTER_STATIC["gen_kwargs"]),
            _compare("openrouter_static.or_extra_body", _literal(openrouter_static_cells["OR_EXTRA_BODY"]), OPENROUTER_STATIC["or_extra_body"]),
            _compare("openrouter_static.dataset_name", _literal(openrouter_static_cells["DATASET_NAME"]), OPENROUTER_STATIC["dataset_name"]),
            _compare("openrouter_static.concurrency", _literal(openrouter_static_cells["CONCURRENCY"]), OPENROUTER_STATIC["concurrency"]),
            _compare("openrouter_static.harm_bench_csv", _literal(openrouter_static_cells["HARM_BENCH_CSV"]), HARM_BENCH_CSV.as_posix()),
        ]
    )

    openrouter_adv_cells = _assignments(_load_notebook_cell("rq1/(openrouter)rq1_prefill.ipynb", 2))
    checks.extend(
        [
            _compare("openrouter_adv.target_models", tuple(_literal(openrouter_adv_cells["TARGET_MODELS"])), OPENROUTER_ADVPREFIX["target_models"]),
            _compare("openrouter_adv.prefix_model_mapping", _literal(openrouter_adv_cells["PREFIX_MODEL_MAPPING"]), OPENROUTER_ADVPREFIX["prefix_model_mapping"]),
            _compare("openrouter_adv.datasets_to_run", tuple(_literal(openrouter_adv_cells["DATASETS_TO_RUN"])), OPENROUTER_ADVPREFIX["datasets_to_run"]),
            _compare("openrouter_adv.followups", _literal(openrouter_adv_cells["FOLLOWUPS"]), FOLLOWUPS),
            _compare("openrouter_adv.followup_ids", tuple(_literal(openrouter_adv_cells["FOLLOWUP_IDS_TO_USE"])), tuple(FOLLOWUPS.keys())),
            _compare("openrouter_adv.api_concurrency", _literal(openrouter_adv_cells["API_CONCURRENCY"]), OPENROUTER_ADVPREFIX["api_concurrency"]),
            _compare("openrouter_adv.max_retries", _literal(openrouter_adv_cells["MAX_RETRIES"]), OPENROUTER_ADVPREFIX["max_retries"]),
            _compare("openrouter_adv.gen_kwargs", _literal(openrouter_adv_cells["GEN_KWARGS"]), OPENROUTER_ADVPREFIX["gen_kwargs"]),
        ]
    )

    sglang_static_cells = _assignments(_load_notebook_cell("rq1/(sglang)rq1_prefill.ipynb", 1))
    checks.extend(
        [
            _compare("sglang_static.model_sets", {key: tuple(value) for key, value in _literal(sglang_static_cells["MODEL_SETS"]).items()}, SGLANG_STATIC["model_sets"]),
            _compare("sglang_static.active_set", _literal(sglang_static_cells["ACTIVE_SET"]), SGLANG_STATIC["active_set"]),
            _compare("sglang_static.dataset_name", _literal(sglang_static_cells["DATASET_NAME"]), SGLANG_STATIC["dataset_name"]),
            _compare("sglang_static.gen_kwargs_initial", _literal(sglang_static_cells["GEN_KWARGS_INITIAL"]), SGLANG_STATIC["gen_kwargs_initial"]),
            _compare("sglang_static.gen_kwargs_followup", _literal(sglang_static_cells["GEN_KWARGS_FOLLOWUP"]), SGLANG_STATIC["gen_kwargs_followup"]),
            _compare("sglang_static.sglang_base_url", _literal(sglang_static_cells["SGLANG_BASE_URL"]), SGLANG_STATIC["sglang_base_url"]),
            _compare("sglang_static.concurrency", _literal(sglang_static_cells["CONCURRENCY"]), SGLANG_STATIC["concurrency"]),
        ]
    )

    sglang_adv_cells = _assignments(_load_notebook_cell("rq1/(sglang)rq1_prefill.ipynb", 3))
    checks.extend(
        [
            _compare("sglang_adv.model_sets", {key: tuple(value) for key, value in _literal(sglang_adv_cells["MODEL_SETS"]).items()}, SGLANG_ADVPREFIX["model_sets"]),
            _compare("sglang_adv.active_set", _literal(sglang_adv_cells["ACTIVE_SET"]), SGLANG_ADVPREFIX["active_set"]),
            _compare("sglang_adv.datasets_to_run", tuple(_literal(sglang_adv_cells["DATASETS_TO_RUN"])), SGLANG_ADVPREFIX["datasets_to_run"]),
            _compare("sglang_adv.followups", _literal(sglang_adv_cells["FOLLOWUPS"]), FOLLOWUPS),
            _compare("sglang_adv.followup_ids", tuple(_literal(sglang_adv_cells["FOLLOWUP_IDS_TO_USE"])), tuple(FOLLOWUPS.keys())),
            _compare("sglang_adv.gen_kwargs", _literal(sglang_adv_cells["GEN_KWARGS"]), SGLANG_ADVPREFIX["gen_kwargs"]),
            _compare("sglang_adv.sglang_base_url", _literal(sglang_adv_cells["SGLANG_BASE_URL"]), SGLANG_ADVPREFIX["sglang_base_url"]),
            _compare("sglang_adv.concurrency", _literal(sglang_adv_cells["CONCURRENCY"]), SGLANG_ADVPREFIX["concurrency"]),
        ]
    )

    judge_prompt_cells = _assignments(_load_notebook_cell("rq1/rq1_judge.ipynb", 2))
    judge_file_cells = _assignments(_load_notebook_cell("rq1/rq1_judge.ipynb", 5))
    checks.extend(
        [
            _compare("judge.model", _literal(judge_prompt_cells["JUDGE_MODEL"]), JUDGE["judge_model"]),
            _compare("judge.system", _literal(judge_prompt_cells["JUDGE_SYSTEM"]), JUDGE["judge_system"]),
            _compare("judge.prompt_template", _literal(judge_prompt_cells["JUDGE_PROMPT_TEMPLATE"]), JUDGE["judge_prompt_template"]),
            _compare("judge.prompt_template_f2", _literal(judge_prompt_cells["JUDGE_PROMPT_TEMPLATE_F2"]), JUDGE["judge_prompt_template_f2"]),
            _compare("judge.file_names", tuple(_literal(judge_file_cells["FILE_NAMES"])), JUDGE["file_names"]),
        ]
    )

    advprefix_cells = _assignments(_load_notebook_cell("rq1/rq1_advprefix.ipynb", 0))
    checks.extend(
        [
            _compare("advprefix.generator_model", _literal(advprefix_cells["GENERATOR_MODEL"]), ADVPREFIX_GENERATOR["generator_model"]),
            _compare("advprefix.target_models", tuple(_literal(advprefix_cells["TARGET_MODELS"])), ADVPREFIX_GENERATOR["target_models"]),
            _compare("advprefix.datasets", tuple(_literal(advprefix_cells["DATASETS"])), ADVPREFIX_GENERATOR["datasets"]),
            _compare("advprefix.judge_model", _literal(advprefix_cells["JUDGE_MODEL"]), ADVPREFIX_GENERATOR["judge_model"]),
            _compare("advprefix.surrogate_prompt", _literal(advprefix_cells["SURROGATE_PROMPT"]), ADVPREFIX_GENERATOR["surrogate_prompt"]),
            _compare("advprefix.pasr_samples", _literal(advprefix_cells["PASR_SAMPLES"]), ADVPREFIX_GENERATOR["pasr_samples"]),
            _compare("advprefix.pasr_temp", _literal(advprefix_cells["PASR_TEMP"]), ADVPREFIX_GENERATOR["pasr_temp"]),
            _compare("advprefix.weight_pasr", _literal(advprefix_cells["WEIGHT_PASR"]), ADVPREFIX_GENERATOR["weight_pasr"]),
            _compare("advprefix.top_k_prefixes", _literal(advprefix_cells["TOP_K_PREFIXES"]), ADVPREFIX_GENERATOR["top_k_prefixes"]),
            _compare("advprefix.max_ce_threshold", _literal(advprefix_cells["MAX_CE_THRESHOLD"]), ADVPREFIX_GENERATOR["max_ce_threshold"]),
            _compare("advprefix.max_workers", _literal(advprefix_cells["MAX_WORKERS"]), ADVPREFIX_GENERATOR["max_workers"]),
            _compare("advprefix.min_token_length", _literal(advprefix_cells["MIN_TOKEN_LENGTH"]), ADVPREFIX_GENERATOR["min_token_length"]),
            _compare("advprefix.start_patterns", tuple(_literal(advprefix_cells["START_PATTERNS"])), ADVPREFIX_GENERATOR["start_patterns"]),
            _compare("advprefix.contain_patterns", tuple(_literal(advprefix_cells["CONTAIN_PATTERNS"])), ADVPREFIX_GENERATOR["contain_patterns"]),
        ]
    )

    checks.extend(
        [
            _compare(
                "output.openrouter_static",
                _static_output_paths("google/gemma-2-27b-it", "harmbench"),
                ("gen_google__gemma-2-27b-it_harmbench.jsonl", "gen_google__gemma-2-27b-it_harmbench_f2.jsonl"),
            ),
            _compare(
                "output.sglang_static",
                _static_output_paths("google/gemma-2-9b-it", "socialharmbench"),
                ("gen_google__gemma-2-9b-it_socialharmbench.jsonl", "gen_google__gemma-2-9b-it_socialharmbench_f2.jsonl"),
            ),
            _compare(
                "output.openrouter_adv",
                _adv_output_paths("google/gemma-2-27b-it", "harmbench"),
                ("gen_adv_google__gemma-2-27b-it_harmbench.jsonl", "gen_adv_google__gemma-2-27b-it_harmbench_f2.jsonl"),
            ),
            _compare(
                "output.sglang_adv",
                _adv_output_paths("meta-llama/Llama-3.1-8B-Instruct", "socialharmbench"),
                ("gen_adv_meta-llama__Llama-3.1-8B-Instruct_socialharmbench.jsonl", "gen_adv_meta-llama__Llama-3.1-8B-Instruct_socialharmbench_f2.jsonl"),
            ),
            _compare("schema.openrouter_static", OPENROUTER_STATIC_RECORD_FIELDS, (
                "prompt_id", "category", "sub_topic", "type", "prompt_text", "model_name", "condition", "prefill_id", "prefill_text", "initial_response", "followup_id", "followup_text", "followup_response", "gen_kwargs", "initial_openrouter_response", "followup_openrouter_response",
            )),
            _compare("schema.sglang_static", SGLANG_STATIC_RECORD_FIELDS, (
                "prompt_id", "category", "sub_topic", "type", "prompt_text", "model_name", "condition", "prefill_id", "prefill_text", "initial_response", "followup_id", "followup_text", "followup_response", "gen_kwargs_initial", "gen_kwargs_followup",
            )),
            _compare("schema.openrouter_adv", ADVPREFIX_RECORD_FIELDS, (
                "prompt_id", "category", "sub_topic", "type", "prompt_text", "model_name", "condition", "prefill_id", "prefill_text", "followup_id", "followup_text", "initial_response", "followup_response", "gen_kwargs",
            )),
            _compare("schema.sglang_adv", ADVPREFIX_RECORD_FIELDS, (
                "prompt_id", "category", "sub_topic", "type", "prompt_text", "model_name", "condition", "prefill_id", "prefill_text", "followup_id", "followup_text", "initial_response", "followup_response", "gen_kwargs",
            )),
            _compare("schema.judged_fields", JUDGED_RECORD_FIELDS, ("judge", "judge_model")),
        ]
    )

    return checks


def render_markdown_report(checks: list[dict]) -> str:
    lines = ["# Notebook Parity Audit", "", "## Summary", ""]
    mismatches = [check for check in checks if check["status"] != "match"]
    lines.append(f"- Total checks: {len(checks)}")
    lines.append(f"- Matches: {len(checks) - len(mismatches)}")
    lines.append(f"- Mismatches: {len(mismatches)}")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    for check in checks:
        status = "PASS" if check["status"] == "match" else "FAIL"
        lines.append(f"- {status}: `{check['label']}`")
        if check["status"] != "match":
            lines.append(f"  expected: `{check['expected']}`")
            lines.append(f"  actual: `{check['actual']}`")
    return "\n".join(lines) + "\n"
