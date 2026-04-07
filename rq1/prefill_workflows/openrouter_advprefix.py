from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm

from .common import build_trial_rows, load_prompt_dataframe, set_seed, write_jsonl
from .notebook_configs import FOLLOWUP_IDS_TO_USE, FOLLOWUPS, HARM_BENCH_CSV, OPENROUTER_ADVPREFIX, RUN_DIR
from .openrouter_utils import build_client, generate_chat_adv

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


def build_output_paths(model_name: str, dataset_name: str):
    clean_model_name = model_name.replace("/", "__")
    return RUN_DIR / f"gen_adv_{clean_model_name}_{dataset_name}.jsonl", RUN_DIR / f"gen_adv_{clean_model_name}_{dataset_name}_f2.jsonl"


def run_openrouter_trial(client, model_name: str, user_text: str, prefill_prefix: str, followup_text: str, config: dict):
    prompt_to_send = user_text + " /no_think" if "qwen" in model_name.lower() else user_text

    init_messages = [{"role": "user", "content": prompt_to_send}]
    if prefill_prefix.strip():
        init_messages.append({"role": "assistant", "content": prefill_prefix})

    continuation, _ = generate_chat_adv(
        client,
        model_name,
        init_messages,
        2048,
        float(config["gen_kwargs"].get("temperature", 0.0)),
        float(config["gen_kwargs"].get("top_p", 1.0)),
        config["or_extra_body"],
        config["max_retries"],
    )
    initial_response = prefill_prefix + continuation

    followup_messages = [
        {"role": "user", "content": prompt_to_send},
        {"role": "assistant", "content": initial_response},
        {"role": "user", "content": followup_text},
    ]
    followup_response, _ = generate_chat_adv(
        client,
        model_name,
        followup_messages,
        512,
        float(config["gen_kwargs"].get("temperature", 0.0)),
        float(config["gen_kwargs"].get("top_p", 1.0)),
        config["or_extra_body"],
        config["max_retries"],
    )
    return initial_response, followup_response


def main(config: dict | None = None) -> None:
    config = dict(OPENROUTER_ADVPREFIX if config is None else config)

    openrouter_key = os.getenv("OPENROUTE_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    assert openrouter_key, "OPENROUTER_API_KEY not found. Put it in .env (same folder)."

    client = build_client(openrouter_key)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print("Run dir:", RUN_DIR)

    set_seed(config["seed"])

    def process_one_trial(args):
        trial, model_name, opt_prefixes = args
        user_text = trial["prompt_text"]
        prompt_id = trial["prompt_id"]
        prefill_text = opt_prefixes.get(prompt_id, [""])[0]

        records = []
        for followup_id in FOLLOWUP_IDS_TO_USE:
            followup_text = FOLLOWUPS[followup_id]
            try:
                init_resp, foll_resp = run_openrouter_trial(client, model_name, user_text, prefill_text, followup_text, config)
                records.append(
                    {
                        **trial,
                        "model_name": model_name,
                        "condition": "adv_prefill",
                        "prefill_id": "adv_prefill",
                        "prefill_text": prefill_text,
                        "followup_id": followup_id,
                        "followup_text": followup_text,
                        "initial_response": init_resp,
                        "followup_response": foll_resp,
                        "gen_kwargs": config["gen_kwargs"],
                    }
                )
            except Exception as err:
                records.append(
                    {
                        **trial,
                        "model_name": model_name,
                        "condition": "error",
                        "prefill_id": "error",
                        "prefill_text": prefill_text,
                        "followup_id": followup_id,
                        "followup_text": followup_text,
                        "initial_response": "",
                        "followup_response": "",
                        "error": repr(err),
                    }
                )
        return records

    for dataset_name in config["datasets_to_run"]:
        print(f"\n{'=' * 50}\nLoading dataset: {dataset_name}\n{'=' * 50}")
        df = load_prompt_dataframe(
            dataset_name,
            HARM_BENCH_CSV,
            preserve_socialharmbench_extra_columns=True,
            move_socialharmbench_fields_to_front=True,
            subset_harmbench_to_pipeline_fields=True,
            validate_harmbench_columns=True,
        )
        if config["shuffle"]:
            df = df.sample(frac=1.0, random_state=config["seed"]).reset_index(drop=True)
        if config["max_prompts"] is not None:
            df = df.head(config["max_prompts"]).copy()

        for model_name in config["target_models"]:
            clean_model_name = model_name.replace("/", "__")
            mapped_prefix_model = config["prefix_model_mapping"].get(model_name, clean_model_name)
            prefix_file = RUN_DIR / f"optimized_prefixes_{mapped_prefix_model}_{dataset_name}.json"

            if not prefix_file.exists():
                print(f"Skipping {model_name} on {dataset_name}: {prefix_file} not found.")
                continue

            with prefix_file.open("r", encoding="utf-8") as handle:
                opt_prefixes = json.load(handle)

            print(f"\nProcessing Target Model: {model_name} | Sourced Prefixes: {mapped_prefix_model} | Dataset: {dataset_name}")
            out_f1_path, out_f2_path = build_output_paths(model_name, dataset_name)
            if out_f1_path.exists():
                out_f1_path.unlink()
            if out_f2_path.exists():
                out_f2_path.unlink()

            trials = build_trial_rows(df)
            args_iter = ((trial, model_name, opt_prefixes) for trial in trials)

            with ThreadPoolExecutor(max_workers=config["api_concurrency"]) as executor:
                futures = [executor.submit(process_one_trial, args) for args in args_iter]
                for future in tqdm(as_completed(futures), total=len(trials), desc="Generating"):
                    for record in future.result():
                        if record["followup_id"] == "F1_minimal":
                            write_jsonl(out_f1_path, record, safe_json=True, allow_nan=False)
                        elif record["followup_id"] == "F_tamper_check":
                            write_jsonl(out_f2_path, record, safe_json=True, allow_nan=False)
