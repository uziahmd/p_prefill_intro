from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm

from .common import build_trial_rows, load_prompt_dataframe, set_seed, write_jsonl
from .notebook_configs import FOLLOWUP_IDS_TO_USE, FOLLOWUPS, HARM_BENCH_CSV, RUN_DIR, SGLANG_ADVPREFIX
from .sglang_utils import (
    build_base_prompt,
    build_followup_prompt,
    create_client,
    fetch_server_model_id,
    generate_with_max_new_tokens,
    load_tokenizer,
    unload_model,
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


def build_output_paths(model_name: str, dataset_name: str):
    clean_model_name = model_name.replace("/", "__")
    return RUN_DIR / f"gen_adv_{clean_model_name}_{dataset_name}.jsonl", RUN_DIR / f"gen_adv_{clean_model_name}_{dataset_name}_f2.jsonl"


def run_prefill_trial(client, model_id: str, tokenizer, user_text: str, prefill_prefix: str, followup_text: str, gen_kwargs: dict):
    base = build_base_prompt(tokenizer, user_text)
    attacked = base + prefill_prefix

    init_kwargs = dict(gen_kwargs)
    init_kwargs["max_new_tokens"] = 2048
    continuation = generate_with_max_new_tokens(client, model_id, attacked, init_kwargs)
    initial_response = prefill_prefix + continuation

    followup_prompt = build_followup_prompt(tokenizer, user_text, initial_response, followup_text)
    followup_kwargs = dict(gen_kwargs)
    followup_kwargs["max_new_tokens"] = 512
    followup_response = generate_with_max_new_tokens(client, model_id, followup_prompt, followup_kwargs)

    return initial_response, followup_response


def main(config: dict | None = None) -> None:
    config = dict(SGLANG_ADVPREFIX if config is None else config)

    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not found. Put it in .env (same folder)."

    client = create_client(config["sglang_base_url"], use_http_client=False)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print("Run dir:", RUN_DIR)

    set_seed(config["seed"], include_torch=True)
    target_models = config["model_sets"][config["active_set"]]

    def process_one_trial(args):
        trial, model_name, model_id, tokenizer, opt_prefixes = args
        user_text = trial["prompt_text"]
        prompt_id = trial["prompt_id"]
        prefill_text = opt_prefixes.get(prompt_id, [""])[0]

        records = []
        for followup_id in FOLLOWUP_IDS_TO_USE:
            followup_text = FOLLOWUPS[followup_id]
            try:
                init_resp, foll_resp = run_prefill_trial(client, model_id, tokenizer, user_text, prefill_text, followup_text, config["gen_kwargs"])
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

        for model_name in target_models:
            clean_model_name = model_name.replace("/", "__")
            prefix_file = RUN_DIR / f"optimized_prefixes_{clean_model_name}_{dataset_name}.json"
            if not prefix_file.exists():
                print(f"Skipping {model_name} on {dataset_name}: {prefix_file} not found.")
                continue

            with prefix_file.open("r", encoding="utf-8") as handle:
                opt_prefixes = json.load(handle)

            print(f"\nProcessing Model: {model_name} | Dataset: {dataset_name}")
            tokenizer = load_tokenizer(model_name, trust_remote_code=False)
            model_id = fetch_server_model_id(config["sglang_base_url"])
            out_f1_path, out_f2_path = build_output_paths(model_name, dataset_name)
            if out_f1_path.exists():
                out_f1_path.unlink()
            if out_f2_path.exists():
                out_f2_path.unlink()

            trials = build_trial_rows(df)
            args_iter = ((trial, model_name, model_id, tokenizer, opt_prefixes) for trial in trials)
            with ThreadPoolExecutor(max_workers=config["concurrency"]) as executor:
                for rec_list in tqdm(executor.map(process_one_trial, args_iter), total=len(trials), desc="Generating"):
                    for record in rec_list:
                        if record["followup_id"] == "F1_minimal":
                            write_jsonl(out_f1_path, record, safe_json=True, allow_nan=False)
                        elif record["followup_id"] == "F_tamper_check":
                            write_jsonl(out_f2_path, record, safe_json=True, allow_nan=False)

            unload_model()
