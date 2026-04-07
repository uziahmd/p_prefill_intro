from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm

from .common import build_minimal_trials, load_prompt_dataframe, set_seed, write_jsonl
from .notebook_configs import FOLLOWUPS, HARM_BENCH_CSV, RUN_DIR, SGLANG_STATIC, STATIC_PREFILLS
from .sglang_utils import (
    build_base_prompt,
    build_followup_prompt,
    create_client,
    fetch_server_model_id,
    generate_from_text,
    load_tokenizer,
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


def build_output_paths(model_name: str, dataset_name: str):
    suffix = "_harmbench" if dataset_name.lower() == "harmbench" else "_socialharmbench"
    safe_model = model_name.replace("/", "__")
    return RUN_DIR / f"gen_{safe_model}{suffix}.jsonl", RUN_DIR / f"gen_{safe_model}{suffix}_f2.jsonl"


def run_control_trial(client, model_id, tokenizer, user_text: str, config: dict):
    base_prompt = build_base_prompt(tokenizer, user_text)
    initial_response = generate_from_text(client, model_id, base_prompt, config["gen_kwargs_initial"])

    f1_prompt = build_followup_prompt(tokenizer, user_text, initial_response, FOLLOWUPS["F1_minimal"])
    f1_response = generate_from_text(client, model_id, f1_prompt, config["gen_kwargs_followup"])

    f2_prompt = build_followup_prompt(tokenizer, user_text, initial_response, FOLLOWUPS["F_tamper_check"])
    f2_response = generate_from_text(client, model_id, f2_prompt, config["gen_kwargs_followup"])

    return initial_response, f1_response, f2_response


def run_prefill_trial(client, model_id, tokenizer, user_text: str, prefill_prefix: str, config: dict):
    base_prompt = build_base_prompt(tokenizer, user_text)
    attacked_prompt = base_prompt + prefill_prefix
    continuation = generate_from_text(client, model_id, attacked_prompt, config["gen_kwargs_initial"])
    initial_response = prefill_prefix + continuation

    f1_prompt = build_followup_prompt(tokenizer, user_text, initial_response, FOLLOWUPS["F1_minimal"])
    f1_response = generate_from_text(client, model_id, f1_prompt, config["gen_kwargs_followup"])

    f2_prompt = build_followup_prompt(tokenizer, user_text, initial_response, FOLLOWUPS["F_tamper_check"])
    f2_response = generate_from_text(client, model_id, f2_prompt, config["gen_kwargs_followup"])

    return initial_response, f1_response, f2_response


def main(config: dict | None = None) -> None:
    config = dict(SGLANG_STATIC if config is None else config)

    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not found. Put it in .env"

    client = create_client(config["sglang_base_url"], use_http_client=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(config["seed"], include_torch=True)
    df = load_prompt_dataframe(
        config["dataset_name"],
        HARM_BENCH_CSV,
        preserve_socialharmbench_extra_columns=True,
        move_socialharmbench_fields_to_front=False,
        subset_harmbench_to_pipeline_fields=False,
        validate_harmbench_columns=False,
    )
    if config["shuffle"]:
        df = df.sample(frac=1.0, random_state=config["seed"]).reset_index(drop=True)
    if config["max_prompts"] is not None:
        df = df.head(config["max_prompts"]).copy()

    print(f"Dataset: {config['dataset_name']} | Loaded rows: {len(df)}")

    trials = build_minimal_trials(df)
    target_models = config["model_sets"][config["active_set"]]

    def process_one_trial(args):
        trial, model_name, model_id, tokenizer = args
        user_text = trial["prompt_text"]

        base_records = []
        f2_records = []

        def create_record_pair(condition, prefill_id, prefill_text, init_resp, f1_resp, f2_resp, error=None):
            base_rec = {
                **trial,
                "model_name": model_name,
                "condition": condition,
                "prefill_id": prefill_id,
                "prefill_text": prefill_text,
                "initial_response": init_resp,
                "followup_id": "F1_minimal" if not error else None,
                "followup_text": FOLLOWUPS["F1_minimal"] if not error else None,
                "followup_response": f1_resp,
                "gen_kwargs_initial": config["gen_kwargs_initial"],
                "gen_kwargs_followup": config["gen_kwargs_followup"],
            }

            f2_rec = base_rec.copy()
            f2_rec["followup_id"] = "F_tamper_check" if not error else None
            f2_rec["followup_text"] = FOLLOWUPS["F_tamper_check"] if not error else None
            f2_rec["followup_response"] = f2_resp

            if error:
                base_rec["error"] = error
                f2_rec["error"] = error

            return base_rec, f2_rec

        try:
            init_resp, f1_resp, f2_resp = run_control_trial(client, model_id, tokenizer, user_text, config)
            b_rec, f_rec = create_record_pair("control", None, None, init_resp, f1_resp, f2_resp)
            base_records.append(b_rec)
            f2_records.append(f_rec)

            for prefill_id, prefill_text in STATIC_PREFILLS.items():
                init_resp, f1_resp, f2_resp = run_prefill_trial(client, model_id, tokenizer, user_text, prefill_text, config)
                b_rec, f_rec = create_record_pair("prefill", prefill_id, prefill_text, init_resp, f1_resp, f2_resp)
                base_records.append(b_rec)
                f2_records.append(f_rec)
        except Exception as err:
            b_rec, f_rec = create_record_pair("error", None, None, "", "", "", error=repr(err))
            base_records.append(b_rec)
            f2_records.append(f_rec)

        return base_records, f2_records

    gen_files = []
    for model_name in target_models:
        tokenizer = load_tokenizer(model_name, trust_remote_code=True)
        model_id = fetch_server_model_id(config["sglang_base_url"])
        out_path_base, out_path_f2 = build_output_paths(model_name, config["dataset_name"])
        if out_path_base.exists():
            out_path_base.unlink()
        if out_path_f2.exists():
            out_path_f2.unlink()

        args_iter = ((trial, model_name, model_id, tokenizer) for trial in trials)
        with ThreadPoolExecutor(max_workers=config["concurrency"]) as executor:
            for base_recs, f2_recs in tqdm(executor.map(process_one_trial, args_iter), total=len(trials), desc=f"Generating ({model_name})"):
                for base_record in base_recs:
                    write_jsonl(out_path_base, base_record)
                for followup_record in f2_recs:
                    write_jsonl(out_path_f2, followup_record)

        print(f"Finished {model_name}. Files saved:\n- {out_path_base.name}\n- {out_path_f2.name}")
        gen_files.extend([out_path_base, out_path_f2])

    print("\nAll tasks complete. Generated files:")
    for generated_file in gen_files:
        print(generated_file)
