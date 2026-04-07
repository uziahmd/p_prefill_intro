from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm

from .common import build_minimal_trials, load_prompt_dataframe, set_seed, write_jsonl
from .notebook_configs import FOLLOWUPS, HARM_BENCH_CSV, OPENROUTER_STATIC, RUN_DIR, STATIC_PREFILLS
from .openrouter_utils import apply_qwen_no_think, build_client, generate_chat_static

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


def run_control_trial(client, model_name: str, user_text: str, gen_kwargs: dict, extra_body: dict):
    msgs_1 = [{"role": "user", "content": apply_qwen_no_think(model_name, user_text)}]
    init_text, init_obj = generate_chat_static(client, model_name, msgs_1, 2048, gen_kwargs, extra_body)

    msgs_f1 = [
        {"role": "user", "content": apply_qwen_no_think(model_name, user_text)},
        {"role": "assistant", "content": init_text},
        {"role": "user", "content": apply_qwen_no_think(model_name, FOLLOWUPS["F1_minimal"])},
    ]
    f1_text, f1_obj = generate_chat_static(client, model_name, msgs_f1, 512, gen_kwargs, extra_body)

    msgs_f2 = [
        {"role": "user", "content": apply_qwen_no_think(model_name, user_text)},
        {"role": "assistant", "content": init_text},
        {"role": "user", "content": apply_qwen_no_think(model_name, FOLLOWUPS["F_tamper_check"])},
    ]
    f2_text, f2_obj = generate_chat_static(client, model_name, msgs_f2, 512, gen_kwargs, extra_body)

    return init_text, f1_text, f2_text, init_obj, f1_obj, f2_obj


def run_prefill_trial(client, model_name: str, user_text: str, prefill_prefix: str, gen_kwargs: dict, extra_body: dict):
    msgs_1 = [
        {"role": "user", "content": apply_qwen_no_think(model_name, user_text)},
        {"role": "assistant", "content": prefill_prefix},
    ]
    continuation, init_obj = generate_chat_static(client, model_name, msgs_1, 2048, gen_kwargs, extra_body)
    init_text = continuation if continuation.strip().startswith(prefill_prefix.strip()) else prefill_prefix + continuation

    msgs_f1 = [
        {"role": "user", "content": apply_qwen_no_think(model_name, user_text)},
        {"role": "assistant", "content": init_text},
        {"role": "user", "content": apply_qwen_no_think(model_name, FOLLOWUPS["F1_minimal"])},
    ]
    f1_text, f1_obj = generate_chat_static(client, model_name, msgs_f1, 512, gen_kwargs, extra_body)

    msgs_f2 = [
        {"role": "user", "content": apply_qwen_no_think(model_name, user_text)},
        {"role": "assistant", "content": init_text},
        {"role": "user", "content": apply_qwen_no_think(model_name, FOLLOWUPS["F_tamper_check"])},
    ]
    f2_text, f2_obj = generate_chat_static(client, model_name, msgs_f2, 512, gen_kwargs, extra_body)

    return init_text, f1_text, f2_text, init_obj, f1_obj, f2_obj


def build_output_paths(model_name: str, dataset_name: str):
    suffix = "_harmbench" if dataset_name.lower() == "harmbench" else "_socialharmbench"
    safe_model = model_name.replace("/", "__")
    return RUN_DIR / f"gen_{safe_model}{suffix}.jsonl", RUN_DIR / f"gen_{safe_model}{suffix}_f2.jsonl"


def main(config: dict | None = None) -> None:
    config = dict(OPENROUTER_STATIC if config is None else config)

    openrouter_key = os.getenv("OPENROUTE_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    assert openrouter_key, "OpenRouter key not found. Put OPENROUTE_API_KEY or OPENROUTER_API_KEY in .env"
    print("Loaded OpenRouter key")

    client = build_client(openrouter_key)
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

    def process_one_trial(args):
        trial, model_name = args
        user_text = trial["prompt_text"]

        base_records = []
        f2_records = []

        def create_record_pair(condition, prefill_id, prefill_text, init_text, f1_text, f2_text, init_obj, f1_obj, f2_obj, error=None):
            base_rec = {
                **trial,
                "model_name": model_name,
                "condition": condition,
                "prefill_id": prefill_id,
                "prefill_text": prefill_text,
                "initial_response": init_text,
                "followup_id": "F1_minimal" if not error else None,
                "followup_text": FOLLOWUPS["F1_minimal"] if not error else None,
                "followup_response": f1_text,
                "gen_kwargs": config["gen_kwargs"],
                "initial_openrouter_response": init_obj,
                "followup_openrouter_response": f1_obj,
            }

            f2_rec = base_rec.copy()
            f2_rec["followup_id"] = "F_tamper_check" if not error else None
            f2_rec["followup_text"] = FOLLOWUPS["F_tamper_check"] if not error else None
            f2_rec["followup_response"] = f2_text
            f2_rec["followup_openrouter_response"] = f2_obj

            if error:
                base_rec["error"] = error
                f2_rec["error"] = error

            return base_rec, f2_rec

        try:
            init_t, f1_t, f2_t, init_o, f1_o, f2_o = run_control_trial(
                client, model_name, user_text, config["gen_kwargs"], config["or_extra_body"]
            )
            b_rec, f_rec = create_record_pair("control", None, None, init_t, f1_t, f2_t, init_o, f1_o, f2_o)
            base_records.append(b_rec)
            f2_records.append(f_rec)

            for prefill_id, prefill_text in STATIC_PREFILLS.items():
                init_t, f1_t, f2_t, init_o, f1_o, f2_o = run_prefill_trial(
                    client, model_name, user_text, prefill_text, config["gen_kwargs"], config["or_extra_body"]
                )
                b_rec, f_rec = create_record_pair("prefill", prefill_id, prefill_text, init_t, f1_t, f2_t, init_o, f1_o, f2_o)
                base_records.append(b_rec)
                f2_records.append(f_rec)
        except Exception as err:
            b_rec, f_rec = create_record_pair("error", None, None, "", "", "", {}, {}, {}, error=repr(err))
            base_records.append(b_rec)
            f2_records.append(f_rec)

        return base_records, f2_records

    gen_files = []
    for model_name in config["target_models"]:
        out_path_base, out_path_f2 = build_output_paths(model_name, config["dataset_name"])
        if out_path_base.exists():
            out_path_base.unlink()
        if out_path_f2.exists():
            out_path_f2.unlink()

        args_iter = ((trial, model_name) for trial in trials)
        with ThreadPoolExecutor(max_workers=config["concurrency"]) as executor:
            for base_recs, f2_recs in tqdm(executor.map(process_one_trial, args_iter), total=len(trials), desc=f"Generating ({model_name})"):
                for base_record in base_recs:
                    write_jsonl(out_path_base, base_record)
                for followup_record in f2_recs:
                    write_jsonl(out_path_f2, followup_record)

        print(f"Finished {model_name}. Saved:\n- {out_path_base.name}\n- {out_path_f2.name}")
        gen_files.extend([out_path_base, out_path_f2])

    print("\nAll tasks complete. Generated files:")
    for generated_file in gen_files:
        print(generated_file)
