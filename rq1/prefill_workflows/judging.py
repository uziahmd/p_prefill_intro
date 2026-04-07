from __future__ import annotations

import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm

from .notebook_configs import JUDGE

JUDGED_RECORD_FIELDS = (
    "judge",
    "judge_model",
)


load_dotenv(override=True)


def build_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl_line(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def judge_one(client: OpenAI, sample: dict, prompt_template: str, max_retries: int = 6, base_sleep: float = 1.0) -> str:
    response_text = sample.get("followup_response", "")
    user_prompt = prompt_template.format(response=response_text)

    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=JUDGE["judge_model"],
                input=[
                    {"role": "system", "content": JUDGE["judge_system"]},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = (resp.output_text or "").strip()
            first = re.split(r"\s+", text.upper().strip(), maxsplit=1)[0]
            first = first.strip().strip(".,:;!?\"'`()[]{}<>")
            if first in {"YES", "NO"}:
                return first
            return "NO"
        except Exception:
            time.sleep(base_sleep * (2 ** attempt) + random.random() * 0.25)

    return "NO"


def judge_file(client: OpenAI, gen_path: Path, max_workers: int = 16):
    judged_path = JUDGE["run_dir"] / ("judged_" + gen_path.name)
    if judged_path.exists():
        judged_path.unlink()

    rows = list(read_jsonl(gen_path))
    results = [None] * len(rows)
    prompt_template = JUDGE["judge_prompt_template_f2"] if gen_path.name.endswith("_f2.jsonl") else JUDGE["judge_prompt_template"]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(judge_one, client, rows[index], prompt_template): index for index in range(len(rows))}
        progress = tqdm(as_completed(future_to_index), total=len(rows), desc=f"Judging ({gen_path.name})")
        for future in progress:
            index = future_to_index[future]
            judgement = future.result()
            results[index] = {**rows[index], "judge": judgement, "judge_model": JUDGE["judge_model"]}

    for row in results:
        write_jsonl_line(judged_path, row)

    return judged_path


def load_all_judged(judged_paths):
    all_rows = []
    for path in judged_paths:
        for row in read_jsonl(path):
            all_rows.append(row)
    return pd.DataFrame(all_rows)


def build_input_paths(file_names=None):
    seen = set()
    gen_files = []
    for name in (JUDGE["file_names"] if file_names is None else file_names):
        path = JUDGE["run_dir"] / name
        if path not in seen:
            gen_files.append(path)
            seen.add(path)
    return gen_files


def main(file_names=None) -> None:
    client = build_client()
    gen_files = build_input_paths(file_names)

    print("Target Files:", gen_files)
    missing = [path for path in gen_files if not path.exists()]
    present = [path for path in gen_files if path.exists()]

    print(f"\nPresent: {len(present)} / {len(gen_files)}")
    if missing:
        print("Missing files:")
        for path in missing:
            print(" -", path)

    judged_files = []
    for path in present:
        print(f"\n=== Judging {path.name} ===")
        judged_files.append(judge_file(client, path))

    print("\nJudged Files Completed:", judged_files)

    if judged_files:
        judged_df = load_all_judged(judged_files)
        if "model_name" not in judged_df.columns and "model" in judged_df.columns:
            judged_df["model_name"] = judged_df["model"]

        print("\nDataFrame shape:", judged_df.shape)
        print("Columns:", judged_df.columns[:20])

        judged_df["judge_answer"] = judged_df["judge"].astype(str).str.upper().str.strip()
        valid = judged_df[judged_df["judge_answer"].isin(["YES", "NO"])].copy()
        main_table = (
            valid.groupby(["model_name", "condition"])["judge_answer"]
            .value_counts(normalize=True)
            .rename("rate")
            .reset_index()
        )
        main_table = (
            main_table.pivot_table(
                index=["model_name", "condition"],
                columns="judge_answer",
                values="rate",
                fill_value=0.0,
            )
            .reset_index()
            .rename(columns={"YES": "yes_rate", "NO": "no_rate"})
        )
        print("\n=== FINAL AGGREGATED RESULTS ===")
        print(main_table)
    else:
        print("\nNo judged files were created. Exiting.")
