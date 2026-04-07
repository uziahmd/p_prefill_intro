from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import pandas as pd


def set_seed(seed: int, include_torch: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_torch:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_prompt_dataframe(
    dataset_name: str,
    harm_bench_csv: Path,
    *,
    preserve_socialharmbench_extra_columns: bool,
    move_socialharmbench_fields_to_front: bool,
    subset_harmbench_to_pipeline_fields: bool,
    validate_harmbench_columns: bool,
) -> pd.DataFrame:
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "socialharmbench":
        from datasets import load_dataset

        ds = load_dataset("psyonp/SocialHarmBench", split="train")
        df = ds.to_pandas()
        for column in ["prompt_id", "category", "sub_topic", "type"]:
            if column not in df.columns:
                df[column] = None

        if not preserve_socialharmbench_extra_columns:
            df = df[["prompt_id", "category", "sub_topic", "type", "prompt_text"]]
        elif move_socialharmbench_fields_to_front:
            front = ["prompt_id", "category", "sub_topic", "type", "prompt_text"]
            rest = [column for column in df.columns if column not in front]
            df = df[front + rest]

        return df.dropna(subset=["prompt_text"]).reset_index(drop=True)

    if dataset_name == "harmbench":
        df = pd.read_csv(harm_bench_csv)
        df.columns = [column.strip().rstrip(".") for column in df.columns]

        if validate_harmbench_columns:
            required = [
                "Behavior",
                "FunctionalCategory",
                "SemanticCategory",
                "Tags",
                "ContextString",
                "BehaviorID",
            ]
            missing = [column for column in required if column not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in HarmBench CSV: {missing}. Found: {list(df.columns)}")

        df["prompt_text"] = df["Behavior"].astype(str).str.strip()
        df["prompt_id"] = df["BehaviorID"]
        df["category"] = df["SemanticCategory"]
        df["sub_topic"] = df["SemanticCategory"]
        df["type"] = df["FunctionalCategory"]

        if subset_harmbench_to_pipeline_fields:
            df = df[["prompt_id", "category", "sub_topic", "type", "prompt_text"]]

        return df.dropna(subset=["prompt_text"]).reset_index(drop=True)

    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def json_safe(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [json_safe(inner) for inner in value]
    return value


def write_jsonl(path: Path, record: dict, *, safe_json: bool = False, allow_nan: bool = True) -> None:
    payload = json_safe(record) if safe_json else record
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=allow_nan) + "\n")


def build_trial_rows(df: pd.DataFrame) -> list[dict]:
    return [row.to_dict() for _, row in df.iterrows()]


def build_minimal_trials(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "prompt_id": row.get("prompt_id", None),
            "category": row.get("category", None),
            "sub_topic": row.get("sub_topic", None),
            "type": row.get("type", None),
            "prompt_text": row["prompt_text"],
        }
        for _, row in df.iterrows()
    ]
