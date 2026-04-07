from __future__ import annotations

import random
import time

import httpx
import requests
import torch
from openai import OpenAI
from transformers import AutoTokenizer


def create_http_client():
    return httpx.Client(
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
        timeout=httpx.Timeout(60.0),
    )


def create_client(base_url: str, *, use_http_client: bool) -> OpenAI:
    if use_http_client:
        return OpenAI(base_url=base_url, api_key="EMPTY", http_client=create_http_client())
    return OpenAI(base_url=base_url, api_key="EMPTY")


def load_tokenizer(model_name: str, *, trust_remote_code: bool) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)


def fetch_server_model_id(base_url: str) -> str:
    models = requests.get(base_url + "/models", timeout=5).json()
    return models["data"][0]["id"]


def unload_model() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_chat_template_no_think_if_supported(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def build_base_prompt(tokenizer, user_text: str) -> str:
    return apply_chat_template_no_think_if_supported(tokenizer, [{"role": "user", "content": user_text}])


def build_followup_prompt(tokenizer, user_text: str, assistant_text: str, followup_text: str) -> str:
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
        {"role": "user", "content": followup_text},
    ]
    return apply_chat_template_no_think_if_supported(tokenizer, messages)


def generate_from_text(client: OpenAI, model_id: str, prompt_text: str, kwargs: dict) -> str:
    for attempt in range(6):
        try:
            resp = client.completions.create(model=model_id, prompt=prompt_text, **kwargs)
            return resp.choices[0].text
        except Exception:
            time.sleep(1.0 * (2 ** attempt) + random.random() * 0.25)
    raise RuntimeError("SGLang generation failed after retries.")


def generate_with_max_new_tokens(client: OpenAI, model_id: str, prompt_text: str, gen_kwargs: dict) -> str:
    max_new = int(gen_kwargs.get("max_new_tokens", 512))
    temperature = float(gen_kwargs.get("temperature", 0.0))
    top_p = float(gen_kwargs.get("top_p", 1.0))

    for attempt in range(6):
        try:
            resp = client.completions.create(
                model=model_id,
                prompt=prompt_text,
                max_tokens=max_new,
                temperature=temperature,
                top_p=top_p,
            )
            return resp.choices[0].text
        except Exception:
            time.sleep(1.0 * (2 ** attempt) + random.random() * 0.25)
    raise RuntimeError("SGLang generation failed after retries.")
