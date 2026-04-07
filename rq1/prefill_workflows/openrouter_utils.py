from __future__ import annotations

import json
import random
import time

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError


def build_client(api_key: str) -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def response_to_jsonable(resp) -> dict:
    if resp is None:
        return {}
    if hasattr(resp, "model_dump"):
        try:
            return resp.model_dump(mode="json")
        except TypeError:
            return resp.model_dump()
    if hasattr(resp, "to_dict"):
        return resp.to_dict()
    try:
        return json.loads(str(resp))
    except Exception:
        return {"_repr": repr(resp)}


def extract_text_from_openrouter_obj(obj) -> str:
    if not obj or not isinstance(obj, dict):
        return ""
    try:
        choices = obj.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            message = first_choice.get("message") if isinstance(first_choice, dict) else None
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
            text = first_choice.get("text") if isinstance(first_choice, dict) else None
            if isinstance(text, str) and text.strip():
                return text
    except Exception:
        pass
    return ""


def apply_qwen_no_think(model_name: str, text: str) -> str:
    if "qwen" in model_name.lower() and not text.endswith(" /no_think"):
        return text + " /no_think"
    return text


def generate_chat_static(client: OpenAI, model_name: str, messages: list, max_tokens: int, gen_kwargs: dict, extra_body: dict):
    last_err = None
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=float(gen_kwargs.get("temperature", 0.0)),
                top_p=float(gen_kwargs.get("top_p", 1.0)),
                extra_body=extra_body,
            )
            text = resp.choices[0].message.content or ""
            return text, response_to_jsonable(resp)
        except (RateLimitError, APIConnectionError) as err:
            last_err = err
            time.sleep(1.0 * (2 ** attempt) + random.random() * 0.25)
        except APIStatusError as err:
            last_err = err
            status = getattr(err, "status_code", None)
            raise RuntimeError(f"OpenRouter APIStatusError status={status} msg={err}") from err
        except Exception as err:
            last_err = err
            time.sleep(1.0 * (2 ** attempt) + random.random() * 0.25)
    raise RuntimeError(f"OpenRouter generation failed. Last error: {repr(last_err)}")


def generate_chat_adv(client: OpenAI, model_name: str, messages: list, max_new_tokens: int, temperature: float, top_p: float, extra_body: dict, max_retries: int):
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_body=extra_body,
            )
            resp_dict = response_to_jsonable(resp)
            text = extract_text_from_openrouter_obj(resp_dict)
            return text, resp_dict
        except Exception as err:
            last_err = err
            time.sleep(1.0 * (2 ** attempt) + random.random() * 0.25)
    raise RuntimeError(f"OpenRouter generation failed after {max_retries} retries. err={repr(last_err)}")
