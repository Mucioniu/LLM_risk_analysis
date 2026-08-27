from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_reasoning_setting(name: str = "OLLAMA_THINK") -> bool | str | None:
    """Parse Ollama's optional thinking control without losing effort levels."""
    value = os.getenv(name)
    if value is None or not value.strip() or value.strip().lower() in {"auto", "default"}:
        return None

    normalized = value.strip().lower()
    if normalized in {"0", "false", "no", "off", "none"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"low", "medium", "high"}:
        return normalized
    raise ValueError(
        f"{name} must be one of auto, off, on, low, medium, or high; got {value!r}."
    )


def clean_llm_markdown(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\*{3,}([^*\n]+?)\*{3,}", r"\1", text)
    text = re.sub(r"\*\*([^*\n]+?)\*\*", r"\1", text)

    cleaned_lines: list[str] = []
    previous_blank = False
    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"[*_\-]{3,}", stripped):
            if not previous_blank:
                cleaned_lines.append("")
                previous_blank = True
            continue

        cleaned_lines.append(line.rstrip())
        previous_blank = stripped == ""

    return "\n".join(cleaned_lines).strip()


def is_ollama_endpoint(base_url: str) -> bool:
    try:
        parsed = urlparse(base_url)
        return parsed.port == 11434
    except ValueError:
        return False


def optional_llm_summary(
    system_prompt: str,
    user_prompt: str,
    *,
    response_format_json: bool = False,
    json_schema: dict[str, Any] | None = None,
    max_tokens_override: int | None = None,
    model_env_name: str | None = None,
    reasoning_env_name: str | None = None,
    temperature_env_name: str | None = None,
    num_ctx_env_name: str | None = None,
    num_predict_env_name: str | None = None,
) -> str | None:
    """Call an OpenAI-compatible chat endpoint when explicitly configured.

    The project remains functional without an API key; this is only the generative
    layer above the deterministic evaluator and retrieved sources.
    """
    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    if not api_key:
        return None

    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    model = (
        os.getenv(model_env_name, "") if model_env_name else ""
    ) or os.getenv("OPENAI_MODEL", "mistral-small3.2")
    if reasoning_env_name and os.getenv(reasoning_env_name) is not None:
        reasoning_setting = env_reasoning_setting(reasoning_env_name)
    else:
        reasoning_setting = env_reasoning_setting()
    reasoning_enabled = reasoning_setting not in {None, False}
    default_timeout = "900" if reasoning_enabled else "180"
    timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", default_timeout))
    max_tokens = max_tokens_override or int(os.getenv("OPENAI_MAX_TOKENS", "3000"))
    temperature_value = (
        os.getenv(temperature_env_name, "") if temperature_env_name else ""
    ) or os.getenv("OPENAI_TEMPERATURE", "0.1")
    temperature = float(temperature_value)
    seed_value = os.getenv("OPENAI_SEED")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    use_native_ollama = env_flag(
        "OLLAMA_NATIVE_CHAT",
        bool(os.getenv("OLLAMA_BASE_URL")) or is_ollama_endpoint(base_url),
    )
    if use_native_ollama:
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", base_url).rstrip("/")
        if ollama_base_url.endswith("/v1"):
            ollama_base_url = ollama_base_url[:-3]
        default_num_predict = str(max(max_tokens, 3000)) if model.startswith("gemma4") else str(max_tokens)
        num_ctx_value = (
            os.getenv(num_ctx_env_name, "") if num_ctx_env_name else ""
        ) or os.getenv("OLLAMA_NUM_CTX", "8192")
        num_predict_value = (
            os.getenv(num_predict_env_name, "") if num_predict_env_name else ""
        ) or os.getenv("OLLAMA_NUM_PREDICT", default_num_predict)
        num_ctx = int(num_ctx_value)
        options: dict[str, Any] = {
            "num_predict": int(num_predict_value),
            "temperature": temperature,
        }
        if seed_value is not None:
            options["seed"] = int(seed_value)
        if num_ctx > 0:
            options["num_ctx"] = num_ctx
        native_payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if reasoning_setting is not None:
            native_payload["think"] = reasoning_setting
        if json_schema is not None:
            native_payload["format"] = json_schema
        elif response_format_json:
            native_payload["format"] = "json"
        try:
            response = httpx.post(
                f"{ollama_base_url}/api/chat",
                json=native_payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            message = data.get("message", {})
            content = str(message.get("content") or "").strip()
            if not content:
                done_reason = data.get("done_reason", "unknown")
                if response_format_json:
                    return (
                        "The LLM is unavailable or incorrectly configured: Ollama did not produce final JSON "
                        f"(done_reason={done_reason}). Increase OLLAMA_NUM_PREDICT/OLLAMA_NUM_CTX "
                        "or set OLLAMA_THINK=false."
                    )
                content = str(message.get("thinking") or message.get("reasoning") or "").strip()
            return content if response_format_json else clean_llm_markdown(content)
        except Exception as exc:
            return f"The LLM is unavailable or incorrectly configured: {exc}"

    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if seed_value is not None:
        payload["seed"] = int(seed_value)
    if json_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_response",
                "strict": True,
                "schema": json_schema,
            },
        }
    elif response_format_json:
        payload["response_format"] = {"type": "json_object"}

    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        return content if response_format_json else clean_llm_markdown(content)
    except Exception as exc:
        return f"The LLM is unavailable or incorrectly configured: {exc}"
