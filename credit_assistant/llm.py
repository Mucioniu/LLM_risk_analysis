from __future__ import annotations

import os
import re
from typing import Any

import httpx


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def optional_llm_summary(
    system_prompt: str,
    user_prompt: str,
    *,
    response_format_json: bool = False,
    max_tokens_override: int | None = None,
) -> str | None:
    """Call an OpenAI-compatible chat endpoint when explicitly configured.

    The project remains functional without an API key; this is only the generative
    layer above the deterministic evaluator and retrieved sources.
    """
    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    if not api_key:
        return None

    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("OPENAI_MODEL", "mistral-small3.2")
    think_enabled = env_flag("OLLAMA_THINK", False)
    default_timeout = "900" if think_enabled else "180"
    timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", default_timeout))
    max_tokens = max_tokens_override or int(os.getenv("OPENAI_MAX_TOKENS", "3000"))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if think_enabled:
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", base_url).rstrip("/")
        if ollama_base_url.endswith("/v1"):
            ollama_base_url = ollama_base_url[:-3]
        default_num_predict = str(max(max_tokens, 3000)) if model.startswith("gemma4") else str(max_tokens)
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "81409692" if model.startswith("gemma4") else "0"))
        options: dict[str, Any] = {
            "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", default_num_predict)),
        }
        if num_ctx > 0:
            options["num_ctx"] = num_ctx
        native_payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": True,
            "options": options,
        }
        if response_format_json:
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
                done_reason = data.get("done_reason", "necunoscut")
                if response_format_json:
                    return (
                        "LLM indisponibil sau configurat incorect: Gemma thinking nu a emis JSON final "
                        f"(done_reason={done_reason}). Creste OLLAMA_NUM_PREDICT/OLLAMA_NUM_CTX "
                        "sau seteaza OLLAMA_THINK=false."
                    )
                content = str(message.get("thinking") or message.get("reasoning") or "").strip()
            return clean_llm_markdown(content)
        except Exception as exc:
            return f"LLM indisponibil sau configurat incorect: {exc}"

    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if response_format_json:
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
        return clean_llm_markdown(data["choices"][0]["message"]["content"].strip())
    except Exception as exc:
        return f"LLM indisponibil sau configurat incorect: {exc}"
