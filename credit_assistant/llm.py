from __future__ import annotations

import os
import re
from typing import Any

import httpx


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


def optional_llm_summary(system_prompt: str, user_prompt: str) -> str | None:
    """Call an OpenAI-compatible chat endpoint when explicitly configured.

    The project remains functional without an API key; this is only the generative
    layer above the deterministic evaluator and retrieved sources.
    """
    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    if not api_key:
        return None

    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("OPENAI_MODEL", "qwen3:8b")
    timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "180"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1800"))
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

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
