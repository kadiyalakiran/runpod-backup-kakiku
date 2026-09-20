"""
Unified LLM client for the cs329a stubs (and anything else in this repo).

The whole point: every stub in cs329a/ takes a plain `llm_call(prompt) -> str`
function. This module gives you ONE implementation of that function, and which
backend it actually talks to is controlled entirely by environment variables —
so switching between "RunPod is up, use the real Qwen2.5-32B" and "RunPod's
out of capacity, use a small local model instead" is an .env edit, never a
code change.

Both Ollama and a RunPod-hosted vLLM server speak the OpenAI-compatible
chat-completions API, which is why one client covers both.

Env vars (see .env.example):
    LLM_BASE_URL   default: http://localhost:11434/v1   (Ollama's local endpoint)
    LLM_MODEL      default: qwen2.5:3b-instruct
    LLM_API_KEY    default: "ollama"  (Ollama ignores it; RunPod vLLM may check it)
"""

from __future__ import annotations

import os
from typing import Callable

from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:3b-instruct"


def get_llm_call(
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.7,
) -> Callable[[str], str]:
    """Returns a plain `llm_call(prompt) -> str`, matching the signature every
    cs329a stub expects (SampleFn / LLMCallFn / GenerateFn — they're all the
    same shape). Reads from the environment if args aren't passed explicitly,
    so the common case is just `get_llm_call()` with everything set in .env.
    """
    resolved_base_url = base_url or os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    resolved_model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    resolved_api_key = api_key or os.environ.get("LLM_API_KEY", "ollama")

    client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)

    def llm_call(prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    return llm_call


def describe_active_backend() -> str:
    """Prints which backend you're actually pointed at right now — run this
    before a debugging session so you're not confused about whether you're
    hitting the tiny local model or your real RunPod-hosted Qwen32B."""
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    kind = "LOCAL (Ollama)" if "localhost" in base_url or "127.0.0.1" in base_url else "REMOTE"
    return f"[{kind}] base_url={base_url} model={model}"


if __name__ == "__main__":
    print(describe_active_backend())
    print()
    print("Sending a test prompt...")
    try:
        call = get_llm_call()
        print(call("Say hello in exactly five words."))
    except Exception as e:
        print(f"Call failed ({type(e).__name__}): {e}")
