"""Ollama provider — local, open-source LLMs via the Ollama daemon.

Default model is ``llama3.1:8b`` (Meta Llama 3.1 8B Instruct). It is a strong
instruction-following open-source model well suited to grounded Q&A over
retrieved USPS context (Pub 28, CASS Technical Guide, AMS). Swap via the
``OLLAMA_MODEL`` env var to ``qwen2.5:7b-instruct`` or ``mistral:7b-instruct``
for comparable size, or ``llama3.2:3b`` for lower-memory Macs.

Install:  https://ollama.com/download
Pull:     ``ollama pull llama3.1:8b``
Daemon:   Ollama listens on ``http://localhost:11434`` by default.
"""
from __future__ import annotations

import json
from typing import Iterable, List

import httpx

from backend.llm.base import LLMMessage, LLMProvider


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str):
        if not base_url:
            raise ValueError("OLLAMA_BASE_URL is required for the ollama provider")
        if not model:
            raise ValueError("OLLAMA_MODEL is required for the ollama provider")
        self._base_url = base_url.rstrip("/")
        self.model = model
        self._timeout = httpx.Timeout(120.0, connect=10.0)

    def _payload(self, messages: List[LLMMessage], stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": {"temperature": 0.2},
        }

    def complete(self, messages: List[LLMMessage]) -> str:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self._base_url}/api/chat",
                json=self._payload(messages, stream=False),
            )
            resp.raise_for_status()
            data = resp.json()
        return data.get("message", {}).get("content", "") or ""

    def stream(self, messages: List[LLMMessage]) -> Iterable[str]:
        with httpx.Client(timeout=self._timeout) as client:
            with client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json=self._payload(messages, stream=True),
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    delta = obj.get("message", {}).get("content")
                    if delta:
                        yield delta
                    if obj.get("done"):
                        break
