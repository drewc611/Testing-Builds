"""Anthropic Claude provider."""
from __future__ import annotations

from typing import Iterable, List

from backend.llm.base import LLMMessage, LLMProvider


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the anthropic provider")
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self.model = model

    def _split(self, messages: List[LLMMessage]):
        sys = "\n\n".join(m.content for m in messages if m.role == "system")
        convo = [{"role": m.role, "content": m.content} for m in messages if m.role in ("user", "assistant")]
        return sys, convo

    def complete(self, messages: List[LLMMessage]) -> str:
        system, convo = self._split(messages)
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            messages=convo,
            max_tokens=1024,
            temperature=0.2,
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")

    def stream(self, messages: List[LLMMessage]) -> Iterable[str]:
        system, convo = self._split(messages)
        with self._client.messages.stream(
            model=self.model, system=system, messages=convo, max_tokens=1024, temperature=0.2
        ) as stream:
            for delta in stream.text_stream:
                yield delta
