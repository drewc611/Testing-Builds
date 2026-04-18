"""LLM provider interface. One method: complete(messages) -> str."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def complete(self, messages: List[LLMMessage]) -> str:
        """Return a single full completion string."""

    def stream(self, messages: List[LLMMessage]) -> Iterable[str]:
        """Default: non-streaming providers yield the whole answer once."""
        yield self.complete(messages)


def build_llm(provider: str, **cfg) -> LLMProvider:
    if provider == "anthropic":
        from backend.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=cfg["anthropic_api_key"], model=cfg["anthropic_model"])
    if provider == "openai":
        from backend.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(api_key=cfg["openai_api_key"], model=cfg["openai_model"])
    if provider == "echo":
        from backend.llm.echo_provider import EchoProvider

        return EchoProvider()
    raise ValueError(f"Unknown LLM provider: {provider}")
