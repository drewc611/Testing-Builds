"""Offline/stub provider. Echoes the retrieved context as a deterministic answer.

Used for CI, for local setup without API keys, and to prove retrieval works
independently of the LLM layer.
"""
from __future__ import annotations

from typing import List

from backend.llm.base import LLMMessage, LLMProvider


class EchoProvider(LLMProvider):
    name = "echo"
    model = "echo-offline"

    def complete(self, messages: List[LLMMessage]) -> str:
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        marker = "Context (retrieved passages"
        context_start = user.find(marker)
        if context_start == -1:
            return "[echo provider] No context found."
        ctx = user[context_start:]
        return (
            "[echo provider — no LLM configured]\n\n"
            "The retrieval system returned the following passages, which would be passed "
            "to the LLM to generate a grounded, cited answer:\n\n" + ctx
        )
