"""
LLM provider abstraction. The rest of the system depends only on `LLMClient`,
never on a concrete SDK, so providers are swappable via the LLM_PROVIDER env var.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LLMMessage:
    role: str          # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict = field(default_factory=dict)


class LLMClient(Protocol):
    """Minimal contract every provider must satisfy."""

    def complete(self, system: str, messages: list[LLMMessage],
                 temperature: float = 0.2, max_tokens: int = 2000) -> LLMResponse:
        ...


class AnthropicClient:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        import anthropic  # imported lazily so the package is optional
        self.model = model or os.environ.get("LLM_MODEL", "claude-opus-4-8")
        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def complete(self, system, messages, temperature=0.2, max_tokens=2000):
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResponse(
            text=text, model=self.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            raw=resp.model_dump(),
        )


class OpenAIClient:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        from openai import OpenAI
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o")
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    def complete(self, system, messages, temperature=0.2, max_tokens=2000):
        full = [{"role": "system", "content": system}] + \
               [{"role": m.role, "content": m.content} for m in messages]
        resp = self._client.chat.completions.create(
            model=self.model, messages=full,
            temperature=temperature, max_tokens=max_tokens,
        )
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
            raw=resp.model_dump(),
        )


def get_llm_client() -> LLMClient:
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic":
        return AnthropicClient()
    if provider == "openai":
        return OpenAIClient()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
