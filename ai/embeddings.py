"""
Embedding provider abstraction.

CRITICAL: the corpus is French + Arabic + Tunisian Darja. Before committing to a
model, test retrieval quality on Darja queries specifically. Store model_name on
every chunk so a model swap only re-embeds stale rows.
"""

from __future__ import annotations

import os
from typing import Protocol


class EmbeddingProvider(Protocol):
    model_name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbeddings:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        from openai import OpenAI
        self.model_name = model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self.dim = int(os.environ.get("EMBEDDING_DIM", "1536"))
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model_name, input=texts)
        return [d.embedding for d in resp.data]


def get_embedding_provider() -> EmbeddingProvider:
    provider = os.environ.get("EMBEDDING_PROVIDER", "openai").lower()
    if provider == "openai":
        return OpenAIEmbeddings()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")
