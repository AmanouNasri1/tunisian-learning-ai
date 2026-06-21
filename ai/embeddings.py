"""
Embedding provider abstraction.

CRITICAL: the corpus is French + Arabic + Tunisian Darja. Before committing to a
model, test retrieval quality on Darja queries specifically. Store model_name on
every chunk so a model swap only re-embeds stale rows.
"""

from __future__ import annotations

import hashlib
import os
import struct
from typing import Protocol


class EmbeddingProvider(Protocol):
    provider_name: str
    model_name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1536"))
MOCK_MODEL_NAME = "mock-deterministic-v1"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


class EmbeddingConfigurationError(RuntimeError):
    """Raised when an embedding provider is selected but not configured."""


def mock_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Deterministic pseudo-vector from text. Reproducible, NOT a real embedding."""
    values: list[float] = []
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
        for i in range(0, len(digest), 4):
            values.append(struct.unpack("I", digest[i:i + 4])[0] / 2**32 - 0.5)
            if len(values) >= dim:
                break
        counter += 1
    return values


class MockEmbeddingProvider:
    provider_name = "mock"
    model_name = MOCK_MODEL_NAME
    dim = EMBEDDING_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [mock_embedding(text, self.dim) for text in texts]


class OpenAIEmbeddingProvider:
    provider_name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model_name = model or os.environ.get(
            "EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL)
        self.dim = EMBEDDING_DIM
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        if not key.strip():
            raise EmbeddingConfigurationError(
                "OPENAI_API_KEY is missing. Set it before running OpenAI embeddings.")
        from openai import OpenAI
        self._client = OpenAI(api_key=key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model_name, input=texts)
        return [d.embedding for d in resp.data]


OpenAIEmbeddings = OpenAIEmbeddingProvider


def get_embedding_provider(provider: str | None = None) -> EmbeddingProvider:
    provider = (provider or os.environ.get("EMBEDDING_PROVIDER", "openai")).lower()
    if provider == "mock":
        return MockEmbeddingProvider()
    if provider == "openai":
        return OpenAIEmbeddingProvider()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")
