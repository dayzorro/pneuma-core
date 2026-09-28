"""Embedding service Protocol and OpenAI implementation."""

from __future__ import annotations

import os

import openai

from pneuma_core.protocols.embedding import EmbeddingService

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

__all__ = ["EmbeddingService", "OpenAIEmbeddingService", "DEFAULT_EMBEDDING_MODEL"]


class OpenAIEmbeddingService:
    """EmbeddingService implementation using OpenAI API."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    @classmethod
    def from_env(cls, model: str = DEFAULT_EMBEDDING_MODEL) -> OpenAIEmbeddingService:
        """Create service from environment variables.

        Prefers ``PNEUMA_EMBEDDING_API_KEY`` / ``PNEUMA_EMBEDDING_BASE_URL``,
        falling back to ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``. The base URL
        is what allows using any OpenAI-compatible embedding endpoint.
        """
        api_key = os.environ.get("PNEUMA_EMBEDDING_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY (or PNEUMA_EMBEDDING_API_KEY) environment "
                "variable is not set"
            )
        base_url = os.environ.get("PNEUMA_EMBEDDING_BASE_URL") or os.environ.get(
            "OPENAI_BASE_URL"
        )
        return cls(api_key=api_key, model=model, base_url=base_url)

    async def embed(self, text: str) -> list[float]:
        """Embed a single text into a vector."""
        response = await self._client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts into vectors.

        Empty or whitespace-only strings are filtered out before calling the API
        and replaced with zero vectors in the output to preserve index alignment.
        """
        if not texts:
            return []

        # Identify non-empty texts and their original indices
        non_empty_indices: list[int] = []
        non_empty_texts: list[str] = []
        for i, text in enumerate(texts):
            if text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)

        # If all texts are empty, return zero vectors without calling API
        if not non_empty_texts:
            # We need to determine dimension; use a default for the model
            dim = self._get_default_dimension()
            return [[0.0] * dim for _ in texts]

        # Embed only non-empty texts
        response = await self._client.embeddings.create(
            model=self.model,
            input=non_empty_texts,
        )
        api_embeddings = [item.embedding for item in response.data]

        # Determine dimension from actual API response
        dim = len(api_embeddings[0]) if api_embeddings else self._get_default_dimension()

        # Reconstruct results with zero vectors for empty texts
        results: list[list[float]] = [[0.0] * dim for _ in texts]
        for api_idx, original_idx in enumerate(non_empty_indices):
            results[original_idx] = api_embeddings[api_idx]

        return results

    def _get_default_dimension(self) -> int:
        """Return default embedding dimension based on model name."""
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimension_map.get(self.model, 1536)
