"""OpenAI-compatible LLM adapter.

Works with any provider exposing an OpenAI-style ``/v1/chat/completions``
endpoint (Aliyun DashScope, DeepSeek, OpenRouter, vLLM, LM Studio, ...).

Only a ``base_url`` + ``api_key`` + ``model`` are required, so the same
adapter covers every OpenAI-compatible vendor.
"""

from __future__ import annotations

import asyncio
import logging
import os

import openai
from openai import APIStatusError, APITimeoutError

from pneuma_core.exceptions import LLMTimeoutError
from pneuma_core.llm.adapter import LLMRequest, LLMResponse

DEFAULT_MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 60.0

logger = logging.getLogger(__name__)


def _is_retryable(error: APIStatusError) -> bool:
    """429 (Rate Limit) と 5xx (Server Error) のみリトライ対象."""
    return error.status_code == 429 or error.status_code >= 500


class OpenAICompatAdapter:
    """LLMAdapter implementation for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        api_key: str,
        default_model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.default_model = default_model
        self.base_url = base_url
        self._max_retries = max_retries
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    @classmethod
    def from_env(cls, env_prefix: str = "PNEUMA_LLM") -> OpenAICompatAdapter:
        """Create adapter from environment variables.

        Reads ``{env_prefix}_API_KEY`` (required), ``{env_prefix}_BASE_URL``,
        ``{env_prefix}_MODEL`` and ``{env_prefix}_TIMEOUT``.
        """
        api_key = os.environ.get(f"{env_prefix}_API_KEY")
        if not api_key:
            raise ValueError(
                f"{env_prefix}_API_KEY environment variable is not set"
            )
        base_url = os.environ.get(f"{env_prefix}_BASE_URL")
        model = os.environ.get(f"{env_prefix}_MODEL", DEFAULT_MODEL)
        timeout = float(os.environ.get(f"{env_prefix}_TIMEOUT", DEFAULT_TIMEOUT))
        return cls(
            api_key=api_key,
            default_model=model,
            base_url=base_url,
            timeout=timeout,
        )

    def _resolve_model(self, requested: str | None) -> str:
        """Resolve which model name to send to the API.

        The framework hardcodes some Claude model names internally (e.g. a
        Haiku model for history summarization, Opus for session-end analysis).
        Those names do not exist on OpenAI-compatible providers, so fall back
        to the configured default model instead of failing the call.
        """
        if not requested:
            return self.default_model
        if requested.startswith("claude-"):
            return self.default_model
        return requested

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from an OpenAI-compatible chat completion API.

        Retries up to ``max_retries`` times with exponential backoff for
        429 (Rate Limit) and 5xx (Server Error) responses.
        """
        messages: list[dict] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.extend(request.messages)

        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self._resolve_model(request.model),
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )

                choice = response.choices[0] if response.choices else None
                content = (choice.message.content or "") if choice else ""
                if not content:
                    logger.warning(
                        "Empty LLM response, model=%s", response.model
                    )

                usage: dict = {}
                if response.usage is not None:
                    usage = {
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                    }

                return LLMResponse(
                    content=content,
                    model=response.model,
                    usage=usage,
                )
            except APITimeoutError as e:
                raise LLMTimeoutError(str(e)) from e
            except APIStatusError as e:
                if not _is_retryable(e) or attempt >= self._max_retries:
                    raise
                last_error = e
                delay = 2.0**attempt  # 1.0, 2.0, 4.0
                logger.warning(
                    "API error (status=%d), retrying in %.1fs (attempt %d/%d)",
                    e.status_code,
                    delay,
                    attempt + 1,
                    self._max_retries,
                )
                await asyncio.sleep(delay)

        # Should not reach here, but raise last error for safety
        raise last_error  # type: ignore[misc]
