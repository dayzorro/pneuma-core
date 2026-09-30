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
from collections.abc import AsyncIterator

import openai
from openai import APIStatusError, APITimeoutError

from pneuma_core.exceptions import LLMTimeoutError
from pneuma_core.llm.adapter import LLMRequest, LLMResponse

DEFAULT_MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 60.0

logger = logging.getLogger(__name__)


def _is_retryable(error: APIStatusError) -> bool:
    """仅对 429 (Rate Limit) 与 5xx (Server Error) 进行重试。"""
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
        thinking_param: str | None = None,
    ) -> None:
        self.default_model = default_model
        self.base_url = base_url
        self._max_retries = max_retries
        # 思考开关的 extra_body 键名（例如 DashScope 的 "enable_thinking"）。
        # 为 None 时完全不干预模型的思考行为，避免对不支持的供应商报 400。
        self._thinking_param = thinking_param
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    @classmethod
    def from_env(cls, env_prefix: str = "PNEUMA_LLM") -> OpenAICompatAdapter:
        """Create adapter from environment variables.

        Reads ``{env_prefix}_API_KEY`` (required), ``{env_prefix}_BASE_URL``,
        ``{env_prefix}_MODEL``, ``{env_prefix}_TIMEOUT`` and
        ``{env_prefix}_THINKING_PARAM``.
        """
        api_key = os.environ.get(f"{env_prefix}_API_KEY")
        if not api_key:
            raise ValueError(
                f"{env_prefix}_API_KEY environment variable is not set"
            )
        base_url = os.environ.get(f"{env_prefix}_BASE_URL")
        model = os.environ.get(f"{env_prefix}_MODEL", DEFAULT_MODEL)
        timeout = float(os.environ.get(f"{env_prefix}_TIMEOUT", DEFAULT_TIMEOUT))
        thinking_param = os.environ.get(f"{env_prefix}_THINKING_PARAM") or None
        return cls(
            api_key=api_key,
            default_model=model,
            base_url=base_url,
            timeout=timeout,
            thinking_param=thinking_param,
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
        messages = self._build_messages(request)
        extra_body = self._build_extra_body(request)

        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                kwargs: dict = {
                    "model": self._resolve_model(request.model),
                    "messages": messages,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                }
                if extra_body is not None:
                    kwargs["extra_body"] = extra_body
                response = await self._client.chat.completions.create(**kwargs)

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

    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream a chat completion, yielding incremental content deltas.

        Unlike ``generate`` this performs no retries: once bytes start
        flowing a retry would duplicate already-emitted text. Callers that
        need retry semantics should fall back to ``generate``.
        """
        messages = self._build_messages(request)
        extra_body = self._build_extra_body(request)

        kwargs: dict = {
            "model": self._resolve_model(request.model),
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }
        if extra_body is not None:
            kwargs["extra_body"] = extra_body

        try:
            stream = await self._client.chat.completions.create(**kwargs)
        except APITimeoutError as e:
            raise LLMTimeoutError(str(e)) from e

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                yield piece

    def _build_messages(self, request: LLMRequest) -> list[dict]:
        """Prepend the system prompt to the message list."""
        messages: list[dict] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.extend(request.messages)
        return messages

    def _build_extra_body(self, request: LLMRequest) -> dict | None:
        """Build the ``extra_body`` for provider-specific options.

        Currently only controls the thinking / reasoning switch. When
        ``thinking_param`` is unset the adapter never touches it, so
        providers that do not understand the parameter are unaffected.
        """
        if not self._thinking_param:
            return None
        # None -> 适配器默认关闭思考；False -> 同样关闭；True -> 不发送，保留思考
        if request.enable_thinking is True:
            return None
        return {self._thinking_param: False}
