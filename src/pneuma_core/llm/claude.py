"""Claude LLM Adapter using the Anthropic SDK."""

from __future__ import annotations

import asyncio
import logging
import os

import anthropic
from anthropic import APIStatusError, APITimeoutError
from anthropic.types import TextBlock

from pneuma_core.exceptions import LLMTimeoutError
from pneuma_core.llm.adapter import LLMRequest, LLMResponse

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_RETRIES = 3

logger = logging.getLogger(__name__)


def _is_retryable(error: APIStatusError) -> bool:
    """仅对 429 (Rate Limit) 与 5xx (Server Error) 进行重试。"""
    return error.status_code == 429 or error.status_code >= 500


class ClaudeAdapter:
    """LLMAdapter implementation for Anthropic Claude API."""

    def __init__(
        self,
        api_key: str,
        default_model: str = DEFAULT_MODEL,
    ) -> None:
        self.default_model = default_model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    @classmethod
    def from_env(cls, default_model: str = DEFAULT_MODEL) -> ClaudeAdapter:
        """Create adapter from ANTHROPIC_API_KEY environment variable."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set"
            )
        return cls(api_key=api_key, default_model=default_model)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from Claude API.

        Retries up to MAX_RETRIES times with exponential backoff
        for 429 (Rate Limit) and 5xx (Server Error) responses.
        """
        last_error: Exception | None = None

        # Build system parameter: structured blocks with cache_control if sections provided
        system_param = self._build_system_param(request)

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self._client.messages.create(
                    model=request.model or self.default_model,
                    system=system_param,
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )

                # Extract and join all text blocks
                text_blocks = [
                    b for b in response.content if isinstance(b, TextBlock)
                ]
                if not text_blocks:
                    logger.warning(
                        "Empty LLM response (no text blocks), stop_reason=%s, model=%s",
                        response.stop_reason,
                        response.model,
                    )
                    content = ""
                else:
                    content = "".join(b.text for b in text_blocks)

                # Build usage dict with cache metrics if available
                usage: dict = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
                cache_creation = getattr(
                    response.usage, "cache_creation_input_tokens", None
                )
                cache_read = getattr(
                    response.usage, "cache_read_input_tokens", None
                )
                if cache_creation is not None:
                    usage["cache_creation_input_tokens"] = cache_creation
                if cache_read is not None:
                    usage["cache_read_input_tokens"] = cache_read

                return LLMResponse(
                    content=content,
                    model=response.model,
                    usage=usage,
                )
            except APITimeoutError as e:
                raise LLMTimeoutError(str(e)) from e
            except APIStatusError as e:
                if not _is_retryable(e) or attempt >= MAX_RETRIES:
                    raise
                last_error = e
                delay = 2.0**attempt  # 1.0, 2.0, 4.0
                logger.warning(
                    "API error (status=%d), retrying in %.1fs (attempt %d/%d)",
                    e.status_code,
                    delay,
                    attempt + 1,
                    MAX_RETRIES,
                )
                await asyncio.sleep(delay)

        # Should not reach here, but raise last error for safety
        raise last_error  # type: ignore[misc]

    @staticmethod
    def _build_system_param(request: LLMRequest) -> str | list[dict]:
        """Build the system parameter for the Anthropic API.

        If the request has cached/dynamic sections, returns a list of
        content blocks with cache_control on the static section.
        Otherwise, returns the plain system_prompt string.
        """
        if request.system_prompt_cached is None:
            return request.system_prompt

        blocks: list[dict] = []

        # Static section with cache_control
        if request.system_prompt_cached:
            blocks.append({
                "type": "text",
                "text": request.system_prompt_cached,
                "cache_control": {"type": "ephemeral"},
            })

        # Dynamic section without cache_control
        if request.system_prompt_dynamic:
            blocks.append({
                "type": "text",
                "text": request.system_prompt_dynamic,
            })

        return blocks
