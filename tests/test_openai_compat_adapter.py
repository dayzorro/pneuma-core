"""Tests for the OpenAI-compatible LLM adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIStatusError, APITimeoutError, InternalServerError, RateLimitError

from pneuma_core.exceptions import LLMTimeoutError
from pneuma_core.llm.adapter import LLMAdapter, LLMRequest
from pneuma_core.llm.openai_compat import OpenAICompatAdapter


def _make_api_status_error(status_code: int, message: str = "error") -> APIStatusError:
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    response = httpx.Response(status_code=status_code, request=request)
    if status_code == 429:
        return RateLimitError(message=message, response=response, body=None)
    if status_code == 500:
        return InternalServerError(message=message, response=response, body=None)
    return APIStatusError(message=message, response=response, body=None)


def _mock_completion(content: str = "やあ！", model: str = "qwen-plus") -> MagicMock:
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    resp.choices = [choice]
    resp.model = model
    resp.usage.prompt_tokens = 12
    resp.usage.completion_tokens = 5
    return resp


class TestOpenAICompatAdapterInit:
    def test_satisfies_llm_adapter_protocol(self) -> None:
        adapter = OpenAICompatAdapter(api_key="test-key")
        assert isinstance(adapter, LLMAdapter)

    def test_default_model(self) -> None:
        adapter = OpenAICompatAdapter(api_key="test-key")
        assert adapter.default_model == "gpt-4o-mini"

    def test_custom_model_and_base_url(self) -> None:
        adapter = OpenAICompatAdapter(
            api_key="test-key",
            default_model="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        assert adapter.default_model == "qwen-plus"
        assert adapter.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def test_from_env(self) -> None:
        env = {
            "PNEUMA_LLM_API_KEY": "env-key",
            "PNEUMA_LLM_BASE_URL": "https://api.deepseek.com/v1",
            "PNEUMA_LLM_MODEL": "deepseek-chat",
        }
        with patch.dict("os.environ", env, clear=True):
            adapter = OpenAICompatAdapter.from_env()
            assert adapter.default_model == "deepseek-chat"
            assert adapter.base_url == "https://api.deepseek.com/v1"

    def test_from_env_raises_without_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="PNEUMA_LLM_API_KEY"):
                OpenAICompatAdapter.from_env()


class TestOpenAICompatAdapterGenerate:
    @pytest.fixture
    def adapter(self) -> OpenAICompatAdapter:
        return OpenAICompatAdapter(api_key="test-key", default_model="qwen-plus")

    @pytest.mark.asyncio
    async def test_generate_basic(self, adapter: OpenAICompatAdapter) -> None:
        with patch.object(
            adapter._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=_mock_completion(),
        ):
            result = await adapter.generate(
                LLMRequest(system_prompt="sys", messages=[{"role": "user", "content": "hi"}])
            )
        assert result.content == "やあ！"
        assert result.model == "qwen-plus"
        assert result.usage == {"input_tokens": 12, "output_tokens": 5}

    @pytest.mark.asyncio
    async def test_generate_injects_system_message(
        self, adapter: OpenAICompatAdapter
    ) -> None:
        with patch.object(
            adapter._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=_mock_completion(),
        ) as mock_create:
            await adapter.generate(
                LLMRequest(system_prompt="SYS", messages=[{"role": "user", "content": "hi"}])
            )
        messages = mock_create.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "SYS"}
        assert messages[1] == {"role": "user", "content": "hi"}

    @pytest.mark.asyncio
    async def test_generate_uses_default_model_when_none(
        self, adapter: OpenAICompatAdapter
    ) -> None:
        with patch.object(
            adapter._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=_mock_completion(),
        ) as mock_create:
            await adapter.generate(LLMRequest(system_prompt="s", messages=[]))
        assert mock_create.call_args.kwargs["model"] == "qwen-plus"

    @pytest.mark.asyncio
    async def test_generate_maps_hardcoded_claude_model_to_default(
        self, adapter: OpenAICompatAdapter
    ) -> None:
        """The framework hardcodes Claude model names internally."""
        with patch.object(
            adapter._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=_mock_completion(),
        ) as mock_create:
            await adapter.generate(
                LLMRequest(
                    system_prompt="s",
                    messages=[],
                    model="claude-opus-4-6",
                )
            )
        assert mock_create.call_args.kwargs["model"] == "qwen-plus"

    @pytest.mark.asyncio
    async def test_generate_honors_non_claude_model_override(
        self, adapter: OpenAICompatAdapter
    ) -> None:
        with patch.object(
            adapter._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=_mock_completion(),
        ) as mock_create:
            await adapter.generate(
                LLMRequest(system_prompt="s", messages=[], model="qwen-max")
            )
        assert mock_create.call_args.kwargs["model"] == "qwen-max"

    @pytest.mark.asyncio
    async def test_timeout_becomes_llm_timeout_error(
        self, adapter: OpenAICompatAdapter
    ) -> None:
        request = httpx.Request("POST", "https://example.com/v1/chat/completions")
        with patch.object(
            adapter._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=APITimeoutError(request=request),
        ):
            with pytest.raises(LLMTimeoutError):
                await adapter.generate(LLMRequest(system_prompt="s", messages=[]))

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_then_succeeds(
        self, adapter: OpenAICompatAdapter
    ) -> None:
        with (
            patch.object(
                adapter._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                side_effect=[
                    _make_api_status_error(429),
                    _mock_completion(content="retried"),
                ],
            ) as mock_create,
            patch(
                "pneuma_core.llm.openai_compat.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await adapter.generate(LLMRequest(system_prompt="s", messages=[]))
        assert result.content == "retried"
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_error_propagates(
        self, adapter: OpenAICompatAdapter
    ) -> None:
        with patch.object(
            adapter._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=_make_api_status_error(400),
        ):
            with pytest.raises(APIStatusError):
                await adapter.generate(LLMRequest(system_prompt="s", messages=[]))
