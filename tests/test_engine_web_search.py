"""Tests for 引擎里的联网检索接入与认知提炼调度。"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from pneuma_core.knowledge.models import KnowledgeChunk, KnowledgeHit
from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree
from pneuma_core.models.message import MessageInput
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.runtime.engine import RuntimeEngine
from pneuma_core.websearch import MODE_ALWAYS, MODE_AUTO, MODE_OFF
from pneuma_core.websearch.models import WebSearchResponse, WebSearchResult

_EMOTION_JSON = json.dumps(
    {
        "pleasure": 0.4,
        "arousal": 0.1,
        "dominance": 0.2,
        "emotion_label": "亲切",
        "situation": "在服务台值班",
    },
    ensure_ascii=False,
)

_SPEECH_JSON = json.dumps(
    {"speech": "您好，我帮您看下。", "thought": "先接住人。", "action": "微笑"},
    ensure_ascii=False,
)

_INSIGHT_JSON = json.dumps(
    {"insights": [{"topic": "行业动向", "content": "社区店占比提升。"}]},
    ensure_ascii=False,
)


def _make_character() -> Character:
    return Character(
        id="frontdesk-001",
        name="小润",
        personality=Personality(
            openness=0.55,
            conscientiousness=0.85,
            extraversion=0.72,
            agreeableness=0.9,
            neuroticism=0.15,
        ),
        values=Values(
            self_transcendence=0.85,
            self_enhancement=0.35,
            openness_to_change=0.45,
            conservation=0.75,
        ),
        role_title="华润万家 · 顾客服务前台",
        service_rules="只依据资料回答。",
    )


def _make_llm() -> AsyncMock:
    """按 system prompt 分派：情绪评估 / 认知提炼 / 正常回复。"""

    def route(request) -> AsyncMock:
        prompt = request.system_prompt or ""
        if "情绪分析专家" in prompt:
            return AsyncMock(content=_EMOTION_JSON)
        if "资深培训师" in prompt:
            return AsyncMock(content=_INSIGHT_JSON)
        return AsyncMock(content=_SPEECH_JSON)

    llm = AsyncMock()
    llm.generate = AsyncMock(side_effect=route)
    return llm


def _setup_mocks(*, local_hits: bool = False):
    storage = AsyncMock()
    storage.get_character = AsyncMock(return_value=_make_character())
    storage.get_emotional_state = AsyncMock(
        return_value=EmotionalState(
            pleasure=0.4, arousal=0.1, dominance=0.2,
            emotion_label="亲切", situation="在服务台值班",
        )
    )
    storage.get_goals = AsyncMock(return_value=GoalTree())
    storage.save_emotional_state = AsyncMock()
    storage.save_change = AsyncMock()

    embedding_service = AsyncMock()
    embedding_service.embed = AsyncMock(return_value=[0.1] * 8)

    memory_store = AsyncMock()
    memory_store.get_episodic_by_character = AsyncMock(return_value=[])
    memory_store.get_semantic_by_character = AsyncMock(return_value=[])

    knowledge = None
    if local_hits:
        knowledge = AsyncMock()
        knowledge.search = AsyncMock(
            return_value=[
                KnowledgeHit(
                    chunk=KnowledgeChunk(
                        id="a#0", doc_id="a", title="营业时间", content="08:00 开门"
                    ),
                    score=0.8,
                )
            ]
        )
    elif local_hits is False:
        knowledge = AsyncMock()
        knowledge.search = AsyncMock(return_value=[])

    return storage, embedding_service, memory_store, knowledge


def _web_response() -> WebSearchResponse:
    return WebSearchResponse(
        query="最近零售行业有什么新动向",
        results=[
            WebSearchResult(
                title="零售周报",
                url="https://example.com/a",
                site_name="示例网",
                summary="多家商超调整门店业态。",
            )
        ],
    )


def _message(text: str = "最近零售行业有什么新动向") -> MessageInput:
    return MessageInput(
        content=text, sender_id="u1", sender_name="张女士", sender_type="human"
    )


def _engine(storage, embedding, memory_store, knowledge, **kwargs) -> RuntimeEngine:
    return RuntimeEngine(
        character_id="frontdesk-001",
        storage=storage,
        llm=_make_llm(),
        embedding_service=embedding,
        memory_store=memory_store,
        knowledge_base=knowledge,
        **kwargs,
    )


class TestWebSearchTrigger:
    @pytest.mark.asyncio
    async def test_no_client_means_no_search(self) -> None:
        storage, embedding, memory_store, knowledge = _setup_mocks()
        engine = _engine(storage, embedding, memory_store, knowledge)

        output = await engine.process_message(_message())

        assert output.web_sources == []
        await engine.aclose()

    @pytest.mark.asyncio
    async def test_always_mode_searches_and_injects_results(self) -> None:
        storage, embedding, memory_store, knowledge = _setup_mocks()
        client = AsyncMock()
        client.search = AsyncMock(return_value=_web_response())
        engine = _engine(
            storage, embedding, memory_store, knowledge,
            web_search_client=client, web_search_mode=MODE_ALWAYS,
        )

        output = await engine.process_message(_message())

        client.search.assert_awaited_once_with("最近零售行业有什么新动向")
        assert output.web_sources == [
            {"title": "零售周报", "url": "https://example.com/a", "site": "示例网"}
        ]

        # 主回复那次 generate 的 system prompt 里必须带上联网区段
        main_request = engine._llm.generate.call_args_list[0][0][0]
        assert "## 联网检索结果（实时）" in main_request.system_prompt
        assert "多家商超调整门店业态" in main_request.system_prompt
        await engine.aclose()

    @pytest.mark.asyncio
    async def test_auto_mode_skips_when_local_knowledge_hits(self) -> None:
        storage, embedding, memory_store, knowledge = _setup_mocks(local_hits=True)
        client = AsyncMock()
        client.search = AsyncMock(return_value=_web_response())
        engine = _engine(
            storage, embedding, memory_store, knowledge,
            web_search_client=client, web_search_mode=MODE_AUTO,
        )

        await engine.process_message(_message())

        client.search.assert_not_awaited()
        await engine.aclose()

    @pytest.mark.asyncio
    async def test_auto_mode_searches_on_time_sensitive_gap(self) -> None:
        storage, embedding, memory_store, knowledge = _setup_mocks(local_hits=False)
        client = AsyncMock()
        client.search = AsyncMock(return_value=_web_response())
        engine = _engine(
            storage, embedding, memory_store, knowledge,
            web_search_client=client, web_search_mode=MODE_AUTO,
        )

        await engine.process_message(_message("最近零售行业有什么新动向"))

        client.search.assert_awaited_once()
        await engine.aclose()

    @pytest.mark.asyncio
    async def test_auto_mode_skips_non_time_sensitive_gap(self) -> None:
        storage, embedding, memory_store, knowledge = _setup_mocks(local_hits=False)
        client = AsyncMock()
        client.search = AsyncMock(return_value=_web_response())
        engine = _engine(
            storage, embedding, memory_store, knowledge,
            web_search_client=client, web_search_mode=MODE_AUTO,
        )

        await engine.process_message(_message("帮我写一首诗"))

        client.search.assert_not_awaited()
        await engine.aclose()

    @pytest.mark.asyncio
    async def test_off_mode_never_searches(self) -> None:
        storage, embedding, memory_store, knowledge = _setup_mocks()
        client = AsyncMock()
        client.search = AsyncMock(return_value=_web_response())
        engine = _engine(
            storage, embedding, memory_store, knowledge,
            web_search_client=client, web_search_mode=MODE_OFF,
        )

        await engine.process_message(_message())

        client.search.assert_not_awaited()
        await engine.aclose()

    @pytest.mark.asyncio
    async def test_search_failure_degrades_gracefully(self) -> None:
        storage, embedding, memory_store, knowledge = _setup_mocks()
        client = AsyncMock()
        client.search = AsyncMock(side_effect=RuntimeError("bocha down"))
        engine = _engine(
            storage, embedding, memory_store, knowledge,
            web_search_client=client, web_search_mode=MODE_ALWAYS,
        )

        output = await engine.process_message(_message())

        assert output.content
        assert output.web_sources == []
        components = {m.component for m in output.system_messages}
        assert "web_search" in components
        await engine.aclose()


class TestInsightScheduling:
    @pytest.mark.asyncio
    async def test_schedules_acquisition_when_results_exist(self) -> None:
        storage, embedding, memory_store, knowledge = _setup_mocks()
        client = AsyncMock()
        client.search = AsyncMock(return_value=_web_response())
        acquirer = AsyncMock()
        acquirer.acquire = AsyncMock(return_value=1)
        engine = _engine(
            storage, embedding, memory_store, knowledge,
            web_search_client=client, web_search_mode=MODE_ALWAYS,
            insight_acquirer=acquirer,
        )

        await engine.process_message(_message())
        await engine.aclose()

        acquirer.acquire.assert_awaited_once()
        query, response = acquirer.acquire.await_args[0]
        assert query == "最近零售行业有什么新动向"
        assert response.results[0].title == "零售周报"

    @pytest.mark.asyncio
    async def test_not_scheduled_for_empty_results(self) -> None:
        storage, embedding, memory_store, knowledge = _setup_mocks()
        client = AsyncMock()
        client.search = AsyncMock(return_value=WebSearchResponse(query="q"))
        acquirer = AsyncMock()
        acquirer.acquire = AsyncMock(return_value=0)
        engine = _engine(
            storage, embedding, memory_store, knowledge,
            web_search_client=client, web_search_mode=MODE_ALWAYS,
            insight_acquirer=acquirer,
        )

        await engine.process_message(_message())
        await engine.aclose()

        acquirer.acquire.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_not_scheduled_without_acquirer(self) -> None:
        storage, embedding, memory_store, knowledge = _setup_mocks()
        client = AsyncMock()
        client.search = AsyncMock(return_value=_web_response())
        engine = _engine(
            storage, embedding, memory_store, knowledge,
            web_search_client=client, web_search_mode=MODE_ALWAYS,
        )

        output = await engine.process_message(_message())

        assert output.web_sources
        await engine.aclose()
