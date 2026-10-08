"""Tests for RuntimeEngine (Issue #29, #45)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pneuma_core.models.change_record import ChangeRecord
from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.models.message import MessageInput, MessageOutput, ToolCall
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.runtime.engine import RuntimeEngine
from pneuma_core.runtime.prompt_cache import PromptCache

NOW = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)


def _make_character() -> Character:
    return Character(
        id="aine-001",
        name="アイネ",
        personality=Personality(
            openness=0.8,
            conscientiousness=0.6,
            extraversion=0.7,
            agreeableness=0.9,
            neuroticism=0.3,
        ),
        values=Values(
            self_transcendence=0.8,
            self_enhancement=0.3,
            openness_to_change=0.7,
            conservation=0.4,
        ),
        profile="好奇心旺盛な AI",
        speaking_style="丁寧だけど親しみやすい口調",
    )


def _make_emotional_state() -> EmotionalState:
    return EmotionalState(
        pleasure=0.3,
        arousal=0.1,
        dominance=0.0,
        emotion_label="安らぎ",
        situation="穏やかな会話中",
    )


def _make_message_input(
    content: str = "こんにちは！",
    sender_id: str = "user-1",
    sender_name: str = "ユーザー",
    sender_type: str = "human",
) -> MessageInput:
    return MessageInput(
        content=content,
        sender_id=sender_id,
        sender_name=sender_name,
        sender_type=sender_type,
    )


def _make_llm_response(content: str = "やあ！元気？") -> AsyncMock:
    response = AsyncMock()
    response.content = content
    response.model = "test-model"
    response.usage = {}
    return response


def _make_tier1_gate_response(changed: bool = True) -> AsyncMock:
    response = AsyncMock()
    response.content = json.dumps({"changed": changed})
    response.model = "test-model"
    response.usage = {}
    return response


def _make_emotion_response() -> AsyncMock:
    response = AsyncMock()
    response.content = json.dumps(
        {
            "pleasure": 0.5,
            "arousal": 0.3,
            "dominance": 0.1,
            "emotion_label": "喜び",
            "situation": "楽しい会話をしている",
        },
        ensure_ascii=False,
    )
    response.model = "test-model"
    response.usage = {}
    return response


def _setup_mocks():
    """Create standard mock dependencies for RuntimeEngine."""
    storage = AsyncMock()
    storage.get_character = AsyncMock(return_value=_make_character())
    storage.get_emotional_state = AsyncMock(return_value=_make_emotional_state())
    storage.get_goals = AsyncMock(return_value=GoalTree())
    storage.save_emotional_state = AsyncMock()
    storage.save_change = AsyncMock()
    storage.get_episodic_memories = AsyncMock(return_value=[])
    storage.get_semantic_memories = AsyncMock(return_value=[])

    llm = AsyncMock()
    # First call: main response, Second call: emotion estimation (direct, no gate)
    llm.generate = AsyncMock(
        side_effect=[
            _make_llm_response(),
            _make_emotion_response(),
        ]
    )

    embedding_service = AsyncMock()
    embedding_service.embed = AsyncMock(return_value=[0.1] * 10)

    memory_store = AsyncMock()
    memory_store.get_episodic_by_character = AsyncMock(return_value=[])
    memory_store.get_semantic_by_character = AsyncMock(return_value=[])

    return storage, llm, embedding_service, memory_store


# --- MessageInput / MessageOutput モデル ---


class TestMessageInput:
    """MessageInput データモデルの検証."""

    def test_required_fields(self) -> None:
        msg = MessageInput(
            content="hello",
            sender_id="user-1",
            sender_name="User",
            sender_type="human",
        )
        assert msg.content == "hello"
        assert msg.sender_id == "user-1"
        assert msg.sender_name == "User"
        assert msg.sender_type == "human"

    def test_optional_fields_defaults(self) -> None:
        msg = MessageInput(
            content="test",
            sender_id="u1",
            sender_name="U",
            sender_type="system",
        )
        assert msg.channel is None
        assert msg.metadata == {}

    def test_all_sender_types(self) -> None:
        for sender_type in ("human", "character", "system"):
            msg = MessageInput(
                content="test",
                sender_id="id",
                sender_name="name",
                sender_type=sender_type,
            )
            assert msg.sender_type == sender_type

    def test_with_channel_and_metadata(self) -> None:
        msg = MessageInput(
            content="test",
            sender_id="u1",
            sender_name="U",
            sender_type="human",
            channel="discord",
            metadata={"key": "value"},
        )
        assert msg.channel == "discord"
        assert msg.metadata == {"key": "value"}

    def test_empty_content_raises(self) -> None:
        with pytest.raises(ValueError, match="content must not be empty"):
            MessageInput(
                content="",
                sender_id="u1",
                sender_name="U",
                sender_type="human",
            )

    def test_invalid_sender_type_raises(self) -> None:
        with pytest.raises(ValueError, match="sender_type must be one of"):
            MessageInput(
                content="test",
                sender_id="u1",
                sender_name="U",
                sender_type="invalid",  # type: ignore[arg-type]
            )


class TestMessageOutput:
    """MessageOutput データモデルの検証."""

    def test_required_fields(self) -> None:
        state = _make_emotional_state()
        out = MessageOutput(content="response", emotion=state)
        assert out.content == "response"
        assert out.emotion == state

    def test_optional_fields_defaults(self) -> None:
        out = MessageOutput(content="test", emotion=_make_emotional_state())
        assert out.tool_calls == []
        assert out.internal_changes == []


class TestToolCall:
    """ToolCall データモデルの検証."""

    def test_creation(self) -> None:
        tc = ToolCall(name="search", arguments={"query": "test"})
        assert tc.name == "search"
        assert tc.arguments == {"query": "test"}


# --- RuntimeEngine process_message ---


class TestProcessMessage:
    """process_message パイプラインの検証."""

    @pytest.mark.asyncio
    async def test_returns_message_output(self) -> None:
        """process_message が MessageOutput を返す."""
        storage, llm, embedding, memory_store = _setup_mocks()
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        result = await engine.process_message(_make_message_input())

        assert isinstance(result, MessageOutput)
        assert isinstance(result.content, str)
        assert len(result.content) > 0
        assert isinstance(result.emotion, EmotionalState)

    @pytest.mark.asyncio
    async def test_calls_llm_with_system_prompt(self) -> None:
        """LLM がシステムプロンプト付きで呼び出される."""
        storage, llm, embedding, memory_store = _setup_mocks()
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        await engine.process_message(_make_message_input())

        # Main LLM call (first call)
        main_call = llm.generate.call_args_list[0]
        request = main_call[0][0]
        assert len(request.system_prompt) > 0
        assert "アイネ" in request.system_prompt

    @pytest.mark.asyncio
    async def test_sender_type_human(self) -> None:
        """human sender_type のメッセージが処理される."""
        storage, llm, embedding, memory_store = _setup_mocks()
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        result = await engine.process_message(
            _make_message_input(sender_type="human")
        )
        assert isinstance(result, MessageOutput)

    @pytest.mark.asyncio
    async def test_sender_type_character(self) -> None:
        """character sender_type のメッセージが同一パイプラインで処理される."""
        storage, llm, embedding, memory_store = _setup_mocks()
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        result = await engine.process_message(
            _make_message_input(sender_type="character", sender_name="他キャラ")
        )
        assert isinstance(result, MessageOutput)

    @pytest.mark.asyncio
    async def test_sender_type_system(self) -> None:
        """system sender_type のメッセージが同一パイプラインで処理される."""
        storage, llm, embedding, memory_store = _setup_mocks()
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        result = await engine.process_message(
            _make_message_input(sender_type="system", sender_name="System")
        )
        assert isinstance(result, MessageOutput)

    @pytest.mark.asyncio
    async def test_emotion_estimation_called(self) -> None:
        """感情推定が非同期で実行され、次ターンで反映される."""
        storage, llm, embedding, memory_store = _setup_mocks()
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        # Use emotional message to trigger hybrid system
        result = await engine.process_message(
            _make_message_input(content="嬉しい！最高だよ！")
        )

        # First turn returns current emotion (async estimation runs in background)
        assert result.emotion.emotion_label == "安らぎ"

        # Wait for background emotion estimation to complete
        await asyncio.sleep(0.05)

        # LLM: 1 (response) + 1 (emotion estimation) = 2 (no Tier 1 gate)
        assert llm.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_emotional_state_saved(self) -> None:
        """感情状態が非同期的に Storage に保存される."""
        storage, llm, embedding, memory_store = _setup_mocks()
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        # Use emotional message to trigger emotion estimation
        await engine.process_message(
            _make_message_input(content="嬉しい！やった！")
        )

        # Wait for background emotion estimation to complete
        await asyncio.sleep(0.05)

        storage.save_emotional_state.assert_called_once()
        call_args = storage.save_emotional_state.call_args
        assert call_args[0][0] == "aine-001"
        assert isinstance(call_args[0][1], EmotionalState)

    @pytest.mark.asyncio
    async def test_memory_search_called(self) -> None:
        """記憶検索が実行される."""
        storage, llm, embedding, memory_store = _setup_mocks()
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        await engine.process_message(_make_message_input())

        # Embedding should be called for the query
        embedding.embed.assert_called()


# --- 会話履歴管理 ---


class TestConversationHistory:
    """会話履歴の管理テスト."""

    @pytest.mark.asyncio
    async def test_history_accumulates(self) -> None:
        """メッセージが会話履歴に蓄積される."""
        storage, llm, embedding, memory_store = _setup_mocks()
        # Multiple calls: each message = response + emotion estimation = 2 calls each
        llm.generate = AsyncMock(
            side_effect=[
                _make_llm_response("応答1"),
                _make_emotion_response(),  # emotion for msg1
                _make_llm_response("応答2"),
                _make_emotion_response(),  # emotion for msg2
            ]
        )
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        await engine.process_message(_make_message_input(content="メッセージ1"))
        await asyncio.sleep(0.05)
        await engine.process_message(_make_message_input(content="メッセージ2"))

        # Second main LLM call (index 2) should have more messages
        second_main_call = llm.generate.call_args_list[2]  # 3rd call = 2nd main response
        request = second_main_call[0][0]
        assert len(request.messages) >= 3  # msg1 + response1 + msg2

    @pytest.mark.asyncio
    async def test_history_limit(self) -> None:
        """会話履歴が上限を超えない（要約メッセージ除く）."""
        storage, llm, embedding, memory_store = _setup_mocks()

        # Neutral messages: only main response per turn (no emotion trigger)
        responses = []
        for i in range(60):
            responses.append(_make_llm_response(f"応答{i}"))
        llm.generate = AsyncMock(side_effect=responses)

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            history_limit=100,
        )

        for i in range(60):
            await engine.process_message(
                _make_message_input(content=f"msg{i}")
            )

        # Internal history should be capped at 100
        assert len(engine._history) <= 100
        # Messages sent to LLM may include +1 summary message
        last_main_call = llm.generate.call_args_list[-2]  # last main call
        request = last_main_call[0][0]
        assert len(request.messages) <= 101  # 100 history + 1 summary


# --- internal_changes ---


class TestInternalChanges:
    """internal_changes の記録テスト."""

    @pytest.mark.asyncio
    async def test_emotion_change_recorded(self) -> None:
        """感情変化が internal_changes に記録される."""
        storage, llm, embedding, memory_store = _setup_mocks()
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        result = await engine.process_message(_make_message_input())

        emotion_changes = [
            c for c in result.internal_changes if c.type == "emotion_updated"
        ]
        assert len(emotion_changes) >= 1

    @pytest.mark.asyncio
    async def test_change_saved_to_storage(self) -> None:
        """ChangeRecord が Storage に保存される."""
        storage, llm, embedding, memory_store = _setup_mocks()
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        await engine.process_message(_make_message_input())

        storage.save_change.assert_called()


# --- エラーハンドリング ---


class TestErrorHandling:
    """エラーハンドリングの検証."""

    @pytest.mark.asyncio
    async def test_llm_failure_returns_fallback(self) -> None:
        """LLM が失敗した場合、フォールバック応答を返す."""
        storage, llm, embedding, memory_store = _setup_mocks()
        llm.generate = AsyncMock(side_effect=RuntimeError("API error"))
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        result = await engine.process_message(_make_message_input())

        assert isinstance(result, MessageOutput)
        assert len(result.content) > 0  # fallback message

    @pytest.mark.asyncio
    async def test_character_not_found_raises(self) -> None:
        """キャラクターが見つからない場合、エラーになる."""
        storage, llm, embedding, memory_store = _setup_mocks()
        storage.get_character = AsyncMock(return_value=None)
        engine = RuntimeEngine(
            character_id="nonexistent",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        with pytest.raises(ValueError, match="Character not found"):
            await engine.process_message(_make_message_input())

    @pytest.mark.asyncio
    async def test_embedding_failure_continues(self) -> None:
        """Embedding 失敗でも応答は返される（記憶検索スキップ）."""
        storage, llm, embedding, memory_store = _setup_mocks()
        embedding.embed = AsyncMock(side_effect=RuntimeError("Embedding error"))
        # LLM still works: main response only (emotion fails gracefully)
        llm.generate = AsyncMock(return_value=_make_llm_response())
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        result = await engine.process_message(_make_message_input())

        assert isinstance(result, MessageOutput)
        assert len(result.content) > 0


# --- Issue #45: PromptCache 統合 ---


class TestPromptCacheIntegration:
    """RuntimeEngine が PromptCache を使用することの検証 (#45)."""

    @pytest.mark.asyncio
    async def test_accepts_prompt_cache_parameter(self) -> None:
        """RuntimeEngine が prompt_cache パラメータを受け取れる."""
        storage, llm, embedding, memory_store = _setup_mocks()
        prompt_cache = PromptCache()

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            prompt_cache=prompt_cache,
        )

        result = await engine.process_message(_make_message_input())
        assert isinstance(result, MessageOutput)

    @pytest.mark.asyncio
    async def test_uses_prompt_cache_for_system_prompt(self) -> None:
        """PromptCache を使ってシステムプロンプトが構築される."""
        storage, llm, embedding, memory_store = _setup_mocks()
        prompt_cache = PromptCache()

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            prompt_cache=prompt_cache,
        )

        await engine.process_message(_make_message_input())

        # LLM should receive a system prompt containing character info
        main_call = llm.generate.call_args_list[0]
        request = main_call[0][0]
        assert "アイネ" in request.system_prompt

    @pytest.mark.asyncio
    async def test_prompt_cache_build_called(self) -> None:
        """PromptCache.build が呼び出される."""
        storage, llm, embedding, memory_store = _setup_mocks()
        prompt_cache = MagicMock(spec=PromptCache)
        # Mock the build method to return a CachedPrompt-like object
        mock_cached = MagicMock()
        mock_cached.full_prompt = "mocked system prompt with アイネ"
        prompt_cache.build = MagicMock(return_value=mock_cached)

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            prompt_cache=prompt_cache,
        )

        await engine.process_message(_make_message_input())

        # PromptCache.build should be called
        prompt_cache.build.assert_called_once()

    @pytest.mark.asyncio
    async def test_static_section_cached_across_turns(self) -> None:
        """複数ターンで静的セクションがキャッシュされる."""
        storage, llm, embedding, memory_store = _setup_mocks()
        # Need multiple LLM responses for multiple turns
        # (main response + emotion estimation per turn)
        llm.generate = AsyncMock(
            side_effect=[
                _make_llm_response("応答1"),
                _make_emotion_response(),
                _make_llm_response("応答2"),
                _make_emotion_response(),
            ]
        )
        prompt_cache = PromptCache()

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            prompt_cache=prompt_cache,
        )

        await engine.process_message(_make_message_input(content="msg1"))
        await asyncio.sleep(0.05)  # Wait for async emotion task
        await engine.process_message(_make_message_input(content="msg2"))

        # Find main LLM calls (not emotion estimation calls)
        main_calls = [
            call for call in llm.generate.call_args_list
            if "アイネ" in call[0][0].system_prompt
        ]
        assert len(main_calls) == 2
        assert "アイネ" in main_calls[0][0][0].system_prompt
        assert "アイネ" in main_calls[1][0][0].system_prompt

    @pytest.mark.asyncio
    async def test_backward_compatible_without_prompt_cache(self) -> None:
        """prompt_cache を渡さなくても従来通り動作する."""
        storage, llm, embedding, memory_store = _setup_mocks()
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            # No prompt_cache parameter
        )

        result = await engine.process_message(_make_message_input())
        assert isinstance(result, MessageOutput)
        assert "アイネ" in llm.generate.call_args_list[0][0][0].system_prompt


# --- Issue #45: 感情推定の非同期化 ---


class TestAsyncEmotionEstimation:
    """感情推定が非同期で実行されることの検証 (#45)."""

    @pytest.mark.asyncio
    async def test_first_turn_returns_current_emotion(self) -> None:
        """最初のターンでは現在の感情（storage から取得）が返される."""
        storage, llm, embedding, memory_store = _setup_mocks()
        # LLM only needs to return main response (emotion is async)
        llm.generate = AsyncMock(return_value=_make_llm_response())
        prompt_cache = PromptCache()

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            prompt_cache=prompt_cache,
        )

        result = await engine.process_message(_make_message_input())

        # First turn should return current emotion from storage (安らぎ)
        # because emotion estimation runs asynchronously
        assert result.emotion.emotion_label == "安らぎ"

    @pytest.mark.asyncio
    async def test_response_not_blocked_by_emotion(self) -> None:
        """応答が感情推定によってブロックされない."""
        storage, llm, embedding, memory_store = _setup_mocks()

        # Emotion estimation takes a long time (simulated)
        emotion_started = asyncio.Event()
        emotion_proceed = asyncio.Event()

        call_count = 0

        async def slow_generate(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Main response: instant
                return _make_llm_response()
            else:
                # Emotion estimation: slow
                emotion_started.set()
                await emotion_proceed.wait()
                return _make_emotion_response()

        llm.generate = AsyncMock(side_effect=slow_generate)
        prompt_cache = PromptCache()

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            prompt_cache=prompt_cache,
        )

        # process_message should return quickly without waiting for emotion
        result = await engine.process_message(_make_message_input())
        assert isinstance(result, MessageOutput)
        assert result.content == "やあ！元気？"

        # Let emotion estimation complete
        emotion_proceed.set()
        # Give the background task time to complete
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_emotion_updated_on_next_turn(self) -> None:
        """感情推定の結果が次のターンで反映される."""
        storage, llm, embedding, memory_store = _setup_mocks()

        call_count = 0

        async def generate_responses(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Turn 1: main response
                return _make_llm_response("応答1")
            elif call_count == 2:
                # Turn 1: direct emotion estimation (no Tier 1 gate)
                return _make_emotion_response()
            elif call_count == 3:
                # Turn 2: main response
                return _make_llm_response("応答2")
            else:
                # Turn 2: emotion estimation
                return _make_emotion_response()

        llm.generate = AsyncMock(side_effect=generate_responses)
        prompt_cache = PromptCache()

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            prompt_cache=prompt_cache,
        )

        # Turn 1: emotion estimation runs in background, current emotion returned
        result1 = await engine.process_message(
            _make_message_input(content="嬉しい！最高だよ！")
        )
        assert result1.emotion.emotion_label == "安らぎ"

        # Wait for background emotion task to complete
        await asyncio.sleep(0.05)

        # Turn 2: emotion should be updated from Turn 1's estimation (喜び)
        result2 = await engine.process_message(
            _make_message_input(content="普通のメッセージ")
        )
        assert result2.emotion.emotion_label == "喜び"

    @pytest.mark.asyncio
    async def test_emotion_estimation_failure_keeps_current(self) -> None:
        """感情推定が失敗しても現在の感情が維持される."""
        storage, llm, embedding, memory_store = _setup_mocks()

        call_count = 0

        async def generate_responses(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Main response
                return _make_llm_response()
            else:
                # Emotion estimation fails (both Tier 1 and Tier 2)
                raise RuntimeError("Emotion API error")

        llm.generate = AsyncMock(side_effect=generate_responses)
        prompt_cache = PromptCache()

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            prompt_cache=prompt_cache,
        )

        # Use emotional message to trigger estimation (which then fails)
        result = await engine.process_message(
            _make_message_input(content="嬉しい！最高！")
        )

        # Should return current emotion (安らぎ) even if estimation fails
        assert result.emotion.emotion_label == "安らぎ"

    @pytest.mark.asyncio
    async def test_emotion_saved_after_async_estimation(self) -> None:
        """非同期感情推定の結果が storage に保存される."""
        storage, llm, embedding, memory_store = _setup_mocks()

        call_count = 0

        async def generate_responses(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_llm_response()
            else:
                # Direct emotion estimation (no Tier 1 gate)
                return _make_emotion_response()

        llm.generate = AsyncMock(side_effect=generate_responses)
        prompt_cache = PromptCache()

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            prompt_cache=prompt_cache,
        )

        # Use emotional message to trigger estimation
        await engine.process_message(
            _make_message_input(content="嬉しい！やった！")
        )

        # Wait for background emotion task to complete
        await asyncio.sleep(0.05)

        # Emotion should be saved to storage by the background task
        storage.save_emotional_state.assert_called()
        # The saved emotion should be the estimated one
        saved_call = storage.save_emotional_state.call_args
        assert saved_call[0][0] == "aine-001"
        saved_emotion = saved_call[0][1]
        assert saved_emotion.emotion_label == "喜び"


# --- Issue #56: モデル選択パラメータ ---


class TestModelSelection:
    """RuntimeEngine のモデル選択パラメータの検証 (#56)."""

    @pytest.mark.asyncio
    async def test_accepts_model_parameters(self) -> None:
        """RuntimeEngine が response_model / emotion_model パラメータを受け取れる."""
        storage, llm, embedding, memory_store = _setup_mocks()

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            response_model="claude-sonnet-4-20250514",
            emotion_model="claude-haiku-4-20250514",
        )

        result = await engine.process_message(_make_message_input())
        assert isinstance(result, MessageOutput)

    @pytest.mark.asyncio
    async def test_response_model_set_in_llm_request(self) -> None:
        """response_model が応答生成の LLMRequest.model に設定される."""
        storage, llm, embedding, memory_store = _setup_mocks()

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            response_model="claude-sonnet-4-20250514",
        )

        await engine.process_message(_make_message_input())

        # Main LLM call (first call) should have the response model
        main_call = llm.generate.call_args_list[0]
        request = main_call[0][0]
        assert request.model == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_emotion_model_passed_to_emotion_engine(self) -> None:
        """emotion_model が EmotionEngine の LLMRequest.model に設定される."""
        storage, llm, embedding, memory_store = _setup_mocks()

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            emotion_model="claude-haiku-4-20250514",
        )

        await engine.process_message(
            _make_message_input(content="嬉しい！やった！")
        )

        # Wait for background emotion estimation to complete
        await asyncio.sleep(0.05)

        # LLM: 1 (response) + 1 (emotion estimation) = 2 (no Tier 1 gate)
        assert llm.generate.call_count == 2
        # Emotion estimation call should have the emotion model
        emotion_call = llm.generate.call_args_list[1]
        assert emotion_call[0][0].model == "claude-haiku-4-20250514"

    @pytest.mark.asyncio
    async def test_default_models_are_none(self) -> None:
        """モデル指定なしの場合、LLMRequest.model は None（後方互換）."""
        storage, llm, embedding, memory_store = _setup_mocks()

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        await engine.process_message(_make_message_input())

        # Main LLM call should have model=None
        main_call = llm.generate.call_args_list[0]
        request = main_call[0][0]
        assert request.model is None

    @pytest.mark.asyncio
    async def test_different_models_for_response_and_emotion(self) -> None:
        """応答生成と感情推定で異なるモデルを使用できる."""
        storage, llm, embedding, memory_store = _setup_mocks()

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            response_model="claude-sonnet-4-20250514",
            emotion_model="claude-haiku-4-20250514",
        )

        await engine.process_message(
            _make_message_input(content="嬉しい！最高！")
        )

        # Wait for background emotion estimation
        await asyncio.sleep(0.05)

        # First call (response) uses Sonnet
        response_request = llm.generate.call_args_list[0][0][0]
        assert response_request.model == "claude-sonnet-4-20250514"

        # Emotion estimation call uses Haiku (no Tier 1 gate)
        emotion_request = llm.generate.call_args_list[1][0][0]
        assert emotion_request.model == "claude-haiku-4-20250514"


# --- Issue #131: 毎ターン直接推定統合 ---


class TestDirectEstimationIntegration:
    """RuntimeEngine の毎ターン直接推定統合テスト (#131)."""

    @pytest.mark.asyncio
    async def test_turn_count_increments(self) -> None:
        """process_message 呼び出しごとにターンカウントが増加する."""
        storage, llm, embedding, memory_store = _setup_mocks()
        # 複数ターン用のレスポンス (each turn = response + emotion)
        llm.generate = AsyncMock(
            side_effect=[
                _make_llm_response("応答1"),
                _make_emotion_response(),
                _make_llm_response("応答2"),
                _make_emotion_response(),
                _make_llm_response("応答3"),
                _make_emotion_response(),
            ]
        )
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        await engine.process_message(_make_message_input(content="msg1"))
        await engine.process_message(_make_message_input(content="msg2"))
        await engine.process_message(_make_message_input(content="msg3"))

        assert engine._turn_count == 3

    @pytest.mark.asyncio
    async def test_emotion_always_estimated_even_on_neutral_message(self) -> None:
        """感情キーワードがない普通のメッセージでも感情推定が実行される (#131)."""
        storage, llm, embedding, memory_store = _setup_mocks()
        llm.generate = AsyncMock(
            side_effect=[
                _make_llm_response(),
                _make_emotion_response(),
            ]
        )
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        result = await engine.process_message(
            _make_message_input(content="明日の天気を教えて")
        )

        await asyncio.sleep(0.05)

        # LLM: 1 (response) + 1 (emotion estimation) = 2 (every turn)
        assert llm.generate.call_count == 2
        # 応答時の感情は直前の状態が返される
        assert result.emotion.emotion_label == "安らぎ"

    @pytest.mark.asyncio
    async def test_emotion_estimated_on_emotional_message(self) -> None:
        """感情的なメッセージでは直接感情推定が実行される (#131)."""
        storage, llm, embedding, memory_store = _setup_mocks()

        call_count = 0

        async def generate_responses(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Main response
                return _make_llm_response()
            else:
                # Direct emotion estimation (no Tier 1 gate)
                return _make_emotion_response()

        llm.generate = AsyncMock(side_effect=generate_responses)
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        result = await engine.process_message(
            _make_message_input(content="めっちゃ嬉しい！最高！")
        )

        await asyncio.sleep(0.05)

        # LLM: 1 (response) + 1 (emotion estimation) = 2 (no Tier 1 gate)
        assert llm.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_emotion_change_record_includes_trigger_type(self) -> None:
        """ChangeRecord にトリガー判定結果が含まれる."""
        storage, llm, embedding, memory_store = _setup_mocks()
        llm.generate = AsyncMock(return_value=_make_llm_response())
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        result = await engine.process_message(
            _make_message_input(content="普通の会話")
        )

        await asyncio.sleep(0.05)

        # emotion_updated の ChangeRecord に trigger_type が含まれる
        emotion_changes = [
            c for c in result.internal_changes if c.type == "emotion_updated"
        ]
        assert len(emotion_changes) >= 1
        # after に trigger_type が含まれる
        assert "trigger_type" in emotion_changes[0].after

    @pytest.mark.asyncio
    async def test_safety_net_triggers_at_interval(self) -> None:
        """Safety Net interval のターンで無条件推定が実行される."""
        storage, llm, embedding, memory_store = _setup_mocks()

        responses = []
        for i in range(16):
            responses.append(_make_llm_response(f"応答{i}"))
        # Safety net at turn 15: Tier 2 estimation
        responses.append(_make_emotion_response())

        llm.generate = AsyncMock(side_effect=responses)

        from pneuma_core.runtime.emotion_engine import EmotionConfig

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        # 15 ターン分のメッセージを送信（感情キーワードなし）
        for i in range(15):
            await engine.process_message(
                _make_message_input(content=f"普通の会話{i}")
            )

        await asyncio.sleep(0.05)

        # 15ターン目で Safety Net がトリガーされるので
        # LLM 呼び出し回数は 15回(応答) + 1回(Tier 2) = 16回以上
        assert llm.generate.call_count >= 16


# --- Issue #86: 構造化出力 (speech/thought/action) ---


def _make_structured_llm_response(
    speech: str = "ふふ、頑張ってるね",
    thought: str | None = "最近ちょっと無理してるかも",
    action: str | None = "優しく微笑む",
) -> AsyncMock:
    """Create a mock LLM response with structured JSON output."""
    data: dict = {"speech": speech}
    if thought is not None:
        data["thought"] = thought
    if action is not None:
        data["action"] = action
    response = AsyncMock()
    response.content = json.dumps(data, ensure_ascii=False)
    response.model = "test-model"
    response.usage = {}
    return response


class TestStructuredOutput:
    """構造化出力 (speech/thought/action) の統合テスト (#86)."""

    @pytest.mark.asyncio
    async def test_json_response_content_is_speech(self) -> None:
        """JSON レスポンスがパースされ、output.content が speech になる."""
        storage, llm, embedding, memory_store = _setup_mocks()
        llm.generate = AsyncMock(
            return_value=_make_structured_llm_response(speech="こんにちは")
        )
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        result = await engine.process_message(_make_message_input())

        assert result.content == "こんにちは"

    @pytest.mark.asyncio
    async def test_json_response_thought_populated(self) -> None:
        """JSON レスポンスの thought が output.thought に格納される."""
        storage, llm, embedding, memory_store = _setup_mocks()
        llm.generate = AsyncMock(
            return_value=_make_structured_llm_response(thought="考え中…")
        )
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        result = await engine.process_message(_make_message_input())

        assert result.thought == "考え中…"

    @pytest.mark.asyncio
    async def test_json_response_action_populated(self) -> None:
        """JSON レスポンスの action が output.action に格納される."""
        storage, llm, embedding, memory_store = _setup_mocks()
        llm.generate = AsyncMock(
            return_value=_make_structured_llm_response(action="首をかしげる")
        )
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        result = await engine.process_message(_make_message_input())

        assert result.action == "首をかしげる"

    @pytest.mark.asyncio
    async def test_plain_text_backward_compatible(self) -> None:
        """プレーンテキスト応答は後方互換（content=全文、thought/action=None）."""
        storage, llm, embedding, memory_store = _setup_mocks()
        llm.generate = AsyncMock(
            return_value=_make_llm_response("やあ！元気？")
        )
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        result = await engine.process_message(_make_message_input())

        assert result.content == "やあ！元気？"
        assert result.thought is None
        assert result.action is None

    @pytest.mark.asyncio
    async def test_history_stores_structured_json(self) -> None:
        """会話履歴には正規化した構造化 JSON を格納する（書式ドリフト防止）."""
        storage, llm, embedding, memory_store = _setup_mocks()

        call_count = 0

        async def generate_responses(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_structured_llm_response(
                    speech="こんにちは",
                    thought="嬉しそうだな",
                    action="微笑む",
                )
            else:
                return _make_structured_llm_response(speech="はい")

        llm.generate = AsyncMock(side_effect=generate_responses)
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        await engine.process_message(_make_message_input())
        await engine.process_message(_make_message_input(content="2回目"))

        # 2回目の LLM 呼び出しで、履歴にある assistant メッセージを確認
        second_call = llm.generate.call_args_list[1]
        request = second_call[0][0]
        assistant_msgs = [m for m in request.messages if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1
        # 履歴は構造化 JSON（thought / action を含む）で、書式の見本として機能する。
        # 純テキストだと数ターン後にモデルが JSON 出力をやめてしまうため。
        payload = json.loads(assistant_msgs[0]["content"])
        assert payload["speech"] == "こんにちは"
        assert payload["thought"] == "嬉しそうだな"
        assert payload["action"] == "微笑む"

    @pytest.mark.asyncio
    async def test_fallback_response_no_thought_action(self) -> None:
        """LLM 失敗時のフォールバック応答は thought/action が None."""
        storage, llm, embedding, memory_store = _setup_mocks()
        llm.generate = AsyncMock(side_effect=RuntimeError("API error"))
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        result = await engine.process_message(_make_message_input())

        assert len(result.content) > 0
        assert result.thought is None
        assert result.action is None

    @pytest.mark.asyncio
    async def test_null_speech_empty_string(self) -> None:
        """speech=null の場合、output.content は空文字列."""
        storage, llm, embedding, memory_store = _setup_mocks()
        response = AsyncMock()
        response.content = json.dumps({
            "speech": None,
            "thought": "考え中",
            "action": "首をかしげる",
        }, ensure_ascii=False)
        response.model = "test-model"
        response.usage = {}
        llm.generate = AsyncMock(return_value=response)
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        result = await engine.process_message(_make_message_input())

        assert result.content == ""
        assert result.thought == "考え中"
        assert result.action == "首をかしげる"


# --- Middleware integration (replaces ConversationLogger direct integration) ---


class TestRuntimeEngineMiddleware:
    """RuntimeEngine とミドルウェアの統合テスト."""

    @pytest.mark.asyncio
    async def test_middleware_post_process_called(self) -> None:
        """process_message でミドルウェアの post_process が呼ばれる."""
        from pneuma_core.runtime.middleware import PipelineContext

        storage, llm, embedding, memory_store = _setup_mocks()
        llm.generate = AsyncMock(return_value=_make_llm_response("やあ！"))

        post_called = False

        class LogMiddleware:
            async def pre_process(self, msg, context):
                return msg

            async def post_process(self, msg, output, context):
                nonlocal post_called
                post_called = True
                return output

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            middlewares=[LogMiddleware()],
        )

        msg = _make_message_input(content="こんにちは！")
        result = await engine.process_message(msg)

        assert post_called
        assert result.content == "やあ！"

    @pytest.mark.asyncio
    async def test_no_middleware_no_error(self) -> None:
        """ミドルウェアなしでもエラーなく処理される."""
        storage, llm, embedding, memory_store = _setup_mocks()
        llm.generate = AsyncMock(return_value=_make_llm_response("やあ！"))

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )

        msg = _make_message_input(content="こんにちは！")
        result = await engine.process_message(msg)

        assert result.content == "やあ！"

    @pytest.mark.asyncio
    async def test_middleware_error_does_not_break_response(self) -> None:
        """ミドルウェアでエラーが発生しても応答は返る."""
        storage, llm, embedding, memory_store = _setup_mocks()
        llm.generate = AsyncMock(return_value=_make_llm_response("やあ！"))

        class BrokenMiddleware:
            async def pre_process(self, msg, context):
                return msg

            async def post_process(self, msg, output, context):
                raise OSError("disk full")

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            middlewares=[BrokenMiddleware()],
        )

        msg = _make_message_input(content="こんにちは！")
        result = await engine.process_message(msg)

        assert result.content == "やあ！"



