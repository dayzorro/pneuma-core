"""Tests for HistorySummarizer (Issue #118).

HistorySummarizer is responsible for:
1. Detecting when conversation history exceeds the limit (30 messages)
2. Summarizing overflow messages using LLM (Haiku)
3. Prepending the summary to conversation history
4. Integrating with RuntimeEngine._trim_history()
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from pneuma_core.llm.adapter import LLMRequest, LLMResponse


# ============================================================
# HistorySummarizer unit tests
# ============================================================


class TestHistorySummarizerInit:
    """HistorySummarizer の初期化テスト."""

    def test_create_with_llm(self) -> None:
        """LLMAdapter を渡して生成できる."""
        from pneuma_core.runtime.history_summarizer import HistorySummarizer

        llm = AsyncMock()
        summarizer = HistorySummarizer(llm=llm)
        assert summarizer is not None

    def test_initial_summary_is_none(self) -> None:
        """初期状態では要約は None."""
        from pneuma_core.runtime.history_summarizer import HistorySummarizer

        llm = AsyncMock()
        summarizer = HistorySummarizer(llm=llm)
        assert summarizer.current_summary is None


class TestHistorySummarizerShouldSummarize:
    """要約が必要かどうかの判定テスト."""

    def test_below_limit_no_summarize(self) -> None:
        """30メッセージ以下では要約不要."""
        from pneuma_core.runtime.history_summarizer import HistorySummarizer

        llm = AsyncMock()
        summarizer = HistorySummarizer(llm=llm)
        history = [{"role": "user", "content": f"msg{i}"} for i in range(20)]
        assert summarizer.should_summarize(history, limit=30) is False

    def test_at_limit_no_summarize(self) -> None:
        """ちょうど30メッセージでは要約不要."""
        from pneuma_core.runtime.history_summarizer import HistorySummarizer

        llm = AsyncMock()
        summarizer = HistorySummarizer(llm=llm)
        history = [{"role": "user", "content": f"msg{i}"} for i in range(30)]
        assert summarizer.should_summarize(history, limit=30) is False

    def test_above_limit_needs_summarize(self) -> None:
        """30メッセージを超えたら要約が必要."""
        from pneuma_core.runtime.history_summarizer import HistorySummarizer

        llm = AsyncMock()
        summarizer = HistorySummarizer(llm=llm)
        history = [{"role": "user", "content": f"msg{i}"} for i in range(35)]
        assert summarizer.should_summarize(history, limit=30) is True


class TestHistorySummarizerSummarize:
    """要約生成のテスト."""

    @pytest.mark.asyncio
    async def test_summarize_overflow_messages(self) -> None:
        """超過分のメッセージを要約してテキストを返す."""
        from pneuma_core.runtime.history_summarizer import HistorySummarizer

        llm = AsyncMock()
        llm.generate = AsyncMock(return_value=LLMResponse(
            content="ユーザーとの挨拶の会話があった。",
            model="claude-haiku-4-5-20251001",
            usage={},
        ))
        summarizer = HistorySummarizer(llm=llm)

        history = [
            {"role": "user", "content": f"msg{i}"}
            for i in range(40)
        ]

        summary = await summarizer.summarize(history, limit=30)

        assert summary is not None
        assert "ユーザーとの挨拶の会話があった" in summary
        # LLM が呼ばれたことを確認
        llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_uses_haiku_model(self) -> None:
        """要約生成に Haiku モデルを使用する."""
        from pneuma_core.runtime.history_summarizer import HistorySummarizer

        llm = AsyncMock()
        llm.generate = AsyncMock(return_value=LLMResponse(
            content="要約テキスト",
            model="claude-haiku-4-5-20251001",
            usage={},
        ))
        summarizer = HistorySummarizer(llm=llm)

        history = [{"role": "user", "content": f"msg{i}"} for i in range(35)]
        await summarizer.summarize(history, limit=30)

        call_args = llm.generate.call_args[0][0]
        assert isinstance(call_args, LLMRequest)
        assert call_args.model == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_summarize_sends_overflow_messages(self) -> None:
        """超過分のメッセージだけを要約対象にする."""
        from pneuma_core.runtime.history_summarizer import HistorySummarizer

        llm = AsyncMock()
        llm.generate = AsyncMock(return_value=LLMResponse(
            content="要約",
            model="claude-haiku-4-5-20251001",
            usage={},
        ))
        summarizer = HistorySummarizer(llm=llm)

        history = [
            {"role": "user", "content": f"msg{i}"}
            for i in range(35)
        ]

        await summarizer.summarize(history, limit=30)

        # LLMリクエストのメッセージに超過分（先頭5件）の内容が含まれること
        call_args = llm.generate.call_args[0][0]
        assert isinstance(call_args, LLMRequest)
        # プロンプト内に超過メッセージの内容が含まれていること
        prompt_content = call_args.system_prompt + str(call_args.messages)
        assert "msg0" in prompt_content
        assert "msg4" in prompt_content

    @pytest.mark.asyncio
    async def test_summarize_stores_result(self) -> None:
        """要約結果が current_summary に保存される."""
        from pneuma_core.runtime.history_summarizer import HistorySummarizer

        llm = AsyncMock()
        llm.generate = AsyncMock(return_value=LLMResponse(
            content="保存される要約",
            model="claude-haiku-4-5-20251001",
            usage={},
        ))
        summarizer = HistorySummarizer(llm=llm)

        history = [{"role": "user", "content": f"msg{i}"} for i in range(35)]
        await summarizer.summarize(history, limit=30)

        assert summarizer.current_summary is not None
        assert "保存される要約" in summarizer.current_summary

    @pytest.mark.asyncio
    async def test_summarize_accumulates_with_previous(self) -> None:
        """既存の要約がある場合、新しい要約に既存の要約が含まれる."""
        from pneuma_core.runtime.history_summarizer import HistorySummarizer

        llm = AsyncMock()
        llm.generate = AsyncMock(return_value=LLMResponse(
            content="累積された要約テキスト",
            model="claude-haiku-4-5-20251001",
            usage={},
        ))
        summarizer = HistorySummarizer(llm=llm)
        # 事前に要約を設定
        summarizer._current_summary = "前回の要約: 天気の話をした"

        history = [{"role": "user", "content": f"msg{i}"} for i in range(35)]
        await summarizer.summarize(history, limit=30)

        # LLMリクエストに前回の要約が含まれていること
        call_args = llm.generate.call_args[0][0]
        prompt_content = call_args.system_prompt + str(call_args.messages)
        assert "前回の要約" in prompt_content or "天気の話をした" in prompt_content

    @pytest.mark.asyncio
    async def test_summarize_prompt_is_japanese(self) -> None:
        """要約プロンプトが日本語である."""
        from pneuma_core.runtime.history_summarizer import HistorySummarizer

        llm = AsyncMock()
        llm.generate = AsyncMock(return_value=LLMResponse(
            content="要約",
            model="claude-haiku-4-5-20251001",
            usage={},
        ))
        summarizer = HistorySummarizer(llm=llm)

        history = [{"role": "user", "content": f"msg{i}"} for i in range(35)]
        await summarizer.summarize(history, limit=30)

        call_args = llm.generate.call_args[0][0]
        # 中国語（簡体字）のプロンプトが含まれていること
        assert any(
            keyword in call_args.system_prompt
            for keyword in ["摘要", "对话", "总结"]
        )

    @pytest.mark.asyncio
    async def test_summarize_handles_llm_failure(self) -> None:
        """LLM呼び出しが失敗しても例外を投げない."""
        from pneuma_core.runtime.history_summarizer import HistorySummarizer

        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=Exception("LLM error"))
        summarizer = HistorySummarizer(llm=llm)

        history = [{"role": "user", "content": f"msg{i}"} for i in range(35)]
        # 例外が発生しないことを確認
        summary = await summarizer.summarize(history, limit=30)
        assert summary is None


class TestHistorySummarizerTrimWithSummary:
    """trim_with_summary: 要約付きのトリミングテスト."""

    @pytest.mark.asyncio
    async def test_trim_returns_summary_and_recent(self) -> None:
        """要約テキストと直近メッセージが返される（system ロールは含まない）."""
        from pneuma_core.runtime.history_summarizer import HistorySummarizer

        llm = AsyncMock()
        llm.generate = AsyncMock(return_value=LLMResponse(
            content="これまでの会話のまとめ",
            model="claude-haiku-4-5-20251001",
            usage={},
        ))
        summarizer = HistorySummarizer(llm=llm)

        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
            for i in range(40)
        ]

        result = await summarizer.trim_with_summary(history, limit=30)

        # 結果: (summary_text, recent_messages)
        summary_text, recent_messages = result
        assert summary_text is not None
        assert "これまでの会話のまとめ" in summary_text
        assert len(recent_messages) == 30
        # messages に system ロールが含まれないこと
        for m in recent_messages:
            assert m["role"] != "system", (
                f"messages に system ロールが含まれている: {m}"
            )
        # 直近のメッセージが含まれている
        assert recent_messages[-1]["content"] == "msg39"

    @pytest.mark.asyncio
    async def test_trim_below_limit_returns_unchanged(self) -> None:
        """limit以下なら要約なし・元の履歴をそのまま返す."""
        from pneuma_core.runtime.history_summarizer import HistorySummarizer

        llm = AsyncMock()
        summarizer = HistorySummarizer(llm=llm)

        history = [
            {"role": "user", "content": f"msg{i}"}
            for i in range(20)
        ]

        summary_text, recent_messages = await summarizer.trim_with_summary(history, limit=30)

        assert summary_text is None
        assert recent_messages == history
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_trim_with_existing_summary(self) -> None:
        """既存の要約がある場合、それも含めて再要約される."""
        from pneuma_core.runtime.history_summarizer import HistorySummarizer

        llm = AsyncMock()
        llm.generate = AsyncMock(return_value=LLMResponse(
            content="新しい累積要約",
            model="claude-haiku-4-5-20251001",
            usage={},
        ))
        summarizer = HistorySummarizer(llm=llm)
        summarizer._current_summary = "以前の要約内容"

        history = [
            {"role": "user", "content": f"msg{i}"}
            for i in range(35)
        ]

        summary_text, recent_messages = await summarizer.trim_with_summary(history, limit=30)

        assert summary_text is not None
        assert "新しい累積要約" in summary_text
        # messages に system ロールが含まれないこと
        for m in recent_messages:
            assert m["role"] != "system"

    @pytest.mark.asyncio
    async def test_trim_fallback_on_llm_failure(self) -> None:
        """LLM失敗時は単純なトリミング（末尾limit件）にフォールバック."""
        from pneuma_core.runtime.history_summarizer import HistorySummarizer

        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=Exception("LLM error"))
        summarizer = HistorySummarizer(llm=llm)

        history = [
            {"role": "user", "content": f"msg{i}"}
            for i in range(40)
        ]

        summary_text, recent_messages = await summarizer.trim_with_summary(history, limit=30)

        # フォールバック: 要約なし、単純切り捨てで末尾30件
        assert summary_text is None
        assert len(recent_messages) == 30
        assert recent_messages[-1]["content"] == "msg39"
        assert recent_messages[0]["content"] == "msg10"


# ============================================================
# RuntimeEngine integration with HistorySummarizer
# ============================================================


def _make_character():
    """テスト用キャラクター."""
    from pneuma_core.models.character import Character
    from pneuma_core.models.personality import Personality
    from pneuma_core.models.values import Values

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


def _make_emotional_state():
    """テスト用感情状態."""
    from pneuma_core.models.emotion import EmotionalState

    return EmotionalState(
        pleasure=0.3,
        arousal=0.1,
        dominance=0.0,
        emotion_label="安らぎ",
        situation="穏やかな会話中",
    )


def _make_message_input(content: str = "こんにちは！"):
    """テスト用メッセージ入力."""
    from pneuma_core.models.message import MessageInput

    return MessageInput(
        content=content,
        sender_id="user-1",
        sender_name="ユーザー",
        sender_type="human",
    )


def _make_llm_response(content: str = "やあ！元気？"):
    """テスト用LLMレスポンス."""
    return LLMResponse(
        content=content,
        model="test-model",
        usage={},
    )


def _setup_engine_mocks():
    """RuntimeEngine 用のモック依存関係."""
    from pneuma_core.models.goals import GoalTree

    storage = AsyncMock()
    storage.get_character = AsyncMock(return_value=_make_character())
    storage.get_emotional_state = AsyncMock(return_value=_make_emotional_state())
    storage.get_goals = AsyncMock(return_value=GoalTree())
    storage.save_emotional_state = AsyncMock()
    storage.save_change = AsyncMock()

    llm = AsyncMock()

    embedding_service = AsyncMock()
    embedding_service.embed = AsyncMock(return_value=[0.1] * 10)

    memory_store = AsyncMock()
    memory_store.get_episodic_by_character = AsyncMock(return_value=[])
    memory_store.get_semantic_by_character = AsyncMock(return_value=[])

    return storage, llm, embedding_service, memory_store


class TestRuntimeEngineDefaultHistoryLimit:
    """RuntimeEngine のデフォルト history_limit が 30 に変更されたことのテスト."""

    def test_default_history_limit_is_30(self) -> None:
        """デフォルトの history_limit が 30 である."""
        from pneuma_core.runtime.engine import RuntimeEngine

        storage, llm, embedding, memory_store = _setup_engine_mocks()
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )
        assert engine._history_limit == 30

    def test_explicit_history_limit_override(self) -> None:
        """明示的に指定した history_limit が使われる."""
        from pneuma_core.runtime.engine import RuntimeEngine

        storage, llm, embedding, memory_store = _setup_engine_mocks()
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            history_limit=50,
        )
        assert engine._history_limit == 50


class TestRuntimeEngineHistorySummarizerIntegration:
    """RuntimeEngine と HistorySummarizer の統合テスト."""

    @pytest.mark.asyncio
    async def test_engine_has_conversation_summary(self) -> None:
        """RuntimeEngine が _conversation_summary 属性を保持している."""
        from pneuma_core.runtime.engine import RuntimeEngine

        storage, llm, embedding, memory_store = _setup_engine_mocks()
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )
        assert hasattr(engine, "_conversation_summary")
        assert engine._conversation_summary is None

    @pytest.mark.asyncio
    async def test_trim_history_uses_summarizer_when_over_limit(self) -> None:
        """履歴が上限を超えた時、HistorySummarizer が使われる."""
        from pneuma_core.runtime.engine import RuntimeEngine

        storage, llm, embedding, memory_store = _setup_engine_mocks()

        # 各メッセージ処理ごとに応答を返す
        responses = []
        for i in range(20):
            responses.append(_make_llm_response(f"応答{i}"))
        llm.generate = AsyncMock(side_effect=responses)

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            history_limit=10,
        )

        # 20メッセージ送信（user + assistant = 各20 = 40メッセージ、limit=10を超える）
        for i in range(20):
            await engine.process_message(
                _make_message_input(content=f"msg{i}")
            )

        # 履歴が limit を大幅に超えないことを確認
        # (要約メッセージ + 直近メッセージ で limit+1 程度)
        assert len(engine._history) <= 11  # limit + 1 (summary)

    @pytest.mark.asyncio
    async def test_summary_appears_in_history_head(self) -> None:
        """要約メッセージが履歴の先頭に含まれる."""
        from pneuma_core.runtime.engine import RuntimeEngine
        from pneuma_core.runtime.history_summarizer import HistorySummarizer

        storage, llm, embedding, memory_store = _setup_engine_mocks()

        # 最初の応答 + 要約生成の応答を用意
        call_count = 0

        async def mock_generate(request):
            nonlocal call_count
            call_count += 1
            # 要約生成のリクエストかどうかをモデル名で判定
            if request.model == "claude-haiku-4-5-20251001":
                return _make_llm_response("これは会話の要約です")
            return _make_llm_response(f"応答{call_count}")

        llm.generate = AsyncMock(side_effect=mock_generate)

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            history_limit=10,
        )

        # limit を超えるメッセージを送信
        for i in range(15):
            await engine.process_message(
                _make_message_input(content=f"msg{i}")
            )

        # 要約が _conversation_summary に保存されていること（履歴には system ロールを入れない）
        assert engine._conversation_summary is not None
        # 履歴に system ロールが含まれないこと
        for m in engine._history:
            assert m["role"] != "system", (
                f"_history に system ロールが含まれている: {m}"
            )


class TestRuntimeEnginePipelineWithSummarizer:
    """パイプラインテスト: メッセージ処理→要約生成→履歴管理の全フロー."""

    @pytest.mark.asyncio
    async def test_long_conversation_maintains_context(self) -> None:
        """長い会話でも要約により文脈が維持される."""
        from pneuma_core.runtime.engine import RuntimeEngine

        storage, llm, embedding, memory_store = _setup_engine_mocks()

        async def mock_generate(request):
            if request.model == "claude-haiku-4-5-20251001":
                return _make_llm_response("前半: ユーザーと天気の話をした。名前はキョウヘイ。")
            return _make_llm_response("応答です")

        llm.generate = AsyncMock(side_effect=mock_generate)

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            history_limit=10,
        )

        # 20ターンの会話を実行
        for i in range(20):
            await engine.process_message(
                _make_message_input(content=f"msg{i}")
            )

        # 履歴がコンパクトに保たれていること
        assert len(engine._history) <= 11

        # 要約が _conversation_summary に保存されていること
        assert engine._conversation_summary is not None
        # 履歴に system ロールが含まれないこと
        for m in engine._history:
            assert m["role"] != "system", (
                f"_history に system ロールが含まれている: {m}"
            )

    @pytest.mark.asyncio
    async def test_summarizer_called_with_overflow_only(self) -> None:
        """要約は超過分のメッセージのみを対象にする（直近の limit 件は保持）."""
        from pneuma_core.runtime.engine import RuntimeEngine
        from pneuma_core.runtime.history_summarizer import HistorySummarizer

        storage, llm, embedding, memory_store = _setup_engine_mocks()

        summarize_calls = []
        original_summarize = None

        async def mock_generate(request):
            if request.model == "claude-haiku-4-5-20251001":
                # 要約リクエストの内容を記録
                summarize_calls.append(request)
                return _make_llm_response("要約テキスト")
            return _make_llm_response("応答")

        llm.generate = AsyncMock(side_effect=mock_generate)

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            history_limit=10,
        )

        # limit を超えるメッセージを送信
        for i in range(15):
            await engine.process_message(
                _make_message_input(content=f"msg{i}")
            )

        # 要約が少なくとも1回呼ばれていること
        assert len(summarize_calls) >= 1
