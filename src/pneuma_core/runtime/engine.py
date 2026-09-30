"""RuntimeEngine: unified message processing pipeline."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from pneuma_core.exceptions import LLMTimeoutError
from pneuma_core.llm.adapter import LLMAdapter, LLMRequest
from pneuma_core.llm.embedding import EmbeddingService
from pneuma_core.memory.search import MemorySearchEngine
from pneuma_core.memory.store import MemoryStore
from pneuma_core.models.change_record import ChangeRecord
from pneuma_core.models.character import Character
from pneuma_core.models.diagnostic import DiagnosticInfo
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.models.message import (
    MessageInput,
    MessageOutput,
    StructuredResponse,
    SystemMessage,
)
from pneuma_core.models.personality import Personality
from pneuma_core.runtime.emotion_engine import NEUTRAL_EMOTION, EmotionEngine
from pneuma_core.runtime.prompt_builder import PromptBuilder
from pneuma_core.runtime.prompt_cache import CachedPrompt, PromptCache
from pneuma_core.runtime.response_parser import (
    extract_partial_speech,
    parse_structured_response,
)
from pneuma_core.runtime.user_context import UserContext
from pneuma_core.runtime.user_context_search import (
    UserContextSearchEngine,
    UserContextSearchResult,
)
from pneuma_core.runtime.middleware import Middleware, PipelineContext
from pneuma_core.storage.backend import StorageBackend

logger = logging.getLogger(__name__)

# 流式增量回调：收到一段（已解码的）speech 文本时被 await
OnDelta = Callable[[str], Awaitable[None]]


class RuntimeEngine:
    """Unified message processing pipeline for a character.

    Pipeline:
        1. Load character state (character, emotion, goals)
        2. Search relevant memories
        3. Build system prompt (PromptCache or PromptBuilder)
        4. Generate LLM response
        5. Evaluate emotion via direct LLM estimation (EmotionEngine)
        6. Return MessageOutput immediately with current emotion
        7. Background task saves updated emotion for next turn

    Note: Episodic/semantic memory saving and todo extraction are handled by
    SessionEndPipeline at session end, not per-turn.
    """

    def __init__(
        self,
        character_id: str,
        storage: StorageBackend,
        llm: LLMAdapter,
        embedding_service: EmbeddingService,
        memory_store: MemoryStore,
        history_limit: int = 30,
        prompt_cache: PromptCache | None = None,
        response_model: str | None = None,
        emotion_model: str | None = None,
        diagnostic_mode: bool = False,
        user_context: UserContext | None = None,
        user_context_search_engine: UserContextSearchEngine | None = None,
        user_goal_tree: GoalTree | None = None,
        character_relations: list | None = None,
        middlewares: list[Middleware] | None = None,
    ) -> None:
        self._character_id = character_id
        self._storage = storage
        self._llm = llm
        self._embedding_service = embedding_service
        self._memory_store = memory_store
        self._history_limit = history_limit
        self._response_model = response_model

        self._prompt_cache = prompt_cache
        self._prompt_builder = PromptBuilder()
        self._emotion_engine = EmotionEngine(llm=llm, model=emotion_model)
        self._memory_search = MemorySearchEngine()

        self._diagnostic_mode = diagnostic_mode

        # User context
        self._user_context = user_context
        self._user_context_search_engine = user_context_search_engine

        # User goal tree (optional)
        self._user_goal_tree = user_goal_tree

        # Character relations (optional)
        self._character_relations = character_relations

        # Middleware chain
        self._middlewares: list[Middleware] = middlewares or []

        self._history: list[dict] = []
        self._conversation_summary: str | None = None
        self._pending_emotion_task: asyncio.Task | None = None
        self._latest_emotion: EmotionalState | None = None
        self._turn_count: int = 0

        self._last_emotion_update: datetime | None = None

    async def process_message(
        self, msg: MessageInput, on_delta: OnDelta | None = None
    ) -> MessageOutput:
        """Process an incoming message and return a response.

        Args:
            msg: The incoming message.
            on_delta: 可选。若提供且适配器支持流式，则在生成过程中
                以增量文本回调（用于 SSE 流式输出）。此时返回的
                MessageOutput 与一次性生成完全一致。
        """
        now = datetime.now(timezone.utc)
        changes: list[ChangeRecord] = []
        system_messages: list[SystemMessage] = []

        # Increment turn count
        self._turn_count += 1

        # 1. Load character state
        character = await self._storage.get_character(self._character_id)
        if character is None:
            raise ValueError(f"Character not found: {self._character_id}")

        current_emotion = await self._storage.get_emotional_state(
            self._character_id
        )
        if current_emotion is None:
            current_emotion = NEUTRAL_EMOTION

        # Use latest async emotion if available
        if self._latest_emotion is not None:
            current_emotion = self._latest_emotion
            self._latest_emotion = None

        # 1.5. Apply emotion decay towards personality baseline
        _DECAY_MIN_ELAPSED = 1.0
        if self._last_emotion_update is not None:
            elapsed = (now - self._last_emotion_update).total_seconds()
            if elapsed >= _DECAY_MIN_ELAPSED:
                current_emotion = self._emotion_engine.decay_towards_baseline(
                    state=current_emotion,
                    personality=character.personality,
                    elapsed_seconds=elapsed,
                )

        goals = await self._storage.get_goals(self._character_id)
        if goals is None:
            goals = GoalTree()

        # 2. Search relevant memories
        memories = []
        try:
            query_embedding = await self._embedding_service.embed(msg.content)
            episodic = await self._memory_store.get_episodic_by_character(
                self._character_id
            )
            semantic = await self._memory_store.get_semantic_by_character(
                self._character_id
            )
            all_memories = list(episodic) + list(semantic)
            if all_memories and query_embedding:
                scored = self._memory_search.search(
                    memories=all_memories,
                    query_embedding=query_embedding,
                    personality=character.personality,
                    now=now,
                )
                memories = [m for m, _score in scored]
        except Exception:
            logger.warning("Memory search failed, continuing without memories")
            system_messages.append(SystemMessage(
                type="warning",
                message="记忆检索失败，本轮将在没有记忆的情况下继续对话",
                component="memory_search",
            ))

        # 2.5. Search user context (RAG Tier 3) if available
        user_context_search_results: list[UserContextSearchResult] = []
        if self._user_context_search_engine is not None:
            try:
                user_context_search_results = (
                    await self._user_context_search_engine.search(msg.content)
                )
            except Exception:
                logger.warning(
                    "User context search failed, continuing without results"
                )
                system_messages.append(SystemMessage(
                    type="warning",
                    message="用户上下文检索失败，本轮将在没有其结果的情况下继续对话",
                    component="user_context_search",
                ))

        # 3. Build system prompt (use PromptCache if available)
        prompt_result = self._build_system_prompt(
            character, current_emotion, goals, memories,
            user_context_search_results=user_context_search_results,
        )
        # Extract system_prompt and optional cached/dynamic sections
        if isinstance(prompt_result, CachedPrompt):
            system_prompt = prompt_result.full_prompt
            _cached_section = prompt_result.static_section
            _dynamic_section = prompt_result.dynamic_section
        else:
            system_prompt = prompt_result
            _cached_section = None
            _dynamic_section = None

        # 3.5. Run middleware pre_process chain
        pipeline_context = PipelineContext(
            character=character,
            emotion=current_emotion,
            goals=goals,
            memories=memories,
            system_prompt=system_prompt,
            history=list(self._history),
            turn_count=self._turn_count,
            metadata={},
        )
        for mw in self._middlewares:
            try:
                msg = await mw.pre_process(msg, pipeline_context)
            except Exception:
                logger.warning(
                    "Middleware %s pre_process failed, continuing",
                    type(mw).__name__,
                )

        # 4. Add user message to history
        self._history.append({
            "role": "user",
            "content": f"[{msg.sender_name}] {msg.content}",
        })
        await self._trim_history()

        # 5. Generate LLM response
        # Append conversation summary to system_prompt (not in messages)
        effective_system_prompt = system_prompt
        if self._conversation_summary is not None:
            effective_system_prompt = (
                f"{system_prompt}\n\n{self._conversation_summary}"
            )

        llm_succeeded = False
        response = None
        structured: StructuredResponse | None = None
        llm_request = LLMRequest(
            system_prompt=effective_system_prompt,
            messages=self._build_messages_for_llm(),
            model=self._response_model,
            system_prompt_cached=_cached_section,
            system_prompt_dynamic=_dynamic_section,
            # 每轮回复在交互关键路径上，关闭思考以降低延迟
            enable_thinking=False,
        )
        use_stream = (
            on_delta is not None and hasattr(self._llm, "generate_stream")
        )
        try:
            if use_stream:
                response_text = await self._generate_streaming(
                    llm_request, on_delta
                )
            else:
                response = await self._llm.generate(llm_request)
                response_text = response.content
            llm_succeeded = True

            # Parse structured response (speech/thought/action)
            structured = parse_structured_response(response_text)
            speech_text = structured.speech if structured.speech is not None else ""
        except Exception as e:
            if isinstance(e, LLMTimeoutError):
                raise
            logger.warning("LLM generate failed, using fallback response: %s: %s", type(e).__name__, e)
            response_text = "抱歉，我一时没能组织好语言。"
            speech_text = response_text
            structured = StructuredResponse(speech=response_text)
            system_messages.append(SystemMessage(
                type="error",
                message="LLM 调用失败，已改用兜底回复",
                component="llm",
            ))

        # Add assistant response to history (speech only, not full JSON)
        self._history.append({
            "role": "assistant",
            "content": speech_text,
        })
        await self._trim_history()

        # 6. Record emotion change with trigger type
        emotion_dict = self._emotion_to_dict(current_emotion)
        emotion_dict["trigger_type"] = "pending"
        emotion_change = ChangeRecord(
            id=str(uuid.uuid4()),
            character_id=self._character_id,
            type="emotion_updated",
            before=self._emotion_to_dict(current_emotion),
            after=emotion_dict,
            reason="emotion estimation after message",
            timestamp=now,
        )
        changes.append(emotion_change)

        # 7. Save change records
        for change in changes:
            await self._storage.save_change(change)

        # 8. Emotion evaluation
        diagnostic: DiagnosticInfo | None = None
        if self._diagnostic_mode and llm_succeeded:
            # Diagnostic mode: run emotion evaluation synchronously
            trigger_type = "error"
            trigger_reasons: list[str] = []
            try:
                emotion_result = await self._emotion_engine.evaluate(
                    personality=character.personality,
                    messages=list(self._history),
                    turn_count=self._turn_count,
                    current_state=current_emotion,
                )
                trigger_type = emotion_result.trigger_type
                trigger_reasons = list(emotion_result.reasons)
                if emotion_result.trigger_type != "skipped":
                    await self._storage.save_emotional_state(
                        self._character_id, emotion_result.state
                    )
            except Exception:
                logger.warning(
                    "Diagnostic emotion evaluation failed, continuing"
                )
            diagnostic = DiagnosticInfo(
                personality={
                    "openness": character.personality.openness,
                    "conscientiousness": character.personality.conscientiousness,
                    "extraversion": character.personality.extraversion,
                    "agreeableness": character.personality.agreeableness,
                    "neuroticism": character.personality.neuroticism,
                },
                emotion={
                    "pleasure": current_emotion.pleasure,
                    "arousal": current_emotion.arousal,
                    "dominance": current_emotion.dominance,
                    "label": current_emotion.emotion_label,
                },
                emotion_trigger={
                    "type": trigger_type,
                    "reasons": trigger_reasons,
                },
                memories_retrieved=[m.content for m in memories],
                token_usage=response.usage if response else {},
                model=response.model if response else "",
            )
        elif llm_succeeded:
            # Normal mode: launch async emotion evaluation (non-blocking).
            # 若上一轮评估仍在进行则不重复启动，避免堆积；
            # 其结果会在后续轮次通过 _latest_emotion 生效，绝不阻塞本轮回复。
            if (
                self._pending_emotion_task is None
                or self._pending_emotion_task.done()
            ):
                self._pending_emotion_task = asyncio.create_task(
                    self._async_emotion_evaluation(
                        personality=character.personality,
                        messages=list(self._history),
                        current_emotion=current_emotion,
                    )
                )

        # Note: Episodic/semantic memory saving moved to SessionEndPipeline (#130)

        # Update last emotion update timestamp
        self._last_emotion_update = now

        output = MessageOutput(
            content=speech_text,
            emotion=current_emotion,
            thought=structured.thought if structured else None,
            action=structured.action if structured else None,
            internal_changes=changes,
            diagnostic=diagnostic,
            system_messages=system_messages,
        )

        # Run middleware post_process chain (reverse order for proper nesting)
        for mw in reversed(self._middlewares):
            try:
                output = await mw.post_process(msg, output, pipeline_context)
            except Exception:
                logger.warning(
                    "Middleware %s post_process failed, continuing",
                    type(mw).__name__,
                )

        return output

    def _build_system_prompt(
        self,
        character: Character,
        emotional_state: EmotionalState,
        goal_tree: GoalTree,
        memories: list[EpisodicMemory | SemanticMemory],
        user_context_search_results: list[UserContextSearchResult] | None = None,
    ) -> str | CachedPrompt:
        """Build system prompt using PromptCache or PromptBuilder.

        Returns CachedPrompt when PromptCache is available (enables API-level
        prompt caching), or a plain string when using PromptBuilder directly.
        """
        if self._prompt_cache is not None:
            return self._prompt_cache.build(
                character=character,
                emotional_state=emotional_state,
                goal_tree=goal_tree,
                memories=memories,
                user_context=self._user_context,
                user_context_search_results=user_context_search_results,
                user_goal_tree=self._user_goal_tree,
                character_relations=self._character_relations,
            )
        return self._prompt_builder.build(
            character=character,
            emotional_state=emotional_state,
            goal_tree=goal_tree,
            memories=memories,
            user_context=self._user_context,
            user_context_search_results=user_context_search_results,
            user_goal_tree=self._user_goal_tree,
            character_relations=self._character_relations,
        )

    @staticmethod
    def _emotion_to_dict(emotion: EmotionalState) -> dict:
        """Convert EmotionalState to a dictionary for ChangeRecord."""
        return {
            "pleasure": emotion.pleasure,
            "arousal": emotion.arousal,
            "dominance": emotion.dominance,
            "emotion_label": emotion.emotion_label,
            "situation": emotion.situation,
        }

    async def _async_emotion_evaluation(
        self,
        personality: Personality,
        messages: list[dict],
        current_emotion: EmotionalState,
    ) -> None:
        """Run emotion evaluation via direct LLM estimation in the background."""
        try:
            result = await self._emotion_engine.evaluate(
                personality=personality,
                messages=messages,
                turn_count=self._turn_count,
                current_state=current_emotion,
            )
            self._latest_emotion = result.state
            if result.trigger_type != "skipped":
                await self._storage.save_emotional_state(
                    self._character_id, result.state
                )
        except Exception:
            logger.warning("Async emotion evaluation failed, keeping current emotion")
            self._latest_emotion = current_emotion
            await self._storage.save_emotional_state(
                self._character_id, current_emotion
            )

    async def aclose(self) -> None:
        """等待仍在进行的情绪评估任务结束（应在会话结束时调用）。

        轮次之间**不会**等待该任务，因此它不会拖慢交互。
        """
        task = self._pending_emotion_task
        if task is not None and not task.done():
            try:
                await task
            except Exception:
                logger.warning("Pending emotion task failed")
        self._pending_emotion_task = None

    async def _generate_streaming(
        self, request: LLMRequest, on_delta: OnDelta
    ) -> str:
        """流式生成，边产生边通过 ``on_delta`` 推送 speech 增量。

        Returns:
            完整的原始输出（等价于一次性 generate 的 content），
            供后续结构化解析使用。
        """
        buffer: list[str] = []
        emitted = 0
        async for piece in self._llm.generate_stream(request):  # type: ignore[attr-defined]
            buffer.append(piece)
            partial = extract_partial_speech("".join(buffer))
            if len(partial) > emitted:
                await on_delta(partial[emitted:])
                emitted = len(partial)

        raw = "".join(buffer)
        # 纯文本兜底等情形下，最终 speech 可能仍多于已推送部分，补发差量
        final_speech = parse_structured_response(raw).speech or ""
        if len(final_speech) > emitted:
            await on_delta(final_speech[emitted:])
        return raw

    _SUMMARIZE_PROMPT = """\
请简洁地总结以下对话内容。总结中需要包含：
- 谈过的主要话题
- 重要的决定或约定
- 对话的情绪基调
- 尚未解决的问题

{previous_summary_section}

对话内容:
{conversation}

请用简体中文、以要点列表的形式输出总结。"""

    def _build_messages_for_llm(self) -> list[dict]:
        """Build message list for LLM (no system role in messages)."""
        return list(self._history)

    async def _summarize_old_messages(self, messages_to_summarize: list[dict]) -> None:
        """Summarize old messages using Haiku and store as conversation summary."""
        try:
            conversation_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in messages_to_summarize
            )

            previous_summary_section = ""
            if self._conversation_summary is not None:
                previous_summary_section = (
                f"上次的总结:\n{self._conversation_summary}\n\n"
                "请把上面的总结与以下新的对话合并后再总结一次。"
            )

            prompt = self._SUMMARIZE_PROMPT.format(
                previous_summary_section=previous_summary_section,
                conversation=conversation_text,
            )

            response = await self._llm.generate(
                LLMRequest(
                    system_prompt=prompt,
                    messages=[{"role": "user", "content": "请总结。"}],
                    model="claude-haiku-4-5-20251001",
                    temperature=0.0,
                    max_tokens=512,
                    # 摘要在历史裁剪时同步执行，关闭思考避免拖慢该轮
                    enable_thinking=False,
                )
            )

            summary_text = response.content.strip()
            self._conversation_summary = f"[此前的对话总结]\n{summary_text}"
        except Exception:
            logger.warning("History summarization failed, continuing with simple trim")

    async def _trim_history(self) -> None:
        """Trim conversation history to limit, summarizing old messages."""
        if len(self._history) > self._history_limit:
            # Calculate how many messages to remove
            overflow = len(self._history) - self._history_limit
            messages_to_summarize = self._history[:overflow]

            # Summarize the old messages
            await self._summarize_old_messages(messages_to_summarize)

            # Keep only the most recent messages within limit
            self._history = self._history[-self._history_limit :]
