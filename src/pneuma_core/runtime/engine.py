"""RuntimeEngine: unified message processing pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from pneuma_core.exceptions import LLMTimeoutError
from pneuma_core.knowledge.models import KnowledgeHit
from pneuma_core.knowledge.retriever import KnowledgeBase
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
from pneuma_core.runtime.latency import LatencyMetrics
from pneuma_core.runtime.prompt_builder import PromptBuilder
from pneuma_core.runtime.prompt_cache import CachedPrompt, PromptCache
from pneuma_core.runtime.response_parser import (
    extract_partial_speech,
    parse_structured_response,
    speech_field_complete,
)
from pneuma_core.runtime.user_context import UserContext
from pneuma_core.runtime.user_context_search import (
    UserContextSearchEngine,
    UserContextSearchResult,
)
from pneuma_core.runtime.middleware import Middleware, PipelineContext
from pneuma_core.runtime.insight_acquirer import InsightAcquirer
from pneuma_core.storage.backend import StorageBackend
from pneuma_core.websearch.models import WebSearchClient, WebSearchResponse
from pneuma_core.websearch.policy import DEFAULT_MODE as DEFAULT_WEB_SEARCH_MODE
from pneuma_core.websearch.policy import should_search_online

logger = logging.getLogger(__name__)

# 流式增量回调：收到一段（已解码的）speech 文本时被 await
OnDelta = Callable[[str], Awaitable[None]]

# 「台词已说完」回调：speech 字段生成完毕时立即触发，早于 thought/action，
# 用于让上层先把回复判为完成，不必等整段 JSON（含内心戏）生成完。
OnSpeechComplete = Callable[[dict], Awaitable[None]]


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
        knowledge_base: KnowledgeBase | None = None,
        web_search_client: WebSearchClient | None = None,
        web_search_mode: str = DEFAULT_WEB_SEARCH_MODE,
        insight_acquirer: InsightAcquirer | None = None,
        metrics: LatencyMetrics | None = None,
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

        # 本地知识库（可选）。用于「专业知识问答」：每轮检索后注入提示词。
        self._knowledge_base = knowledge_base

        # 联网检索（可选）：本地资料答不上、且问题指向外部实时信息时启用
        self._web_search_client = web_search_client
        self._web_search_mode = web_search_mode
        # 联网结果的异步提炼器（把实时信息沉淀成行业通识认知块）
        self._insight_acquirer = insight_acquirer

        # User goal tree (optional)
        self._user_goal_tree = user_goal_tree

        # Character relations (optional)
        self._character_relations = character_relations

        # 中间件链
        self._middlewares: list[Middleware] = middlewares or []

        # 时延埋点聚合器（可选）。为 None 时只采集不统计。
        self._metrics = metrics

        self._history: list[dict] = []
        self._conversation_summary: str | None = None
        self._pending_emotion_task: asyncio.Task | None = None
        # 历史摘要异步化：裁剪同步完成，摘要走后台。此前摘要在关键路径上
        # 同步 await，历史超限后每轮会触发两次 LLM 往返。
        self._pending_summary_messages: list[dict] = []
        self._summary_task: asyncio.Task | None = None
        # 后台的认知提炼任务（联网结果 → 行业认知块）
        self._insight_tasks: set[asyncio.Task] = set()
        self._latest_emotion: EmotionalState | None = None
        self._turn_count: int = 0

        self._last_emotion_update: datetime | None = None

    async def process_message(
        self,
        msg: MessageInput,
        on_delta: OnDelta | None = None,
        on_speech_complete: OnSpeechComplete | None = None,
        web_search_mode: str | None = None,
    ) -> MessageOutput:
        """Process an incoming message and return a response.

        Args:
            msg: The incoming message.
            on_delta: 可选。若提供且适配器支持流式，则在生成过程中
                以增量文本回调（用于 SSE 流式输出）。此时返回的
                MessageOutput 与一次性生成完全一致。
            on_speech_complete: 可选。speech 字段生成完毕时立即回调，
                用于让上层先把「回复」判为完成，而不必等 thought/action。
            web_search_mode: 可选。覆盖本轮的联网模式（``off`` 可确保
                本轮完全不触发外网调用）。
        """
        now = datetime.now(timezone.utc)
        changes: list[ChangeRecord] = []
        system_messages: list[SystemMessage] = []

        # 时延埋点：按阶段记录耗时（毫秒），轮末交给 _metrics 聚合
        t_start = time.perf_counter()
        timings: dict[str, float | None] = {}
        _last = t_start

        def _lap(stage: str) -> None:
            nonlocal _last
            current = time.perf_counter()
            timings[stage] = (current - _last) * 1000.0
            _last = current

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
        _lap("state")

        # 2. Query embedding —— 只算一次，供记忆 / 知识 / 用户上下文检索复用。
        #    此前每个检索器各自调用一次 embedding，同一 query 最多触发 3 次
        #    远程往返，全部串在首字之前。
        query_embedding: list[float] | None = None
        try:
            query_embedding = await self._embedding_service.embed(msg.content)
        except Exception:
            logger.warning(
                "Query embedding failed, continuing without vector search"
            )
            system_messages.append(SystemMessage(
                type="warning",
                message="向量化失败，本轮记忆与知识检索已降级",
                component="embedding",
            ))
        _lap("embedding")

        # 2.1 三类检索互不依赖，并发执行，把关键路径从「求和」压成「取最大」
        async def _search_memories() -> list:
            if not query_embedding:
                return []
            branch_start = time.perf_counter()
            try:
                episodic = await self._memory_store.get_episodic_by_character(
                    self._character_id
                )
                semantic = await self._memory_store.get_semantic_by_character(
                    self._character_id
                )
                all_memories = list(episodic) + list(semantic)
                if not all_memories:
                    return []
                scored = self._memory_search.search(
                    memories=all_memories,
                    query_embedding=query_embedding,
                    personality=character.personality,
                    now=now,
                )
                return [m for m, _score in scored]
            except Exception:
                logger.warning(
                    "Memory search failed, continuing without memories"
                )
                system_messages.append(SystemMessage(
                    type="warning",
                    message="记忆检索失败，本轮将在没有记忆的情况下继续对话",
                    component="memory_search",
                ))
                return []
            finally:
                timings["retrieve_memory"] = (
                    time.perf_counter() - branch_start
                ) * 1000.0

        async def _search_user_context() -> list[UserContextSearchResult]:
            if self._user_context_search_engine is None:
                return []
            branch_start = time.perf_counter()
            try:
                return await self._user_context_search_engine.search(msg.content)
            except Exception:
                logger.warning(
                    "User context search failed, continuing without results"
                )
                system_messages.append(SystemMessage(
                    type="warning",
                    message="用户上下文检索失败，本轮将在没有其结果的情况下继续对话",
                    component="user_context_search",
                ))
                return []
            finally:
                timings["retrieve_user_context"] = (
                    time.perf_counter() - branch_start
                ) * 1000.0

        async def _search_knowledge() -> list[KnowledgeHit]:
            if self._knowledge_base is None:
                return []
            branch_start = time.perf_counter()
            try:
                return await self._knowledge_base.search(
                    msg.content, query_embedding=query_embedding
                )
            except Exception:
                logger.warning(
                    "Knowledge search failed, continuing without knowledge"
                )
                system_messages.append(SystemMessage(
                    type="warning",
                    message="知识库检索失败，本轮将在没有资料的情况下继续对话",
                    component="knowledge_search",
                ))
                return []
            finally:
                timings["retrieve_knowledge"] = (
                    time.perf_counter() - branch_start
                ) * 1000.0

        (
            memories,
            user_context_search_results,
            knowledge_hits,
        ) = await asyncio.gather(
            _search_memories(),
            _search_user_context(),
            _search_knowledge(),
        )
        _lap("retrieve_wait")

        # 2.7. 联网检索：本地资料答不上、且问题指向外部实时信息时才走外网
        #      （依赖知识库结果，故在并发检索之后判定）
        web_response: WebSearchResponse | None = None
        effective_web_mode = web_search_mode or self._web_search_mode
        if self._web_search_client is not None and should_search_online(
            msg.content,
            mode=effective_web_mode,
            has_local_hits=bool(knowledge_hits),
        ):
            try:
                web_response = await self._web_search_client.search(msg.content)
            except Exception as e:
                logger.warning(
                    "Web search failed, continuing without live info: %s: %s",
                    type(e).__name__,
                    e,
                )
                system_messages.append(SystemMessage(
                    type="warning",
                    message="联网检索失败，本轮将不包含实时信息",
                    component="web_search",
                ))

            # 检索结果异步提炼成行业认知块，不占用本轮回复的时延
            if (
                web_response is not None
                and not web_response.is_empty
                and self._insight_acquirer is not None
            ):
                self._schedule_insight_acquisition(msg.content, web_response)
        _lap("web_search")

        # 2.8. speech 说完即回调：让上层先把「回复」判为完成，不被后面的
        #      thought / action 生成拖住（它们与前段是同一次生成的后半截）。
        async def _emit_speech_complete(speech: str) -> None:
            if on_speech_complete is None:
                return
            await on_speech_complete({
                "reply": speech,
                "emotion": {
                    "pleasure": current_emotion.pleasure,
                    "arousal": current_emotion.arousal,
                    "dominance": current_emotion.dominance,
                    "label": current_emotion.emotion_label,
                },
                "web_sources": (
                    web_response.sources() if web_response else []
                ),
            })

        # 3. Build system prompt (use PromptCache if available)
        prompt_result = self._build_system_prompt(
            character, current_emotion, goals, memories,
            user_context_search_results=user_context_search_results,
            knowledge_hits=knowledge_hits,
            web_search=web_response,
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
        _lap("prompt_build")

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
        _lap("preprocess")

        # 4. Add user message to history
        self._history.append({
            "role": "user",
            "content": f"[{msg.sender_name}] {msg.content}",
        })
        await self._trim_history()
        _lap("history")

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
        llm_start = time.perf_counter()
        _last = llm_start

        async def _timed_on_delta(text: str) -> None:
            """记录首字延迟（仅流式路径），并把增量转发给上层。"""
            if "llm_first_token" not in timings:
                timings["llm_first_token"] = (
                    time.perf_counter() - llm_start
                ) * 1000.0
            if on_delta is not None:
                await on_delta(text)

        try:
            if use_stream:
                response_text = await self._generate_streaming(
                    llm_request, _timed_on_delta, _emit_speech_complete
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
        finally:
            _lap("llm_total")

        # Add assistant response to history.
        # 存规范化的结构化 JSON（而非 speech 纯文本）：否则上下文里会累积
        # 一批「纯文本助手示例」，与 system prompt 要求的 JSON 格式不一致，
        # 模型会在几轮后被带偏、不再输出 thought / action。
        self._history.append({
            "role": "assistant",
            "content": self._history_assistant_content(structured, speech_text),
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
        _lap("finalize")

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
                    expressiveness=character.emotional_expressiveness,
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
                        expressiveness=character.emotional_expressiveness,
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
            web_sources=web_response.sources() if web_response else [],
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
        _lap("post_process")

        # 时延埋点：整轮耗时与分阶段样本
        timings["total"] = (time.perf_counter() - t_start) * 1000.0
        if self._metrics is not None:
            self._metrics.observe(timings)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "turn latency (ms): %s",
                {k: round(v, 1) for k, v in timings.items() if v is not None},
            )

        return output

    def _build_system_prompt(
        self,
        character: Character,
        emotional_state: EmotionalState,
        goal_tree: GoalTree,
        memories: list[EpisodicMemory | SemanticMemory],
        user_context_search_results: list[UserContextSearchResult] | None = None,
        knowledge_hits: list[KnowledgeHit] | None = None,
        web_search: WebSearchResponse | None = None,
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
                knowledge_hits=knowledge_hits,
                web_search=web_search,
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
            knowledge_hits=knowledge_hits,
            web_search=web_search,
        )

    @staticmethod
    def _history_assistant_content(
        structured: StructuredResponse | None, speech_text: str
    ) -> str:
        """把助手轮次规范成结构化 JSON 文本，作为上下文中的格式示例。

        无法解析成结构化响应（模型直接回了纯文本）时退回原文，至少不丢内容。
        """
        if structured is None:
            return speech_text
        return json.dumps(
            {
                "speech": structured.speech,
                "thought": structured.thought,
                "action": structured.action,
            },
            ensure_ascii=False,
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
        expressiveness: float = 1.0,
    ) -> None:
        """Run emotion evaluation via direct LLM estimation in the background."""
        try:
            result = await self._emotion_engine.evaluate(
                personality=personality,
                messages=messages,
                turn_count=self._turn_count,
                current_state=current_emotion,
                expressiveness=expressiveness,
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

    def _schedule_insight_acquisition(
        self, query: str, response: WebSearchResponse
    ) -> None:
        """把联网结果丢到后台提炼成行业认知块。

        刻意不 await：提炼要走一次大模型，放在关键路径上会明显拖慢回复。
        """
        assert self._insight_acquirer is not None
        task = asyncio.create_task(
            self._insight_acquirer.acquire(query, response)
        )
        self._insight_tasks.add(task)
        task.add_done_callback(self._insight_tasks.discard)

    async def aclose(self) -> None:
        """等待仍在进行的后台任务结束（应在会话结束时调用）。

        轮次之间**不会**等待这些任务，因此它们不会拖慢交互。
        """
        task = self._pending_emotion_task
        if task is not None and not task.done():
            try:
                await task
            except Exception:
                logger.warning("Pending emotion task failed")
        self._pending_emotion_task = None

        summary_task = self._summary_task
        if summary_task is not None and not summary_task.done():
            try:
                await summary_task
            except Exception:
                logger.warning("Pending summary task failed")
        self._summary_task = None

        pending = [t for t in self._insight_tasks if not t.done()]
        if pending:
            try:
                await asyncio.gather(*pending, return_exceptions=True)
            except Exception:  # noqa: BLE001 - 收尾失败不应影响会话结束
                logger.warning("Pending insight tasks failed")
        self._insight_tasks.clear()

    async def _generate_streaming(
        self,
        request: LLMRequest,
        on_delta: OnDelta,
        on_speech_complete: OnSpeechComplete | None = None,
    ) -> str:
        """流式生成，边产生边通过 ``on_delta`` 推送 speech 增量。

        ``on_speech_complete`` 在 speech 字段闭合的那一刻立即触发一次，
        早于 thought / action 生成完毕。

        Returns:
            完整的原始输出（等价于一次性 generate 的 content），
            供后续结构化解析使用。
        """
        buffer: list[str] = []
        emitted = 0
        speech_closed = False
        async for piece in self._llm.generate_stream(request):  # type: ignore[attr-defined]
            buffer.append(piece)
            joined = "".join(buffer)
            partial = extract_partial_speech(joined)
            if len(partial) > emitted:
                await on_delta(partial[emitted:])
                emitted = len(partial)
            if (
                not speech_closed
                and on_speech_complete is not None
                and speech_field_complete(joined)
            ):
                speech_closed = True
                await on_speech_complete(partial)

        raw = "".join(buffer)
        # 纯文本兜底等情形下，最终 speech 可能仍多于已推送部分，补发差量
        final_speech = parse_structured_response(raw).speech or ""
        if len(final_speech) > emitted:
            await on_delta(final_speech[emitted:])
        # 流中未能判定 speech 闭合（如 speech 为 null）时，兜底补一次
        if not speech_closed and on_speech_complete is not None:
            await on_speech_complete(final_speech)
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
        """裁剪历史到上限；被裁掉的消息交给后台摘要，不阻塞本轮回复。

        此前摘要在关键路径上同步 await（一次完整 LLM 调用），而历史超限后
        每次 append（用户消息、助手消息各一次）都会触发，等于每轮凭空多出
        一到两次 LLM 往返。现在裁剪同步完成、摘要走后台任务。
        """
        if len(self._history) <= self._history_limit:
            return

        overflow = len(self._history) - self._history_limit
        self._pending_summary_messages.extend(self._history[:overflow])
        self._history = self._history[-self._history_limit :]
        self._schedule_summarization()
        # 让刚排入的后台任务立刻起步：真实网络调用会在此挂起并让出控制权，
        # 不占用本轮回复；对不含真实 await 的场景也能确保它被及时执行。
        await asyncio.sleep(0)

    def _schedule_summarization(self) -> None:
        """把待摘要消息丢进后台任务；已有任务在跑时并入同一批处理。"""
        if self._summary_task is None or self._summary_task.done():
            self._summary_task = asyncio.create_task(self._run_summarization())

    async def _run_summarization(self) -> None:
        """按到达顺序逐批摘要，保证摘要可以层层累积。"""
        while self._pending_summary_messages:
            batch = self._pending_summary_messages
            self._pending_summary_messages = []
            await self._summarize_old_messages(batch)
