"""Chat service: wires Pneuma Core into a single-user conversation session."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from pneuma_core.character_sheet import CharacterSheet
from pneuma_core.llm.embedding import OpenAIEmbeddingService
from pneuma_core.llm.openai_compat import OpenAICompatAdapter
from pneuma_core.models.message import MessageInput, MessageOutput
from pneuma_core.runtime.engine import RuntimeEngine
from pneuma_core.runtime.prompt_cache import PromptCache
from pneuma_core.runtime.session import ConversationSession
from pneuma_core.runtime.session_end_pipeline import SessionEndPipeline
from pneuma_core.server.config import ServerConfig
from pneuma_core.storage.sqlite import SQLiteStorageBackend

logger = logging.getLogger(__name__)


class SessionNotStartedError(RuntimeError):
    """Raised when a chat is attempted without an active session."""


class ChatService:
    """Single-user chat service.

    Lifecycle:
        setup() -> start_session(user) -> chat(message) ... -> end_session()

    Memories (episodic / semantic / relations) are written when the session
    ends, mirroring the framework's per-session phase.
    """

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._storage = SQLiteStorageBackend(str(config.db_path))
        self._llm = OpenAICompatAdapter(
            api_key=config.llm_api_key,
            default_model=config.llm_model,
            base_url=config.llm_base_url,
            timeout=config.llm_timeout,
        )
        self._embedding = OpenAIEmbeddingService(
            api_key=config.embedding_api_key,
            model=config.embedding_model,
            base_url=config.embedding_base_url,
        )
        self._sheet: CharacterSheet | None = None
        self._engine: RuntimeEngine | None = None
        self._session: ConversationSession | None = None
        self._user_id: str = ""
        self._user_name: str = ""

    # ── Setup ────────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Initialise storage and persist the character definition."""
        self._config.db_path.parent.mkdir(parents=True, exist_ok=True)
        await self._storage.initialize()

        sheet = CharacterSheet.load(self._config.character_file)
        self._sheet = sheet
        character = sheet.character

        if await self._storage.get_character(character.id) is None:
            await self._storage.save_character(character)
        if sheet.goal_tree is not None:
            await self._storage.save_goals(character.id, sheet.goal_tree)
        if (
            sheet.initial_state is not None
            and await self._storage.get_emotional_state(character.id) is None
        ):
            await self._storage.save_emotional_state(character.id, sheet.initial_state)

        logger.info(
            "Chat service ready (character=%s, db=%s)",
            character.name,
            self._config.db_path,
        )

    # ── Character info ───────────────────────────────────────────────────

    def _require_sheet(self) -> CharacterSheet:
        if self._sheet is None:
            raise RuntimeError("Service not initialised: call setup() first")
        return self._sheet

    def character_info(self) -> dict[str, Any]:
        character = self._require_sheet().character
        return {
            "id": character.id,
            "name": character.name,
            "profile": character.profile,
            "appearance": character.appearance,
            "speaking_style": character.speaking_style,
            "personality": {
                "openness": character.personality.openness,
                "conscientiousness": character.personality.conscientiousness,
                "extraversion": character.personality.extraversion,
                "agreeableness": character.personality.agreeableness,
                "neuroticism": character.personality.neuroticism,
            },
        }

    # ── Session lifecycle ────────────────────────────────────────────────

    async def start_session(self, user_id: str, user_name: str) -> dict[str, Any]:
        """Start a session, ending any previously active one first."""
        if self._session is not None:
            await self.end_session()

        character = self._require_sheet().character
        now = datetime.now(timezone.utc)
        self._user_id = user_id
        self._user_name = user_name
        self._session = ConversationSession(
            session_id=f"sess-{uuid.uuid4().hex[:12]}",
            character_id=character.id,
            user_id=user_id,
            channel_id="api",
            started_at=now,
            last_active_at=now,
        )
        self._engine = RuntimeEngine(
            character_id=character.id,
            storage=self._storage,
            llm=self._llm,
            embedding_service=self._embedding,
            memory_store=self._storage,
            history_limit=self._config.history_limit,
            diagnostic_mode=self._config.diagnostic_mode,
            prompt_cache=PromptCache(),
        )
        return {
            "session_id": self._session.session_id,
            "user": {"id": user_id, "name": user_name},
            "character": {
                "id": character.id,
                "name": character.name,
                "profile": character.profile,
            },
        }

    @staticmethod
    def _output_payload(output: MessageOutput) -> dict[str, Any]:
        """把 MessageOutput 转成 API 返回体。"""
        return {
            "reply": output.content,
            "thought": output.thought,
            "action": output.action,
            "emotion": {
                "pleasure": output.emotion.pleasure,
                "arousal": output.emotion.arousal,
                "dominance": output.emotion.dominance,
                "label": output.emotion.emotion_label,
            },
            "system_messages": [
                {"type": m.type, "message": m.message, "component": m.component}
                for m in output.system_messages
            ],
        }

    def _record_turn(self, message: str, output: MessageOutput) -> None:
        """把本轮对话写入会话记录。"""
        assert self._session is not None
        self._session.messages.append(
            {"role": "user", "content": f"[{self._user_name}] {message}"}
        )
        self._session.messages.append(
            {"role": "assistant", "content": output.content}
        )
        self._session.last_active_at = datetime.now(timezone.utc)

    async def chat(self, message: str) -> dict[str, Any]:
        """Process one user message and return the character's reply."""
        if self._engine is None or self._session is None:
            raise SessionNotStartedError("No active session")

        output = await self._engine.process_message(
            MessageInput(
                content=message,
                sender_id=self._user_id,
                sender_name=self._user_name,
                sender_type="human",
            )
        )

        self._record_turn(message, output)
        return self._output_payload(output)

    async def chat_stream(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """流式处理一条消息，逐段 yield 事件。

        事件类型：
            {"type": "delta", "text": "..."}  —— speech 增量
            {"type": "done", ...返回体...}     —— 完成（含 emotion 等）
            {"type": "error", "message": "..."}
        """
        if self._engine is None or self._session is None:
            raise SessionNotStartedError("No active session")

        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def on_delta(text: str) -> None:
            await queue.put({"type": "delta", "text": text})

        async def run() -> None:
            try:
                assert self._engine is not None
                output = await self._engine.process_message(
                    MessageInput(
                        content=message,
                        sender_id=self._user_id,
                        sender_name=self._user_name,
                        sender_type="human",
                    ),
                    on_delta=on_delta,
                )
                self._record_turn(message, output)
                await queue.put({"type": "done", **self._output_payload(output)})
            except Exception as e:  # noqa: BLE001 - 边界处转成事件而非抛出
                logger.warning("chat_stream failed: %s: %s", type(e).__name__, e)
                await queue.put({"type": "error", "message": str(e)})
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            if not task.done():
                task.cancel()

    async def end_session(self) -> dict[str, Any]:
        """End the session and consolidate memories via the LLM."""
        if self._session is None:
            raise SessionNotStartedError("No active session")

        session = self._session
        engine = self._engine
        self._session = None
        self._engine = None

        # 等待仍在进行的情绪评估任务收尾，避免留下悬挂任务
        if engine is not None:
            await engine.aclose()

        existing_semantics = await self._storage.get_semantic_memories(
            session.character_id
        )
        existing_relations = await self._storage.list_relations(
            owner_id=session.character_id
        )

        pipeline = SessionEndPipeline(
            llm=self._llm,
            embedding_service=self._embedding,
            memory_store=self._storage,
            storage=self._storage,
            user_context_dir=(
                str(self._config.user_context_dir)
                if self._config.user_context_dir
                else None
            ),
            model=self._config.llm_model,
        )
        result = await pipeline.run(
            session=session,
            character_id=session.character_id,
            existing_semantics=existing_semantics,
            existing_relations=existing_relations,
        )

        return {
            "session_id": session.session_id,
            "success": result.success,
            "episodic_memories_saved": result.episodic_memories_saved,
            "semantic_updates_applied": result.semantic_updates_applied,
            "user_context_updates": result.user_context_updates,
            "relationship_changes": result.relationship_changes,
        }

    # ── State inspection ─────────────────────────────────────────────────

    async def state(self) -> dict[str, Any]:
        """Return the current persisted state of the character."""
        character_id = self._require_sheet().character.id

        emotion = await self._storage.get_emotional_state(character_id)
        episodic = await self._storage.get_episodic_memories(character_id)
        semantic = await self._storage.get_semantic_memories(character_id)
        relations = await self._storage.list_relations(owner_id=character_id)

        return {
            "character_id": character_id,
            "session_active": self._session is not None,
            "session_id": self._session.session_id if self._session else None,
            "emotion": (
                {
                    "pleasure": emotion.pleasure,
                    "arousal": emotion.arousal,
                    "dominance": emotion.dominance,
                    "label": emotion.emotion_label,
                    "situation": emotion.situation,
                }
                if emotion
                else None
            ),
            "memory_counts": {
                "episodic": len(episodic),
                "semantic": len(semantic),
            },
            "relations": [
                {
                    "id": r.id,
                    "target_name": r.target_name,
                    "relationship_type": r.relationship_type,
                    "closeness": r.closeness,
                    "trust": r.trust,
                }
                for r in relations
            ],
        }
