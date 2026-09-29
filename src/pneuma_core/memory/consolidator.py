"""记忆整合：从对话中按重要度抽取情节记忆。"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from pneuma_core.llm.adapter import LLMAdapter, LLMRequest, LLMResponse
from pneuma_core.llm.embedding import EmbeddingService
from pneuma_core.memory.store import MemoryStore
from pneuma_core.models.memory import EpisodicMemory

EXTRACTION_SYSTEM_PROMPT = """\
你是一个从对话中抽取情节记忆的助手。
请阅读以下对话历史，抽取角色应当记住的重要事件。
所有文本字段必须使用简体中文。

输出为 JSON 数组，每个元素格式如下:
[
  {
    "content": "所记事件的描述（1〜2 句）",
    "importance": 0.0〜1.0 的数值,
    "emotional_valence": -1.0〜1.0 的数值（不悦〜愉悦）
  }
]

重要度基准:
- 0.9-1.0: 改变人生的事件、决定性约定
- 0.7-0.8: 印象强烈、重要信息的披露
- 0.6: 记忆的最低门槛
- 低于 0.6: 不必保存的琐碎事件（可保留，但请如实标注重要度）

只输出 JSON 数组，不要任何说明文字。
"""


@dataclass(frozen=True)
class ConsolidationConfig:
    """记忆整合的可调参数。"""

    importance_threshold: float = 0.6
    max_episodes_per_conversation: int = 3
    duplicate_similarity_threshold: float = 0.95


@dataclass
class ConsolidationResult:
    """记忆整合的结果。"""

    saved: list[EpisodicMemory] = field(default_factory=list)
    filtered_by_importance: int = 0
    filtered_by_duplicate: int = 0
    filtered_by_limit: int = 0


class MemoryConsolidator:
    """选择性抽取与存储情节记忆的引擎。

    流程:
        1. LLM 从对话历史中抽取情节
        2. 按重要度阈值过滤
        3. 限制数量（按重要度取前 N 条）
        4. 与已有记忆做重复检查
        5. 生成 embedding 并保存到 store
    """

    def __init__(
        self,
        llm: LLMAdapter,
        embedding_service: EmbeddingService,
        store: MemoryStore,
        config: ConsolidationConfig | None = None,
        model: str | None = None,
    ) -> None:
        self.llm = llm
        self.embedding_service = embedding_service
        self.store = store
        self.config = config or ConsolidationConfig()
        self._model = model

    async def consolidate(
        self,
        character_id: str,
        conversation_history: list[dict],
        now: datetime,
        conversation_id: str | None = None,
    ) -> ConsolidationResult:
        """从对话中抽取、过滤并保存情节记忆。"""
        result = ConsolidationResult()

        # 1. 用 LLM 抽取情节
        raw_episodes = await self._extract_episodes(conversation_history)
        if not raw_episodes:
            return result

        # 2. 按重要度过滤
        filtered = []
        for ep in raw_episodes:
            if ep["importance"] >= self.config.importance_threshold:
                filtered.append(ep)
            else:
                result.filtered_by_importance += 1

        if not filtered:
            return result

        # 3. 数量限制（按 importance 降序取前 N 条）
        filtered.sort(key=lambda x: x["importance"], reverse=True)
        max_n = self.config.max_episodes_per_conversation
        if len(filtered) > max_n:
            result.filtered_by_limit = len(filtered) - max_n
            filtered = filtered[:max_n]

        # 4. 生成 Embedding
        contents = [ep["content"] for ep in filtered]
        embeddings = await self.embedding_service.embed_batch(contents)

        # 5. 重复检查（经由 find_similar_episodic）
        non_duplicate = []
        for ep, emb in zip(filtered, embeddings):
            similar = await self.store.find_similar_episodic(
                character_id, emb, self.config.duplicate_similarity_threshold
            )
            if similar:
                result.filtered_by_duplicate += 1
            else:
                non_duplicate.append((ep, emb))

        # 6. 创建并保存 EpisodicMemory
        for ep, emb in non_duplicate:
            memory = EpisodicMemory(
                id=f"ep-{uuid.uuid4().hex[:12]}",
                character_id=character_id,
                content=ep["content"],
                timestamp=now,
                emotional_valence=ep.get("emotional_valence", 0.0),
                importance=ep["importance"],
                conversation_id=conversation_id,
                embedding=emb,
            )
            await self.store.add_episodic(memory)
            result.saved.append(memory)

        return result

    async def _extract_episodes(
        self, conversation_history: list[dict]
    ) -> list[dict]:
        """Use LLM to extract episodes from conversation history."""
        request = LLMRequest(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            messages=conversation_history,
            model=self._model,
            temperature=0.3,
            max_tokens=2048,
        )
        try:
            response: LLMResponse = await self.llm.generate(request)
        except Exception:
            return []

        try:
            episodes = json.loads(response.content)
            if not isinstance(episodes, list):
                return []
            return [ep for ep in episodes if self._validate_episode(ep)]
        except (json.JSONDecodeError, TypeError):
            return []

    @staticmethod
    def _validate_episode(ep: object) -> bool:
        """Validate that an episode dict has required fields with correct types."""
        if not isinstance(ep, dict):
            return False
        if "content" not in ep or not isinstance(ep["content"], str):
            return False
        if "importance" not in ep:
            return False
        try:
            importance = float(ep["importance"])
        except (TypeError, ValueError):
            return False
        if not (0.0 <= importance <= 1.0):
            return False
        if "emotional_valence" in ep:
            try:
                valence = float(ep["emotional_valence"])
            except (TypeError, ValueError):
                return False
            if not (-1.0 <= valence <= 1.0):
                return False
        return True

