"""认知库：从联网信息中异步提炼出的「行业通识认知块」。

与人工整理的资料库（``knowledge/data/``）分开存放，因为它的写入时机与内容
来源完全不同：

    - 资料库：人工撰写、随仓库分发、只读
    - 认知库：由联网检索结果经 LLM 提炼而来、运行期持续增长、与具体公司无关

认知块的定位是**前台岗位的行业通识**——零售/商超行业的惯例、消费者权益
常识、服务经验、行业动向等；不包含任何与特定公司绑定的经营细节，也不包含
个人隐私或保密信息（由提炼提示词负责约束）。

存储用 JSON 文件（人类可读、便于检查），检索复用 ``KnowledgeBase``
（向量优先 + 关键词兜底），新增认知块后懒重建索引。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pneuma_core.knowledge.models import KnowledgeChunk, KnowledgeHit
from pneuma_core.knowledge.retriever import (
    DEFAULT_KEYWORD_MIN_SCORE,
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    KnowledgeBase,
)
from pneuma_core.memory.similarity import cosine_similarity
from pneuma_core.protocols.embedding import EmbeddingService

logger = logging.getLogger(__name__)

STORE_VERSION = 1
DEFAULT_MAX_ITEMS = 2000
DEFAULT_DEDUP_THRESHOLD = 0.92

_NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)


@dataclass(frozen=True)
class Insight:
    """一条行业通识认知块。"""

    id: str
    content: str
    topic: str = ""
    source_urls: tuple[str, ...] = ()
    created_at: str = ""
    embedding: list[float] | None = None

    def to_chunk(self) -> KnowledgeChunk:
        """转成可检索的知识块。"""
        title = f"行业认知 · {self.topic}" if self.topic else "行业认知"
        return KnowledgeChunk(
            id=f"insight:{self.id}",
            doc_id="insight",
            title=title,
            content=self.content,
            source="联网检索提炼",
            tags=("行业认知",),
            embedding=self.embedding,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "content": self.content,
            "topic": self.topic,
            "source_urls": list(self.source_urls),
            "created_at": self.created_at,
        }
        if self.embedding is not None:
            data["embedding"] = self.embedding
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Insight:
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:12]),
            content=str(data.get("content") or ""),
            topic=str(data.get("topic") or ""),
            source_urls=tuple(str(u) for u in data.get("source_urls", [])),
            created_at=str(data.get("created_at") or ""),
            embedding=list(data["embedding"]) if data.get("embedding") else None,
        )


def _normalize(text: str) -> str:
    return _NON_WORD_RE.sub("", text.lower())


class InsightStore:
    """认知库的持久化与去重。"""

    def __init__(
        self,
        path: Path,
        *,
        embedding_service: EmbeddingService | None = None,
        max_items: int = DEFAULT_MAX_ITEMS,
        dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    ) -> None:
        self._path = path
        self._embedding_service = embedding_service
        self._max_items = max(1, max_items)
        self._dedup_threshold = dedup_threshold
        self._insights: list[Insight] = []
        self._normalized: set[str] = set()

    # ── 读写 ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """从磁盘加载。文件不存在或损坏时按空库处理。"""
        if not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            logger.warning("Failed to read insight store %s: %s", self._path, e)
            return
        if not isinstance(data, dict):
            return

        insights = [
            Insight.from_dict(item)
            for item in data.get("insights", [])
            if isinstance(item, dict)
        ]
        self._insights = [i for i in insights if i.content]
        self._normalized = {_normalize(i.content) for i in self._insights}

    def save(self) -> None:
        """原子写回磁盘。"""
        payload = {
            "version": STORE_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "insights": [i.to_dict() for i in self._insights],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        tmp.replace(self._path)

    # ── 写入 ────────────────────────────────────────────────────────────

    async def add_many(self, insights: list[Insight]) -> int:
        """去重后追加，返回真正新增的条数。"""
        candidates = [i for i in insights if i.content.strip()]
        if not candidates:
            return 0

        if self._embedding_service is not None:
            candidates = await self._fill_embeddings(candidates)

        added: list[Insight] = []
        for insight in candidates:
            if self._is_duplicate(insight):
                continue
            self._insights.append(insight)
            self._normalized.add(_normalize(insight.content))
            added.append(insight)

        if not added:
            return 0

        if len(self._insights) > self._max_items:
            overflow = len(self._insights) - self._max_items
            self._insights = self._insights[overflow:]
            self._normalized = {_normalize(i.content) for i in self._insights}

        self.save()
        return len(added)

    async def _fill_embeddings(self, insights: list[Insight]) -> list[Insight]:
        pending = [i for i in insights if i.embedding is None]
        if not pending:
            return insights

        try:
            vectors = await self._embedding_service.embed_batch(  # type: ignore[union-attr]
                [i.content for i in pending]
            )
        except Exception as e:  # noqa: BLE001 - 没有向量也要能入库
            logger.warning(
                "Failed to embed insights (%s: %s), storing without vectors",
                type(e).__name__,
                e,
            )
            return insights

        by_id = {
            i.id: replace(i, embedding=v or None)
            for i, v in zip(pending, vectors)
        }
        return [by_id.get(i.id, i) for i in insights]

    def _is_duplicate(self, insight: Insight) -> bool:
        if _normalize(insight.content) in self._normalized:
            return True
        if insight.embedding is None:
            return False

        for existing in self._insights:
            if existing.embedding is None:
                continue
            if len(existing.embedding) != len(insight.embedding):
                continue
            try:
                if (
                    cosine_similarity(existing.embedding, insight.embedding)
                    >= self._dedup_threshold
                ):
                    return True
            except ValueError:
                continue
        return False

    # ── 读取 ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._insights)

    def all(self) -> list[Insight]:
        return list(self._insights)

    def chunks(self) -> list[KnowledgeChunk]:
        return [i.to_chunk() for i in self._insights]


class InsightKnowledgeBase:
    """认知库的检索视图。

    认知库会在运行期增长，因此这里只在「有新内容」时重建底层索引，
    检索请求之间复用同一个 ``KnowledgeBase``。
    """

    def __init__(
        self,
        store: InsightStore,
        *,
        embedding_service: EmbeddingService | None = None,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        keyword_min_score: float = DEFAULT_KEYWORD_MIN_SCORE,
    ) -> None:
        self._store = store
        self._embedding_service = embedding_service
        self._top_k = top_k
        self._min_score = min_score
        self._keyword_min_score = keyword_min_score
        self._base: KnowledgeBase | None = None
        self._built_size = -1

    def invalidate(self) -> None:
        self._base = None
        self._built_size = -1

    def _ensure(self) -> KnowledgeBase:
        if self._base is not None and self._built_size == self._store.size:
            return self._base

        self._base = KnowledgeBase(
            self._store.chunks(),
            embedding_service=self._embedding_service,
            top_k=self._top_k,
            min_score=self._min_score,
            keyword_min_score=self._keyword_min_score,
        )
        self._built_size = self._store.size
        return self._base

    @property
    def size(self) -> int:
        return self._store.size

    @property
    def search_mode(self) -> str:
        return self._ensure().search_mode

    async def search(
        self, query: str, *, query_embedding: list[float] | None = None
    ) -> list[KnowledgeHit]:
        if self._store.size == 0:
            return []
        return await self._ensure().search(query, query_embedding=query_embedding)


class MergedKnowledgeBase:
    """把多个知识库合并成一个检索入口。

    各子库已按自身的 top_k 截断，这里再按分数归并取全局 top_k。
    注意：不同子库的分数尺度可能不同（例如向量余弦 vs 关键词覆盖率），
    归并只做粗排，用于把「资料 + 认知」一起喂给提示词。
    """

    def __init__(self, bases: list[Any], *, top_k: int = DEFAULT_TOP_K) -> None:
        self._bases = bases
        self._top_k = max(1, top_k)

    async def search(
        self, query: str, *, query_embedding: list[float] | None = None
    ) -> list[KnowledgeHit]:
        merged: list[KnowledgeHit] = []
        seen: set[str] = set()

        for base in self._bases:
            try:
                hits = await base.search(query, query_embedding=query_embedding)
            except Exception as e:  # noqa: BLE001 - 单个子库失败不影响其它库
                logger.warning(
                    "Knowledge base %s search failed: %s: %s",
                    type(base).__name__,
                    type(e).__name__,
                    e,
                )
                continue
            for hit in hits:
                if hit.chunk.id in seen:
                    continue
                seen.add(hit.chunk.id)
                merged.append(hit)

        merged.sort(key=lambda hit: hit.score, reverse=True)
        return merged[: self._top_k]

    @property
    def size(self) -> int:
        total = 0
        for base in self._bases:
            total += int(getattr(base, "size", 0))
        return total

    @property
    def search_mode(self) -> str:
        modes = [
            str(getattr(base, "search_mode", "keyword")) for base in self._bases
        ]
        return "vector" if modes and all(m == "vector" for m in modes) else "keyword"


__all__ = [
    "DEFAULT_DEDUP_THRESHOLD",
    "DEFAULT_MAX_ITEMS",
    "Insight",
    "InsightKnowledgeBase",
    "InsightStore",
    "MergedKnowledgeBase",
    "STORE_VERSION",
]
