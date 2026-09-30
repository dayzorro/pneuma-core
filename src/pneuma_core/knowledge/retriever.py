"""知识库检索：向量检索 + 关键词兜底。

服务启动时加载索引后，每轮对话用顾客的问题做一次检索，把命中的知识块
注入提示词，用于「专业知识问答」。

两种模式：
    - 向量检索：索引里有向量且提供了 embedding 服务时使用（优先）
    - 关键词检索：没有向量、没有 embedding 服务，或向量检索失败时使用
      （基于字符二元组 + IDF 加权，中文无需分词）
"""

from __future__ import annotations

import logging
import math
import re

from pneuma_core.knowledge.index import KnowledgeIndex, load_documents
from pneuma_core.knowledge.models import KnowledgeChunk, KnowledgeHit
from pneuma_core.memory.similarity import cosine_similarity
from pneuma_core.protocols.embedding import EmbeddingService

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3
# 阈值按实测标定（见 tests/test_knowledge_retriever.py）：
#   向量模式：相关问题余弦相似度普遍 > 0.55，离题问题 < 0.40
#   关键词模式：原句命中 0.8〜1.0，离题问题 < 0.25
DEFAULT_MIN_SCORE = 0.45
DEFAULT_KEYWORD_MIN_SCORE = 0.30

_NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)


def _normalize(text: str) -> str:
    """去掉标点与空白，保留文字/数字/字母。"""
    return _NON_WORD_RE.sub("", text.lower())


def _bigrams(text: str) -> set[str]:
    """字符二元组（中文免分词）。单字文本退化为单字集合。"""
    normalized = _normalize(text)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[i : i + 2] for i in range(len(normalized) - 1)}


class KnowledgeBase:
    """可检索的本地知识库。"""

    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        *,
        embedding_service: EmbeddingService | None = None,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        keyword_min_score: float = DEFAULT_KEYWORD_MIN_SCORE,
    ) -> None:
        self._chunks = list(chunks)
        self._embedding_service = embedding_service
        self._top_k = max(1, top_k)
        self._min_score = min_score
        self._keyword_min_score = keyword_min_score

        self._chunk_bigrams: list[set[str]] = [
            _bigrams(chunk.full_text()) for chunk in self._chunks
        ]
        self._idf = self._compute_idf(self._chunk_bigrams)
        self._default_idf = (
            math.log(len(self._chunks) + 1) + 1.0 if self._chunks else 1.0
        )

    # ── 构造 ────────────────────────────────────────────────────────────

    @classmethod
    def from_index(
        cls,
        index: KnowledgeIndex,
        *,
        embedding_service: EmbeddingService | None = None,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        keyword_min_score: float = DEFAULT_KEYWORD_MIN_SCORE,
    ) -> KnowledgeBase:
        return cls(
            index.chunks,
            embedding_service=embedding_service,
            top_k=top_k,
            min_score=min_score,
            keyword_min_score=keyword_min_score,
        )

    @classmethod
    def from_directory(
        cls,
        docs_dir,
        *,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        keyword_min_score: float = DEFAULT_KEYWORD_MIN_SCORE,
    ) -> KnowledgeBase:
        """直接从 Markdown 目录构造（无向量，只能关键词检索）。"""
        return cls(
            load_documents(docs_dir),
            embedding_service=None,
            top_k=top_k,
            min_score=min_score,
            keyword_min_score=keyword_min_score,
        )

    # ── 检索 ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def has_embeddings(self) -> bool:
        return any(chunk.embedding for chunk in self._chunks)

    @property
    def search_mode(self) -> str:
        """当前实际可用的检索模式。"""
        if self._embedding_service is not None and self.has_embeddings:
            return "vector"
        return "keyword"

    async def search(self, query: str) -> list[KnowledgeHit]:
        """检索与 query 最相关的知识块。"""
        query = (query or "").strip()
        if not query or not self._chunks:
            return []

        if self._embedding_service is not None and self.has_embeddings:
            try:
                query_embedding = await self._embedding_service.embed(query)
                hits = self._search_vector(query_embedding)
                if hits:
                    return hits
            except Exception as e:  # noqa: BLE001 - 检索失败不应影响对话
                logger.warning(
                    "Knowledge vector search failed (%s: %s), falling back to keyword",
                    type(e).__name__,
                    e,
                )

        return self.search_keyword(query)

    def _search_vector(self, query_embedding: list[float]) -> list[KnowledgeHit]:
        scored: list[KnowledgeHit] = []
        for chunk in self._chunks:
            if not chunk.embedding:
                continue
            # 维度不一致（换过 embedding 模型）时跳过，交由关键词兜底
            if len(chunk.embedding) != len(query_embedding):
                continue
            score = cosine_similarity(chunk.embedding, query_embedding)
            if score >= self._min_score:
                scored.append(KnowledgeHit(chunk=chunk, score=score, matched_by="vector"))

        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[: self._top_k]

    def search_keyword(self, query: str) -> list[KnowledgeHit]:
        """基于字符二元组 + IDF 加权的关键词检索。"""
        query_grams = _bigrams(query)
        if not query_grams:
            return []

        total_weight = sum(self._weight_of(g) for g in query_grams)
        if total_weight <= 0:
            return []

        scored: list[KnowledgeHit] = []
        for chunk, grams in zip(self._chunks, self._chunk_bigrams):
            matched_weight = sum(
                self._weight_of(g) for g in query_grams if g in grams
            )
            score = matched_weight / total_weight
            if score >= self._keyword_min_score:
                scored.append(
                    KnowledgeHit(chunk=chunk, score=score, matched_by="keyword")
                )

        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[: self._top_k]

    def _weight_of(self, gram: str) -> float:
        """查询词的权重。

        未在语料中出现过的词按「最高权重」处理——它理应最有区分度，
        若按 1.0 处理会稀释分母，让偶然命中的常见词获得虚高的覆盖率。
        """
        return self._idf.get(gram, self._default_idf)

    @staticmethod
    def _compute_idf(chunk_bigrams: list[set[str]]) -> dict[str, float]:
        """字符二元组的逆文档频率，用于给关键词加权。"""
        total = len(chunk_bigrams)
        if total == 0:
            return {}

        df: dict[str, int] = {}
        for grams in chunk_bigrams:
            for gram in grams:
                df[gram] = df.get(gram, 0) + 1

        return {
            gram: math.log((total + 1) / (count + 1)) + 1.0
            for gram, count in df.items()
        }
