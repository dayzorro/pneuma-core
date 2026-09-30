"""Tests for 知识库检索（retriever）。"""

from __future__ import annotations

import pytest

from pneuma_core.knowledge.index import KnowledgeIndex
from pneuma_core.knowledge.models import KnowledgeChunk
from pneuma_core.knowledge.retriever import KnowledgeBase

_LEXICON = ["营业", "时间", "会员", "积分", "退货", "停车", "开门"]


class FakeEmbeddingService:
    """词频向量 + 可选故障注入。"""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.embed_calls = 0

    def _vector(self, text: str) -> list[float]:
        return [float(text.count(word)) for word in _LEXICON]

    async def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        if self.fail:
            raise RuntimeError("embedding endpoint down")
        return self._vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("embedding endpoint down")
        return [self._vector(t) for t in texts]


def _chunk(id_: str, title: str, content: str, *, embedded: bool = True):
    return KnowledgeChunk(
        id=id_,
        doc_id=id_.split("#")[0],
        title=title,
        content=content,
        embedding=(
            [float((title + content).count(w)) for w in _LEXICON]
            if embedded
            else None
        ),
    )


CHUNKS = [
    _chunk(
        "hours#0",
        "常见问题 · 你们几点开门？",
        "本店营业时间为每日 08:00 到 22:30，节假日以门店公告为准。",
    ),
    _chunk(
        "member#0",
        "会员积分 · 积分怎么算",
        "会员消费一元积一分，积分有效期为三年。",
    ),
    _chunk(
        "returns#0",
        "退换货 · 能退吗",
        "不影响二次销售的商品可以凭小票到服务台退货。",
    ),
]


class TestKeywordSearch:
    def test_matches_relevant_chunk(self) -> None:
        kb = KnowledgeBase(CHUNKS)

        hits = kb.search_keyword("你们几点开门")

        assert hits
        assert hits[0].chunk.id == "hours#0"
        assert hits[0].matched_by == "keyword"

    def test_ranks_more_relevant_chunk_first(self) -> None:
        kb = KnowledgeBase(CHUNKS)

        hits = kb.search_keyword("会员积分怎么算")

        assert hits[0].chunk.id == "member#0"

    def test_irrelevant_query_returns_nothing(self) -> None:
        kb = KnowledgeBase(CHUNKS)

        assert kb.search_keyword("今天天气怎么样") == []

    def test_blank_query_returns_nothing(self) -> None:
        kb = KnowledgeBase(CHUNKS)

        assert kb.search_keyword("   ") == []

    def test_punctuation_only_query_returns_nothing(self) -> None:
        kb = KnowledgeBase(CHUNKS)

        assert kb.search_keyword("！@#￥%……") == []

    def test_respects_top_k(self) -> None:
        chunks = [
            _chunk(f"doc{i}#0", "营业时间", "营业时间 08:00 开门") for i in range(6)
        ]
        kb = KnowledgeBase(chunks, top_k=2)

        assert len(kb.search_keyword("营业时间几点开门")) == 2


class TestVectorSearch:
    @pytest.mark.asyncio
    async def test_uses_vector_mode_when_embeddings_available(self) -> None:
        kb = KnowledgeBase(CHUNKS, embedding_service=FakeEmbeddingService())

        assert kb.search_mode == "vector"

        hits = await kb.search("你们几点开门")

        assert hits
        assert hits[0].matched_by == "vector"
        assert hits[0].chunk.id == "hours#0"

    @pytest.mark.asyncio
    async def test_falls_back_to_keyword_when_embedding_fails(self) -> None:
        service = FakeEmbeddingService(fail=True)
        kb = KnowledgeBase(CHUNKS, embedding_service=service)

        hits = await kb.search("会员积分怎么算")

        assert hits
        assert hits[0].matched_by == "keyword"
        assert hits[0].chunk.id == "member#0"
        assert service.embed_calls == 1

    @pytest.mark.asyncio
    async def test_chunks_without_embeddings_use_keyword_mode(self) -> None:
        chunks = [
            _chunk("a#0", "常见问题 · 你们几点开门？", "08:00 开门", embedded=False)
        ]
        kb = KnowledgeBase(chunks, embedding_service=FakeEmbeddingService())

        assert kb.search_mode == "keyword"

        hits = await kb.search("你们几点开门")

        assert hits and hits[0].matched_by == "keyword"

    @pytest.mark.asyncio
    async def test_dimension_mismatch_falls_back_to_keyword(self) -> None:
        chunks = [
            KnowledgeChunk(
                id="a#0",
                doc_id="a",
                title="常见问题 · 你们几点开门？",
                content="08:00 开门",
                embedding=[1.0, 0.0, 0.0],
            )
        ]
        kb = KnowledgeBase(chunks, embedding_service=FakeEmbeddingService())

        hits = await kb.search("你们几点开门")

        assert hits and hits[0].matched_by == "keyword"

    @pytest.mark.asyncio
    async def test_empty_query_short_circuits(self) -> None:
        service = FakeEmbeddingService()
        kb = KnowledgeBase(CHUNKS, embedding_service=service)

        assert await kb.search("") == []
        assert service.embed_calls == 0

    @pytest.mark.asyncio
    async def test_min_score_filters_off_topic_content(self) -> None:
        chunks = [
            KnowledgeChunk(
                id="a#0",
                doc_id="a",
                title="无关资料",
                content="完全不相关的内容",
                embedding=[0.0] * len(_LEXICON),
            )
        ]
        kb = KnowledgeBase(chunks, embedding_service=FakeEmbeddingService())

        assert await kb.search("你们几点开门") == []


class TestConstruction:
    def test_from_index(self) -> None:
        index = KnowledgeIndex(chunks=CHUNKS, model="m", dimension=len(_LEXICON))

        kb = KnowledgeBase.from_index(index, embedding_service=FakeEmbeddingService())

        assert kb.size == len(CHUNKS)
        assert kb.has_embeddings

    def test_from_directory_builds_keyword_only_base(self, tmp_path) -> None:
        (tmp_path / "a.md").write_text(
            "---\ndoc_id: a\ntitle: 常见问题\n---\n\n## 你们几点开门\n08:00 开门\n",
            encoding="utf-8",
        )

        kb = KnowledgeBase.from_directory(tmp_path)

        assert kb.search_mode == "keyword"
        assert kb.size >= 1

    def test_shipped_corpus_answers_common_questions(self) -> None:
        from pneuma_core.knowledge import DEFAULT_DATA_DIR

        kb = KnowledgeBase.from_directory(DEFAULT_DATA_DIR)

        cases = [
            ("你们几点开门", "faq"),
            ("会员积分怎么算", "membership"),
            ("买的东西能退吗", "faq"),
            ("购物卡过期了怎么办", "faq"),
            ("能借雨伞吗", "faq"),
            ("我想投诉，找谁", "faq"),
        ]
        for question, expected_doc in cases:
            hits = kb.search_keyword(question)
            assert hits, f"未命中：{question}"
            assert any(expected_doc in h.chunk.doc_id for h in hits), question

    def test_shipped_corpus_ignores_off_topic_questions(self) -> None:
        from pneuma_core.knowledge import DEFAULT_DATA_DIR

        kb = KnowledgeBase.from_directory(DEFAULT_DATA_DIR)

        for question in ["今天天气怎么样", "帮我写一首诗", "1+1 等于几"]:
            assert kb.search_keyword(question) == [], question
