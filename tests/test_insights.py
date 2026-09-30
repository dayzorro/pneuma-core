"""Tests for 认知库：存储、去重、检索与合并。"""

from __future__ import annotations

import json

import pytest

from pneuma_core.knowledge.insights import (
    Insight,
    InsightKnowledgeBase,
    InsightStore,
    MergedKnowledgeBase,
)
from pneuma_core.knowledge.models import KnowledgeChunk
from pneuma_core.knowledge.retriever import KnowledgeBase


class FakeEmbeddingService:
    """按关键词给出向量的假 embedding，便于控制相似度。"""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.batch_calls = 0

    def _vector(self, text: str) -> list[float]:
        if "退货" in text:
            return [1.0, 0.0]
        if "促销" in text:
            return [0.0, 1.0]
        return [0.5, 0.5]

    async def embed(self, text: str) -> list[float]:
        if self.fail:
            raise RuntimeError("embedding down")
        return self._vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls += 1
        if self.fail:
            raise RuntimeError("embedding down")
        return [self._vector(t) for t in texts]


def _insight(content: str, topic: str = "服务规范") -> Insight:
    return Insight(id=f"i{abs(hash(content)) % 10**6}", content=content, topic=topic)


class TestInsightModel:
    def test_to_chunk_carries_metadata(self) -> None:
        chunk = _insight("退货需保留小票。", topic="售后").to_chunk()

        assert chunk.id.startswith("insight:")
        assert chunk.title == "行业认知 · 售后"
        assert chunk.content == "退货需保留小票。"
        assert chunk.tags == ("行业认知",)
        assert chunk.embedding is None

    def test_to_chunk_without_topic(self) -> None:
        assert _insight("x", topic="").to_chunk().title == "行业认知"

    def test_dict_roundtrip(self) -> None:
        original = Insight(
            id="abc",
            content="内容",
            topic="主题",
            source_urls=("https://a",),
            created_at="2026-09-30T00:00:00+00:00",
            embedding=[0.1, 0.2],
        )

        restored = Insight.from_dict(original.to_dict())

        assert restored == original

    def test_dict_omits_absent_embedding(self) -> None:
        assert "embedding" not in _insight("x").to_dict()


class TestInsightStore:
    def test_missing_file_loads_empty(self, tmp_path) -> None:
        store = InsightStore(tmp_path / "nope.json")

        store.load()

        assert store.size == 0

    def test_corrupt_file_loads_empty(self, tmp_path) -> None:
        path = tmp_path / "insights.json"
        path.write_text("not json", encoding="utf-8")
        store = InsightStore(path)

        store.load()

        assert store.size == 0

    @pytest.mark.asyncio
    async def test_add_and_reload(self, tmp_path) -> None:
        path = tmp_path / "insights.json"
        store = InsightStore(path, embedding_service=FakeEmbeddingService())

        added = await store.add_many([_insight("退货需保留小票。")])
        assert added == 1
        assert path.is_file()

        reloaded = InsightStore(path)
        reloaded.load()

        assert reloaded.size == 1
        assert reloaded.all()[0].content == "退货需保留小票。"

    @pytest.mark.asyncio
    async def test_stores_embeddings(self, tmp_path) -> None:
        store = InsightStore(
            tmp_path / "i.json", embedding_service=FakeEmbeddingService()
        )

        await store.add_many([_insight("退货需保留小票。")])

        assert store.all()[0].embedding == [1.0, 0.0]

    @pytest.mark.asyncio
    async def test_exact_duplicate_is_skipped(self, tmp_path) -> None:
        store = InsightStore(tmp_path / "i.json")

        await store.add_many([_insight("退货需保留小票。")])
        added = await store.add_many([_insight("退货需保留小票。")])

        assert added == 0
        assert store.size == 1

    @pytest.mark.asyncio
    async def test_near_duplicate_is_skipped_by_embedding(self, tmp_path) -> None:
        store = InsightStore(
            tmp_path / "i.json", embedding_service=FakeEmbeddingService()
        )

        await store.add_many([_insight("消费者享有七日无理由退货的权利。")])
        added = await store.add_many([_insight("门店办理退货时应当核验购物凭证。")])

        assert added == 0
        assert store.size == 1

    @pytest.mark.asyncio
    async def test_distinct_contents_are_kept(self, tmp_path) -> None:
        store = InsightStore(
            tmp_path / "i.json", embedding_service=FakeEmbeddingService()
        )

        await store.add_many(
            [
                _insight("消费者享有七日无理由退货的权利。"),
                _insight("节假日促销通常提前两周备货。"),
            ]
        )

        assert store.size == 2

    @pytest.mark.asyncio
    async def test_blank_content_is_ignored(self, tmp_path) -> None:
        store = InsightStore(tmp_path / "i.json")

        assert await store.add_many([_insight("   ")]) == 0
        assert store.size == 0

    @pytest.mark.asyncio
    async def test_embedding_failure_still_stores(self, tmp_path) -> None:
        store = InsightStore(
            tmp_path / "i.json", embedding_service=FakeEmbeddingService(fail=True)
        )

        added = await store.add_many([_insight("退货需保留小票。")])

        assert added == 1
        assert store.all()[0].embedding is None

    @pytest.mark.asyncio
    async def test_max_items_drops_oldest(self, tmp_path) -> None:
        store = InsightStore(tmp_path / "i.json", max_items=2)

        await store.add_many([_insight("第一条"), _insight("第二条")])
        await store.add_many([_insight("第三条")])

        assert store.size == 2
        assert [i.content for i in store.all()] == ["第二条", "第三条"]

    @pytest.mark.asyncio
    async def test_saved_json_is_readable(self, tmp_path) -> None:
        path = tmp_path / "i.json"
        store = InsightStore(path)
        await store.add_many([_insight("内容", topic="主题")])

        data = json.loads(path.read_text(encoding="utf-8"))

        assert data["version"] == 1
        assert data["insights"][0]["topic"] == "主题"


class TestInsightKnowledgeBase:
    @pytest.mark.asyncio
    async def test_empty_store_returns_nothing(self, tmp_path) -> None:
        store = InsightStore(tmp_path / "i.json")

        kb = InsightKnowledgeBase(store)

        assert kb.size == 0
        assert await kb.search("无理由退货") == []

    @pytest.mark.asyncio
    async def test_retrieves_added_insights(self, tmp_path) -> None:
        store = InsightStore(tmp_path / "i.json")
        kb = InsightKnowledgeBase(store)
        await store.add_many(
            [_insight("消费者享有七日无理由退货的权利。", topic="消费者权益")]
        )

        hits = await kb.search("无理由退货")

        assert hits
        assert hits[0].chunk.title == "行业认知 · 消费者权益"

    @pytest.mark.asyncio
    async def test_sees_insights_added_after_construction(self, tmp_path) -> None:
        """认知库在运行期增长，检索视图必须能看到新增内容。"""
        store = InsightStore(tmp_path / "i.json")
        kb = InsightKnowledgeBase(store)
        assert await kb.search("无理由退货") == []

        await store.add_many([_insight("消费者享有七日无理由退货的权利。")])

        assert await kb.search("无理由退货")


def _base(chunk_id: str, title: str, content: str) -> KnowledgeBase:
    return KnowledgeBase(
        [
            KnowledgeChunk(
                id=chunk_id, doc_id=chunk_id, title=title, content=content
            )
        ]
    )


class TestMergedKnowledgeBase:
    @pytest.mark.asyncio
    async def test_merges_results_from_all_bases(self) -> None:
        merged = MergedKnowledgeBase(
            [
                _base("a#0", "退货政策", "无理由退货需要购物凭证。"),
                _base("b#0", "行业认知", "退货时应当核验商品是否影响二次销售。"),
            ],
            top_k=5,
        )

        hits = await merged.search("退货")

        assert {h.chunk.id for h in hits} == {"a#0", "b#0"}

    @pytest.mark.asyncio
    async def test_respects_top_k(self) -> None:
        merged = MergedKnowledgeBase(
            [
                _base("a#0", "退货政策", "无理由退货需要购物凭证。"),
                _base("b#0", "退货流程", "退货到服务台办理。"),
            ],
            top_k=1,
        )

        assert len(await merged.search("退货")) == 1

    @pytest.mark.asyncio
    async def test_failing_base_does_not_break_others(self) -> None:
        class Broken:
            async def search(self, query: str):
                raise RuntimeError("boom")

        merged = MergedKnowledgeBase(
            [Broken(), _base("a#0", "退货政策", "无理由退货需要购物凭证。")],
            top_k=3,
        )

        hits = await merged.search("退货")

        assert hits and hits[0].chunk.id == "a#0"

    def test_size_and_mode_aggregate(self) -> None:
        merged = MergedKnowledgeBase(
            [_base("a#0", "t", "c"), _base("b#0", "t", "c")]
        )

        assert merged.size == 2
        assert merged.search_mode == "keyword"
