"""Tests for 知识库索引构建与落盘（index）。"""

from __future__ import annotations

import json

import pytest

from pneuma_core.knowledge.index import (
    DEFAULT_DATA_DIR,
    KnowledgeIndex,
    build_index,
    load_documents,
)
from pneuma_core.knowledge.models import KnowledgeChunk

_LEXICON = ["营业", "会员", "积分", "退货", "停车"]

_DOCS = {
    "10-会员积分.md": """\
---
doc_id: membership
title: 会员积分
tags: 会员
---

# 会员积分

## 怎么办会员

带证件到服务台办理。

## 积分规则

消费一元积一分。
""",
    "20-退换货.md": """\
---
doc_id: returns
title: 退换货
---

# 退换货

## 能退吗

不影响二次销售就可以退。
""",
}


class FakeEmbeddingService:
    """把文本映射为词频向量，并记录每次批请求的条数。"""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def _vector(self, text: str) -> list[float]:
        return [float(text.count(word)) for word in _LEXICON]

    async def embed(self, text: str) -> list[float]:
        return self._vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [self._vector(t) for t in texts]


@pytest.fixture
def docs_dir(tmp_path):
    for name, content in _DOCS.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    return tmp_path


class TestLoadDocuments:
    def test_loads_all_markdown_files(self, docs_dir) -> None:
        chunks = load_documents(docs_dir)

        assert {c.doc_id for c in chunks} == {"membership", "returns"}

    def test_missing_directory_returns_empty(self, tmp_path) -> None:
        assert load_documents(tmp_path / "nope") == []

    def test_underscore_prefixed_files_are_skipped(self, docs_dir) -> None:
        (docs_dir / "_notes.md").write_text("## 说明\n不该被收录\n", encoding="utf-8")

        chunks = load_documents(docs_dir)

        assert all("不该被收录" not in c.content for c in chunks)


class TestBuildIndex:
    @pytest.mark.asyncio
    async def test_attaches_embeddings_and_metadata(self, docs_dir) -> None:
        service = FakeEmbeddingService()

        index = await build_index(docs_dir, service, model="fake-model")

        assert index.model == "fake-model"
        assert index.dimension == len(_LEXICON)
        assert index.built_at
        assert index.has_embeddings
        assert all(len(c.embedding) == len(_LEXICON) for c in index.chunks)

    @pytest.mark.asyncio
    async def test_splits_requests_into_batches(self, docs_dir) -> None:
        service = FakeEmbeddingService()

        await build_index(docs_dir, service, batch_size=2)

        assert all(size <= 2 for size in service.batch_sizes)
        assert sum(service.batch_sizes) == len(load_documents(docs_dir))

    @pytest.mark.asyncio
    async def test_empty_directory_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="No knowledge documents"):
            await build_index(tmp_path, FakeEmbeddingService())

    @pytest.mark.asyncio
    async def test_empty_vectors_raise(self, docs_dir) -> None:
        class EmptyEmbeddingService:
            async def embed(self, text: str) -> list[float]:
                return []

            async def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[] for _ in texts]

        with pytest.raises(ValueError, match="empty vectors"):
            await build_index(docs_dir, EmptyEmbeddingService())


class TestIndexPersistence:
    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(self, docs_dir, tmp_path) -> None:
        index = await build_index(
            docs_dir, FakeEmbeddingService(), model="fake-model"
        )
        path = tmp_path / "nested" / "index.json"

        index.save(path)
        restored = KnowledgeIndex.load(path)

        assert restored.model == index.model
        assert restored.dimension == index.dimension
        assert restored.built_at == index.built_at
        assert len(restored.chunks) == len(index.chunks)
        assert restored.chunks[0].title == index.chunks[0].title
        assert restored.chunks[0].embedding == index.chunks[0].embedding
        assert restored.chunks[0].tags == index.chunks[0].tags

    def test_saved_json_is_utf8_and_versioned(self, tmp_path) -> None:
        path = tmp_path / "index.json"
        KnowledgeIndex(chunks=[], model="m").save(path)

        data = json.loads(path.read_text(encoding="utf-8"))

        assert data["version"] == 1
        assert data["chunks"] == []

    def test_load_rejects_non_object_payload(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("[]", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid knowledge index"):
            KnowledgeIndex.load(path)

    def test_chunk_to_dict_omits_absent_embedding(self) -> None:
        chunk = KnowledgeChunk(id="a#0", doc_id="a", title="T", content="C")

        assert "embedding" not in chunk.to_dict()


class TestShippedCorpusBuild:
    @pytest.mark.asyncio
    async def test_real_corpus_builds(self) -> None:
        index = await build_index(
            DEFAULT_DATA_DIR, FakeEmbeddingService(), model="fake-model"
        )

        assert len(index.chunks) > 20
