"""Tests for 知识库文档切分（chunker）。"""

from __future__ import annotations

from pathlib import Path

from pneuma_core.knowledge import DEFAULT_DATA_DIR
from pneuma_core.knowledge.chunker import parse_front_matter, split_markdown

_DOC = """\
---
doc_id: membership
title: 会员与积分
source: 官网
tags: 会员, 积分
---

# 会员与积分

## 如何办理会员

带有效证件到门店服务台填写申请表即可。

## 积分怎么算

消费 1 元积 1 分。
"""


class TestParseFrontMatter:
    def test_parses_metadata_and_strips_block(self) -> None:
        meta, body = parse_front_matter(_DOC, fallback_doc_id="fallback")

        assert meta.doc_id == "membership"
        assert meta.title == "会员与积分"
        assert meta.source == "官网"
        assert meta.tags == ("会员", "积分")
        assert not body.startswith("---")
        assert body.lstrip().startswith("# 会员与积分")

    def test_without_front_matter_uses_fallback(self) -> None:
        meta, body = parse_front_matter("# 标题\n正文", fallback_doc_id="doc-x")

        assert meta.doc_id == "doc-x"
        assert meta.title == "doc-x"
        assert meta.source == ""
        assert body.startswith("# 标题")


class TestSplitMarkdown:
    def test_splits_by_second_level_heading(self) -> None:
        chunks = split_markdown(_DOC, fallback_doc_id="doc")

        titles = [c.title for c in chunks]
        assert titles == [
            "会员与积分 · 如何办理会员",
            "会员与积分 · 积分怎么算",
        ]
        assert all(c.doc_id == "membership" for c in chunks)
        assert chunks[0].content == "带有效证件到门店服务台填写申请表即可。"
        assert chunks[0].tags == ("会员", "积分")

    def test_ids_are_sequential_and_unique(self) -> None:
        chunks = split_markdown(_DOC, fallback_doc_id="doc")

        assert [c.id for c in chunks] == ["membership#0", "membership#1"]

    def test_preamble_becomes_chunk_titled_by_document(self) -> None:
        chunks = split_markdown(
            "---\ndoc_id: d\ntitle: 文档\n---\n\n开门见山的开场白。\n\n## 小节\n正文\n",
            fallback_doc_id="d",
        )

        assert chunks[0].title == "文档"
        assert chunks[0].content == "开门见山的开场白。"
        assert chunks[1].title == "文档 · 小节"

    def test_long_section_is_split_into_multiple_chunks(self) -> None:
        paragraph = "甲" * 80
        body = "\n\n".join([paragraph] * 6)

        chunks = split_markdown(
            f"---\ndoc_id: d\ntitle: 长文\n---\n\n## 小节\n{body}\n",
            fallback_doc_id="d",
            max_chars=200,
        )

        assert len(chunks) > 1
        assert all(len(c.content) <= 250 for c in chunks)
        assert all(c.title == "长文 · 小节" for c in chunks)
        # 内容不丢失
        joined = "".join(c.content for c in chunks).replace("\n", "")
        assert joined.count("甲") == 80 * 6

    def test_single_over_long_paragraph_is_hard_wrapped(self) -> None:
        lines = "\n".join("乙" * 60 for _ in range(5))

        chunks = split_markdown(
            f"---\ndoc_id: d\ntitle: 长段落\n---\n\n## 小节\n{lines}\n",
            fallback_doc_id="d",
            max_chars=100,
        )

        assert len(chunks) >= 3

    def test_h1_only_preamble_is_dropped(self) -> None:
        chunks = split_markdown(
            "---\ndoc_id: d\ntitle: 文档\n---\n\n# 文档\n\n## 小节\n正文\n",
            fallback_doc_id="d",
        )

        assert [c.title for c in chunks] == ["文档 · 小节"]

    def test_empty_document_returns_no_chunks(self) -> None:
        assert split_markdown("---\ndoc_id: d\n---\n", fallback_doc_id="d") == []


class TestShippedCorpus:
    """随包分发的华润万家语料必须始终可切分。"""

    def test_corpus_loads(self) -> None:
        from pneuma_core.knowledge import load_documents

        chunks = load_documents(DEFAULT_DATA_DIR)

        assert len(chunks) > 20
        assert all(c.title and c.content for c in chunks)
        doc_ids = {c.doc_id for c in chunks}
        assert {"membership", "returns", "faq"} <= doc_ids

    def test_corpus_directory_exists(self) -> None:
        assert DEFAULT_DATA_DIR.is_dir()
        assert any(Path(DEFAULT_DATA_DIR).glob("*.md"))
