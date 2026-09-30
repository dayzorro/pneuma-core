"""Tests for 认知提炼：联网结果 → 行业通识认知块。"""

from __future__ import annotations

import hashlib
import json
import random
from unittest.mock import AsyncMock

import pytest

from pneuma_core.knowledge.insights import InsightStore
from pneuma_core.runtime.insight_acquirer import (
    InsightAcquirer,
    _parse_insights,
)
from pneuma_core.websearch.models import WebSearchResponse, WebSearchResult


class FakeEmbeddingService:
    """每条文本得到各自独立的稳定向量（同文本同向量），便于验证去重行为。"""

    @staticmethod
    def _vector(text: str) -> list[float]:
        rng = random.Random(hashlib.sha256(text.encode()).hexdigest())
        return [rng.uniform(-1.0, 1.0) for _ in range(32)]

    async def embed(self, text: str) -> list[float]:
        return self._vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


def _response() -> WebSearchResponse:
    return WebSearchResponse(
        query="最近零售行业有什么新动向",
        results=[
            WebSearchResult(
                title="零售行业周报",
                url="https://example.com/a",
                site_name="示例网",
                summary="本周多家商超调整门店业态，社区店占比提升。",
                published_at="2026-09-28T00:00:00+08:00",
            ),
            WebSearchResult(
                title="消费者权益观察",
                url="https://example.com/b",
                snippet="线下无理由退货属于商家自愿承诺。",
            ),
        ],
    )


def _llm(content: str) -> AsyncMock:
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=AsyncMock(content=content))
    return llm


def _make_acquirer(tmp_path, llm, **kwargs) -> tuple[InsightAcquirer, InsightStore]:
    store = InsightStore(tmp_path / "i.json", embedding_service=FakeEmbeddingService())
    acquirer = InsightAcquirer(llm=llm, store=store, **kwargs)
    return acquirer, store


class TestParseInsights:
    def test_parses_plain_json(self) -> None:
        raw = json.dumps(
            {"insights": [{"topic": "行业动向", "content": "社区店占比提升。"}]},
            ensure_ascii=False,
        )

        assert _parse_insights(raw) == [
            {"topic": "行业动向", "content": "社区店占比提升。"}
        ]

    def test_parses_code_fenced_json(self) -> None:
        raw = '```json\n{"insights": [{"content": "内容"}]}\n```'

        assert _parse_insights(raw) == [{"topic": "", "content": "内容"}]

    def test_parses_json_embedded_in_prose(self) -> None:
        raw = '好的，结果如下：{"insights": [{"content": "内容"}]} 以上。'

        assert _parse_insights(raw) == [{"topic": "", "content": "内容"}]

    def test_accepts_bare_strings(self) -> None:
        raw = json.dumps({"insights": ["纯字符串认知"]}, ensure_ascii=False)

        assert _parse_insights(raw) == [{"topic": "", "content": "纯字符串认知"}]

    def test_skips_blank_contents(self) -> None:
        raw = json.dumps({"insights": [{"content": "  "}, {"content": "有效"}]})

        assert _parse_insights(raw) == [{"topic": "", "content": "有效"}]

    @pytest.mark.parametrize(
        "raw",
        ["", "   ", "not json", '{"insights": "nope"}', "[]", '{"other": 1}'],
    )
    def test_malformed_returns_empty(self, raw: str) -> None:
        assert _parse_insights(raw) == []


class TestBuildMaterial:
    def test_prefers_summary_and_includes_meta(self) -> None:
        material = InsightAcquirer._build_material(_response())

        assert "零售行业周报" in material
        assert "2026-09-28" in material
        assert "社区店占比提升" in material
        # 第二条没有 summary，用 snippet
        assert "线下无理由退货属于商家自愿承诺" in material

    def test_skips_results_without_text(self) -> None:
        response = WebSearchResponse(
            query="q", results=[WebSearchResult(title="只有标题", url="u")]
        )

        assert InsightAcquirer._build_material(response) == ""


class TestAcquire:
    @pytest.mark.asyncio
    async def test_extracts_and_stores_insights(self, tmp_path) -> None:
        payload = json.dumps(
            {
                "insights": [
                    {"topic": "行业动向", "content": "商超社区店占比持续提升。"},
                    {"topic": "服务规范", "content": "退货核对凭证是通行做法。"},
                ]
            },
            ensure_ascii=False,
        )
        acquirer, store = _make_acquirer(tmp_path, _llm(payload))

        added = await acquirer.acquire("最近零售行业有什么新动向", _response())

        assert added == 2
        assert store.size == 2
        assert store.all()[0].topic == "行业动向"
        assert store.all()[0].source_urls == (
            "https://example.com/a",
            "https://example.com/b",
        )
        assert store.all()[0].created_at

    @pytest.mark.asyncio
    async def test_prompt_forbids_company_specific_and_private_content(
        self, tmp_path
    ) -> None:
        llm = _llm('{"insights": []}')
        acquirer, _ = _make_acquirer(tmp_path, llm)

        await acquirer.acquire("q", _response())

        system_prompt = llm.generate.call_args[0][0].system_prompt
        assert "与任何一家具体公司无关" in system_prompt
        assert "个人隐私" in system_prompt
        assert "必须剔除" in system_prompt

    @pytest.mark.asyncio
    async def test_user_message_carries_query_and_material(self, tmp_path) -> None:
        llm = _llm('{"insights": []}')
        acquirer, _ = _make_acquirer(tmp_path, llm)

        await acquirer.acquire("最近零售行业有什么新动向", _response())

        user_message = llm.generate.call_args[0][0].messages[0]["content"]
        assert "最近零售行业有什么新动向" in user_message
        assert "零售行业周报" in user_message

    @pytest.mark.asyncio
    async def test_max_insights_is_respected(self, tmp_path) -> None:
        payload = json.dumps(
            {"insights": [{"content": f"认知{i}"} for i in range(10)]},
            ensure_ascii=False,
        )
        acquirer, store = _make_acquirer(tmp_path, _llm(payload), max_insights=3)

        added = await acquirer.acquire("q", _response())

        assert added == 3
        assert store.size == 3

    @pytest.mark.asyncio
    async def test_empty_material_skips_llm_call(self, tmp_path) -> None:
        llm = _llm("{}")
        acquirer, store = _make_acquirer(tmp_path, llm)

        added = await acquirer.acquire(
            "q", WebSearchResponse(query="q", results=[])
        )

        assert added == 0
        assert llm.generate.await_count == 0
        assert store.size == 0

    @pytest.mark.asyncio
    async def test_llm_failure_is_swallowed(self, tmp_path) -> None:
        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=RuntimeError("llm down"))
        acquirer, store = _make_acquirer(tmp_path, llm)

        added = await acquirer.acquire("q", _response())

        assert added == 0
        assert store.size == 0

    @pytest.mark.asyncio
    async def test_store_failure_is_swallowed(self, tmp_path) -> None:
        payload = json.dumps({"insights": [{"content": "认知"}]}, ensure_ascii=False)
        acquirer, store = _make_acquirer(tmp_path, _llm(payload))
        store.add_many = AsyncMock(side_effect=OSError("disk full"))  # type: ignore[method-assign]

        assert await acquirer.acquire("q", _response()) == 0

    @pytest.mark.asyncio
    async def test_no_insights_means_nothing_stored(self, tmp_path) -> None:
        acquirer, store = _make_acquirer(tmp_path, _llm('{"insights": []}'))

        assert await acquirer.acquire("q", _response()) == 0
        assert store.size == 0

    @pytest.mark.asyncio
    async def test_long_material_is_truncated(self, tmp_path) -> None:
        llm = _llm('{"insights": []}')
        acquirer, _ = _make_acquirer(tmp_path, llm, max_material_chars=50)
        response = WebSearchResponse(
            query="q",
            results=[
                WebSearchResult(
                    title="很长的标题", url="u", summary="占" * 500
                )
            ],
        )

        await acquirer.acquire("q", response)

        user_message = llm.generate.call_args[0][0].messages[0]["content"]
        assert len(user_message) < 300
