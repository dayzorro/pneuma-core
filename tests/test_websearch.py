"""Tests for 博查联网检索客户端与联网时机策略。"""

from __future__ import annotations

import json

import httpx
import pytest

from pneuma_core.websearch import (
    MODE_ALWAYS,
    MODE_AUTO,
    MODE_OFF,
    BochaSearchClient,
    WebSearchError,
    WebSearchResponse,
    WebSearchResult,
    is_time_sensitive,
    normalize_mode,
    should_search_online,
)

# 真实报文 A：/v1/web-search —— Bing 兼容，结果套在 data 里
_SAMPLE = {
    "code": 200,
    "msg": None,
    "data": {
        "_type": "SearchResponse",
        "queryContext": {"originalQuery": "最近零售行业有什么新动向"},
        "webPages": {
            "webSearchUrl": "https://bochaai.com/search?q=...",
            "totalEstimatedMatches": 1234,
            "value": [
                {
                    "name": "零售行业周报",
                    "url": "https://example.com/a",
                    "siteName": "示例网",
                    "snippet": "简短片段",
                    "summary": "更完整的摘要内容",
                    "datePublished": "2026-09-28T00:00:00+08:00",
                },
                {
                    "name": "第二条",
                    "url": "https://example.com/b",
                    "snippet": "只有片段",
                },
                {"name": "无链接的脏数据"},
            ],
        },
        "images": {"value": [{"name": "图一", "contentUrl": "https://img/1.png"}]},
    },
}


def _ai_search_sample(*, with_answer: bool = True) -> dict:
    """真实报文 B：/v1/ai-search —— 内容都在 messages 里，content 是字符串化 JSON。"""
    messages = [
        {
            "role": "assistant",
            "type": "source",
            "content_type": "webpage",
            "content": json.dumps(
                {
                    "webSearchUrl": "https://bochaai.com/search?q=...",
                    "value": [
                        {
                            "name": "零售行业周报",
                            "url": "https://example.com/a",
                            "siteName": "示例网",
                            "snippet": "简短片段",
                            "summary": "更完整的摘要内容",
                            "datePublished": "2026-09-28T00:00:00+08:00",
                        },
                        {
                            "name": "第二条",
                            "url": "https://example.com/b",
                            "snippet": "只有片段",
                        },
                        {"name": "无链接的脏数据"},
                    ],
                    "someResultsRemoved": False,
                },
                ensure_ascii=False,
            ),
        },
        {
            "role": "assistant",
            "type": "source",
            "content_type": "image",
            "content": json.dumps({"value": [{"name": "图一"}]}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "type": "source",
            "content_type": "video",
            "content": "{}",
        },
    ]
    if with_answer:
        messages.append(
            {
                "role": "assistant",
                "type": "answer",
                "content_type": "text",
                "content": "联网总结的结论",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "type": "follow_up",
                "content_type": "text",
                "content": json.dumps(["追问一", "追问二"], ensure_ascii=False),
            }
        )
    return {
        "code": 200,
        "log_id": "abc",
        "conversation_id": "conv",
        "messages": messages,
    }


def _make_client(handler) -> BochaSearchClient:
    transport = httpx.MockTransport(handler)
    return BochaSearchClient(
        "test-key", http_client=httpx.AsyncClient(transport=transport)
    )


class TestSearchRequest:
    @pytest.mark.asyncio
    async def test_sends_auth_header_and_payload(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("Authorization")
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_SAMPLE)

        client = _make_client(handler)
        await client.search("最近零售行业有什么新动向", count=5, freshness="oneWeek")

        assert seen["url"] == "https://api.bocha.cn/v1/ai-search"
        assert seen["auth"] == "Bearer test-key"
        assert seen["body"] == {
            "query": "最近零售行业有什么新动向",
            "freshness": "oneWeek",
            "count": 5,
            "answer": False,
            "stream": False,
        }

    @pytest.mark.asyncio
    async def test_custom_base_url_and_endpoint(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return httpx.Response(200, json=_SAMPLE)

        transport = httpx.MockTransport(handler)
        client = BochaSearchClient(
            "k",
            base_url="https://api.bocha.cn/",
            endpoint="v1/web-search",
            http_client=httpx.AsyncClient(transport=transport),
        )
        await client.search("x")

        assert seen["url"] == "https://api.bocha.cn/v1/web-search"

    @pytest.mark.asyncio
    async def test_answer_flag_is_forwarded(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_ai_search_sample())

        transport = httpx.MockTransport(handler)
        client = BochaSearchClient(
            "k", answer=True, http_client=httpx.AsyncClient(transport=transport)
        )
        await client.search("q")

        assert seen["body"]["answer"] is True

    @pytest.mark.asyncio
    async def test_blank_query_skips_request(self) -> None:
        called = False

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(200, json=_SAMPLE)

        client = _make_client(handler)
        response = await client.search("   ")

        assert response.is_empty
        assert called is False

    def test_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="API key"):
            BochaSearchClient("")


class TestSearchParsing:
    @pytest.mark.asyncio
    async def test_parses_web_pages(self) -> None:
        client = _make_client(lambda r: httpx.Response(200, json=_SAMPLE))

        response = await client.search("最近零售行业有什么新动向")

        assert response.query == "最近零售行业有什么新动向"
        assert len(response.results) == 3
        first = response.results[0]
        assert first.title == "零售行业周报"
        assert first.url == "https://example.com/a"
        assert first.site_name == "示例网"
        assert first.summary == "更完整的摘要内容"
        assert first.published_at.startswith("2026-09-28")

    @pytest.mark.asyncio
    async def test_prefers_summary_over_snippet(self) -> None:
        client = _make_client(lambda r: httpx.Response(200, json=_SAMPLE))

        response = await client.search("q")

        assert response.results[0].best_text() == "更完整的摘要内容"
        assert response.results[1].best_text() == "只有片段"

    @pytest.mark.asyncio
    async def test_web_search_wrapped_in_data(self) -> None:
        """真实报文 A：/v1/web-search 把结果包在 data 里。"""
        client = _make_client(lambda r: httpx.Response(200, json=_SAMPLE))

        response = await client.search("q")

        assert response.results[0].title == "零售行业周报"
        assert response.cards[0]["content_type"] == "image"

    @pytest.mark.asyncio
    async def test_sources_for_ui(self) -> None:
        client = _make_client(lambda r: httpx.Response(200, json=_SAMPLE))

        response = await client.search("q")

        assert response.sources(limit=1) == [
            {"title": "零售行业周报", "url": "https://example.com/a", "site": "示例网"}
        ]

    @pytest.mark.asyncio
    async def test_empty_response_is_flagged(self) -> None:
        client = _make_client(lambda r: httpx.Response(200, json={"webPages": {}}))

        response = await client.search("q")

        assert response.is_empty
        assert response.results == []

    @pytest.mark.asyncio
    async def test_extracts_answer_from_nested_dict(self) -> None:
        payload = {"webPages": {"value": []}, "answer": {"text": "总结内容"}}
        client = _make_client(lambda r: httpx.Response(200, json=payload))

        response = await client.search("q")

        assert response.answer == "总结内容"


class TestAiSearchParsing:
    """真实报文 B：/v1/ai-search 的内容全在 messages 里。

    这是实际踩过的坑：只认顶层 webPages 会让 ai-search 静默返回 0 条。
    """

    @pytest.mark.asyncio
    async def test_extracts_web_pages_from_messages(self) -> None:
        client = _make_client(
            lambda r: httpx.Response(200, json=_ai_search_sample())
        )

        response = await client.search("最近零售行业有什么新动向")

        assert response.query == "最近零售行业有什么新动向"
        # 3 条里有一条是只有标题的脏数据，同样保留下来
        assert len(response.results) == 3
        assert sum(1 for r in response.results if r.url) == 2
        assert response.results[0].title == "零售行业周报"
        assert response.results[0].url == "https://example.com/a"
        assert response.results[0].summary == "更完整的摘要内容"

    @pytest.mark.asyncio
    async def test_extracts_answer_and_follow_ups(self) -> None:
        client = _make_client(
            lambda r: httpx.Response(200, json=_ai_search_sample())
        )

        response = await client.search("q")

        assert response.answer == "联网总结的结论"
        kinds = [c.get("content_type") for c in response.cards]
        assert kinds == ["image", "follow_up"]

    @pytest.mark.asyncio
    async def test_without_answer_message(self) -> None:
        client = _make_client(
            lambda r: httpx.Response(
                200, json=_ai_search_sample(with_answer=False)
            )
        )

        response = await client.search("q")

        assert response.answer is None
        assert response.results  # 网页照常拿到

    @pytest.mark.asyncio
    async def test_dedupes_by_url(self) -> None:
        duplicated = {
            "name": "零售行业周报",
            "url": "https://example.com/a",
            "summary": "重复来源",
        }
        sample = _ai_search_sample()
        pages = json.loads(sample["messages"][0]["content"])
        pages["value"].append(duplicated)
        sample["messages"][0]["content"] = json.dumps(pages, ensure_ascii=False)
        client = _make_client(lambda r: httpx.Response(200, json=sample))

        response = await client.search("q")

        assert len(response.results) == 3
        assert (
            sum(1 for r in response.results if r.url == "https://example.com/a")
            == 1
        )

    @pytest.mark.asyncio
    async def test_tolerates_unparsable_message_content(self) -> None:
        payload = {
            "code": 200,
            "messages": [
                {"type": "source", "content_type": "webpage", "content": "not json"},
                {"type": "answer", "content": None},
                {"type": "follow_up", "content": "not json"},
                "脏数据",
            ],
        }
        client = _make_client(lambda r: httpx.Response(200, json=payload))

        response = await client.search("q")

        assert response.is_empty


class TestSearchErrors:
    @pytest.mark.asyncio
    async def test_body_error_code_raises(self) -> None:
        """HTTP 200 但业务码非 200（配额/权限）不能静默当成「没搜到」。"""
        payload = {"code": 403, "msg": "余额不足"}
        client = _make_client(lambda r: httpx.Response(200, json=payload))

        with pytest.raises(WebSearchError, match="403.*余额不足"):
            await client.search("q")

    @pytest.mark.asyncio
    async def test_non_200_raises_with_detail(self) -> None:
        client = _make_client(
            lambda r: httpx.Response(429, json={"message": "quota exceeded"})
        )

        with pytest.raises(WebSearchError, match="429.*quota exceeded"):
            await client.search("q")

    @pytest.mark.asyncio
    async def test_missing_permission_hints_at_endpoint_switch(self) -> None:
        client = _make_client(
            lambda r: httpx.Response(401, json={"message": "无接口调用权限"})
        )

        with pytest.raises(WebSearchError, match="web-search"):
            await client.search("q")

    @pytest.mark.asyncio
    async def test_invalid_json_raises(self) -> None:
        client = _make_client(
            lambda r: httpx.Response(
                200, content=b"not json", headers={"Content-Type": "application/json"}
            )
        )

        with pytest.raises(WebSearchError, match="JSON"):
            await client.search("q")

    @pytest.mark.asyncio
    async def test_network_error_is_wrapped(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        client = _make_client(handler)

        with pytest.raises(WebSearchError, match="请求失败"):
            await client.search("q")


class TestFromEnv:
    def test_returns_none_without_key(self, monkeypatch) -> None:
        monkeypatch.delenv("PNEUMA_BOCHA_API_KEY", raising=False)

        assert BochaSearchClient.from_env() is None

    def test_builds_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("PNEUMA_BOCHA_API_KEY", "  sk-test  ")
        monkeypatch.setenv("PNEUMA_BOCHA_ENDPOINT", "/v1/web-search")
        monkeypatch.setenv("PNEUMA_BOCHA_COUNT", "3")

        client = BochaSearchClient.from_env()

        assert client is not None
        assert client.endpoint_url == "https://api.bocha.cn/v1/web-search"

    def test_count_is_clamped(self) -> None:
        client = BochaSearchClient("k", count=999)

        assert client._count == 50


class TestResultModel:
    def test_best_text_falls_back_to_snippet(self) -> None:
        result = WebSearchResult(title="t", url="u", snippet="片段")

        assert result.best_text() == "片段"

    def test_from_bocha_tolerates_missing_fields(self) -> None:
        result = WebSearchResult.from_bocha({})

        assert result.title == ""
        assert result.url == ""
        assert result.best_text() == ""

    def test_answer_only_response_is_not_empty(self) -> None:
        assert WebSearchResponse(query="q", answer="有话").is_empty is False


class TestPolicy:
    @pytest.mark.parametrize(
        "query",
        [
            "今天天气怎么样",
            "最近有什么新闻",
            "华润万家最近有什么促销活动",
            "今年零售行业有什么新规",
            "茅台现在什么价格",
        ],
    )
    def test_time_sensitive_queries(self, query: str) -> None:
        assert is_time_sensitive(query) is True

    @pytest.mark.parametrize(
        "query",
        ["会员卡怎么办理", "能退货吗", "帮我写一首诗", "你是谁"],
    )
    def test_non_time_sensitive_queries(self, query: str) -> None:
        assert is_time_sensitive(query) is False

    def test_mode_off_never_searches(self) -> None:
        assert (
            should_search_online("今天天气", mode=MODE_OFF, has_local_hits=False)
            is False
        )

    def test_mode_always_always_searches(self) -> None:
        assert (
            should_search_online("会员卡怎么办理", mode=MODE_ALWAYS, has_local_hits=True)
            is True
        )

    def test_auto_skips_when_local_hits_exist(self) -> None:
        assert (
            should_search_online("最近的活动", mode=MODE_AUTO, has_local_hits=True)
            is False
        )

    def test_auto_searches_only_for_time_sensitive_gap(self) -> None:
        assert (
            should_search_online("最近的活动", mode=MODE_AUTO, has_local_hits=False)
            is True
        )
        assert (
            should_search_online("帮我写一首诗", mode=MODE_AUTO, has_local_hits=False)
            is False
        )

    def test_blank_query_never_searches(self) -> None:
        assert should_search_online("  ", mode=MODE_ALWAYS) is False

    def test_normalize_mode(self) -> None:
        assert normalize_mode(" ALWAYS ") == MODE_ALWAYS
        assert normalize_mode("whatever") == MODE_AUTO
        assert normalize_mode(None) == MODE_AUTO
        assert normalize_mode("off") == MODE_OFF
