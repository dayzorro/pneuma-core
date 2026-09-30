"""博查（Bocha）AI Search 联网检索客户端。

接口文档：https://bocha-ai.feishu.cn/wiki/HmtOw1z6vik14Fkdu5uc9VaInBb

    POST {base_url}{endpoint}
    Authorization: Bearer <API_KEY>
    Content-Type: application/json

    {
      "query": "杭州天气",
      "freshness": "noLimit",
      "count": 10,
      "answer": false,
      "stream": false
    }

返回体以 Bing Search API 的格式为基底：网页结果在 ``webPages.value`` 里
（name / url / snippet / summary / siteName / siteIcon / datePublished），
AI Search 还会额外返回垂直领域模态卡与大模型总结。

说明：``/v1/ai-search`` 可能需要单独开通权限；若返回 401 且提示
「无接口调用权限」，把 ``PNEUMA_BOCHA_ENDPOINT`` 换成 ``/v1/web-search`` 即可。
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from pneuma_core.websearch.models import WebSearchResponse, WebSearchResult

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.bocha.cn"
DEFAULT_ENDPOINT = "/v1/ai-search"
DEFAULT_COUNT = 8
DEFAULT_FRESHNESS = "noLimit"
DEFAULT_TIMEOUT = 20.0

# freshness 允许的取值
_VALID_FRESHNESS = frozenset(
    {"noLimit", "oneDay", "oneWeek", "oneMonth", "oneYear"}
)


class WebSearchError(RuntimeError):
    """联网检索失败（网络、鉴权、配额、权限等）。"""


def _extract_answer(data: dict[str, Any]) -> str | None:
    """从返回体里提取大模型总结（不同版本字段名不一致，做兼容）。"""
    for key in ("answer", "summary", "aiAnswer"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            text = value.get("text") or value.get("content")
            if isinstance(text, str) and text.strip():
                return text.strip()

    # 部分版本放在 messages 里
    messages = data.get("messages")
    if isinstance(messages, list):
        for item in reversed(messages):
            if not isinstance(item, dict):
                continue
            if item.get("role") not in ("assistant", None):
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _extract_cards(data: dict[str, Any]) -> list[dict[str, Any]]:
    """提取模态卡（天气卡/百科卡等），字段名做兼容。"""
    for key in ("modalityCards", "modality_cards", "cards"):
        value = data.get(key)
        if isinstance(value, list):
            return [c for c in value if isinstance(c, dict)]
    return []


class BochaSearchClient:
    """博查 AI Search 客户端。"""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        endpoint: str = DEFAULT_ENDPOINT,
        count: int = DEFAULT_COUNT,
        freshness: str = DEFAULT_FRESHNESS,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Bocha API key is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self._count = max(1, min(50, count))
        self._freshness = freshness
        self._timeout = timeout
        self._client = http_client
        self._owns_client = http_client is None

    @classmethod
    def from_env(cls) -> BochaSearchClient | None:
        """从环境变量构建。未配置 API Key 时返回 None（= 关闭联网检索）。"""
        api_key = os.environ.get("PNEUMA_BOCHA_API_KEY", "").strip()
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            base_url=os.environ.get("PNEUMA_BOCHA_BASE_URL", DEFAULT_BASE_URL),
            endpoint=os.environ.get("PNEUMA_BOCHA_ENDPOINT", DEFAULT_ENDPOINT),
            count=int(os.environ.get("PNEUMA_BOCHA_COUNT", str(DEFAULT_COUNT))),
            freshness=os.environ.get("PNEUMA_BOCHA_FRESHNESS", DEFAULT_FRESHNESS),
            timeout=float(os.environ.get("PNEUMA_BOCHA_TIMEOUT", str(DEFAULT_TIMEOUT))),
        )

    @property
    def endpoint_url(self) -> str:
        return f"{self._base_url}{self._endpoint}"

    async def search(
        self,
        query: str,
        *,
        count: int | None = None,
        freshness: str | None = None,
    ) -> WebSearchResponse:
        """执行一次联网检索。

        Raises:
            WebSearchError: 网络异常、非 2xx 响应或返回体无法解析。
        """
        query = (query or "").strip()
        if not query:
            return WebSearchResponse(query="")

        payload = {
            "query": query,
            "freshness": freshness or self._freshness,
            "count": count or self._count,
            # 前台只需要参考源，不需要博查再套一层大模型总结（省时省钱）
            "answer": False,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await self._request(headers, payload)
        except httpx.TimeoutException as e:
            raise WebSearchError(f"联网检索超时：{e}") from e
        except httpx.HTTPError as e:
            raise WebSearchError(f"联网检索请求失败：{e}") from e

        if response.status_code != 200:
            raise self._build_error(response)

        try:
            data = response.json()
        except ValueError as e:
            raise WebSearchError(f"联网检索返回体不是合法 JSON：{e}") from e
        if not isinstance(data, dict):
            raise WebSearchError("联网检索返回体结构异常")

        return self._parse(query, data)

    async def _request(
        self, headers: dict[str, str], payload: dict[str, Any]
    ) -> httpx.Response:
        if self._client is not None:
            return await self._client.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await client.post(
                self.endpoint_url, headers=headers, json=payload
            )

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _build_error(response: httpx.Response) -> WebSearchError:
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                detail = str(
                    body.get("message")
                    or body.get("msg")
                    or (body.get("error") or {}).get("message")
                    or ""
                )
        except ValueError:
            detail = response.text[:200]

        hint = ""
        if response.status_code == 401 and "无接口调用权限" in detail:
            hint = (
                "（当前 Key 未开通 AI Search 权限，"
                "可把 PNEUMA_BOCHA_ENDPOINT 改为 /v1/web-search）"
            )
        return WebSearchError(
            f"联网检索返回 {response.status_code}：{detail or '无详细信息'}{hint}"
        )

    @staticmethod
    def _parse(query: str, data: dict[str, Any]) -> WebSearchResponse:
        web_pages = data.get("webPages")
        raw_values: list[Any] = []
        if isinstance(web_pages, dict):
            value = web_pages.get("value")
            if isinstance(value, list):
                raw_values = value

        results = [
            WebSearchResult.from_bocha(item)
            for item in raw_values
            if isinstance(item, dict)
        ]
        results = [r for r in results if r.title or r.url]

        return WebSearchResponse(
            query=query,
            results=results,
            cards=_extract_cards(data),
            answer=_extract_answer(data),
        )


__all__ = [
    "BochaSearchClient",
    "DEFAULT_BASE_URL",
    "DEFAULT_COUNT",
    "DEFAULT_ENDPOINT",
    "DEFAULT_FRESHNESS",
    "WebSearchError",
]
