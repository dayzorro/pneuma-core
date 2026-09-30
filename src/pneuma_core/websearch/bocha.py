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

**两个端点的返回结构不一样**，解析层做了统一（见 ``_parse``）：

``/v1/web-search``（Bing 兼容，结果套在 ``data`` 里）::

    {"code":200,"data":{"webPages":{"value":[{name,url,snippet,summary,siteName,...}]}}}

``/v1/ai-search``（网页被塞进 ``messages`` 里，content 是字符串化的 JSON）::

    {"code":200,"messages":[
        {"type":"source","content_type":"webpage","content":"{\"value\":[...]}"},
        {"type":"source","content_type":"image","content":"{\"value\":[...]}"},
        {"type":"answer","content_type":"text","content":"大模型总结"},
        {"type":"follow_up","content_type":"text","content":"[\"追问1\"]"}
    ]}

AI Search 需要单独开通权限；若返回 401 且提示「无接口调用权限」，
把 ``PNEUMA_BOCHA_ENDPOINT`` 换成 ``/v1/web-search`` 即可。
"""

from __future__ import annotations

import json
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
# 是否让博查再返回一份大模型总结（多一次其侧调用；默认关，由前台自己综合）
DEFAULT_ANSWER = False

# 结果条数上限（接口允许 1〜50）
MAX_RESULTS = 50

# freshness 允许的取值
_VALID_FRESHNESS = frozenset(
    {"noLimit", "oneDay", "oneWeek", "oneMonth", "oneYear"}
)


class WebSearchError(RuntimeError):
    """联网检索失败（网络、鉴权、配额、权限等）。"""


def _maybe_json(value: Any) -> Any:
    """博查 messages 里的 content 是字符串化的 JSON，这里做兼容解析。"""
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _pages_from_web_pages(node: Any) -> list[WebSearchResult]:
    """从 ``webPages``（或只含 ``value`` 的同形结构）里取出网页结果。"""
    if not isinstance(node, dict):
        return []
    value = node.get("value")
    if not isinstance(value, list):
        return []

    results: list[WebSearchResult] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        result = WebSearchResult.from_bocha(item)
        if result.title or result.url:
            results.append(result)
    return results


def _dedupe(results: list[WebSearchResult]) -> list[WebSearchResult]:
    """按 URL 去重，保持原有顺序。"""
    seen: set[str] = set()
    unique: list[WebSearchResult] = []
    for result in results:
        key = result.url or result.title
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def _extract_answer(data: dict[str, Any]) -> str | None:
    """从返回体里提取大模型总结（不同版本字段名不一致，做兼容）。"""
    for key in ("answer", "aiAnswer", "summary"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            text = value.get("text") or value.get("content")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return None


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
        answer: bool = DEFAULT_ANSWER,
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
        self._answer = answer
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
            answer=os.environ.get("PNEUMA_BOCHA_ANSWER", "").strip().lower()
            in {"1", "true", "yes", "on"},
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
            "answer": self._answer,
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

        code = data.get("code")
        if isinstance(code, int) and code != 200:
            message = data.get("msg") or data.get("message") or "无详细信息"
            raise WebSearchError(f"联网检索业务错误码 {code}：{message}")

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
        """把两种端点的返回体统一成 WebSearchResponse。"""
        results: list[WebSearchResult] = []
        cards: list[dict[str, Any]] = []
        answer: str | None = None

        # web-search 把结果套在 data 里；ai-search 直接放在顶层
        body = data.get("data") if isinstance(data.get("data"), dict) else data

        results.extend(_pages_from_web_pages(body.get("webPages")))
        images = body.get("images")
        if isinstance(images, dict) and images.get("value"):
            cards.append({"content_type": "image", "value": images["value"]})

        # ai-search：网页 / 图片 / 总结 / 追问都在 messages 里
        messages = data.get("messages")
        if isinstance(messages, list):
            for item in messages:
                if not isinstance(item, dict):
                    continue
                kind = item.get("type")
                content = item.get("content")

                if kind == "source":
                    payload = _maybe_json(content)
                    if not isinstance(payload, dict):
                        continue
                    if item.get("content_type") == "webpage":
                        results.extend(_pages_from_web_pages(payload))
                    elif payload.get("value"):
                        cards.append(
                            {
                                "content_type": item.get("content_type"),
                                "value": payload["value"],
                            }
                        )
                elif kind == "answer":
                    text = content if isinstance(content, str) else None
                    if text and text.strip():
                        answer = answer or text.strip()
                elif kind == "follow_up":
                    questions = _maybe_json(content)
                    if isinstance(questions, list) and questions:
                        cards.append(
                            {"content_type": "follow_up", "value": questions}
                        )

        # 兜底：有的版本把总结放在顶层
        if answer is None:
            answer = _extract_answer(data)

        results = _dedupe(results)[:MAX_RESULTS]

        return WebSearchResponse(
            query=query, results=results, cards=cards, answer=answer
        )


__all__ = [
    "BochaSearchClient",
    "DEFAULT_BASE_URL",
    "DEFAULT_COUNT",
    "DEFAULT_ENDPOINT",
    "DEFAULT_FRESHNESS",
    "WebSearchError",
]
