"""联网检索的数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = ["WebSearchClient", "WebSearchResponse", "WebSearchResult"]


@dataclass(frozen=True)
class WebSearchResult:
    """一条联网检索结果（对应博查的 webPages.value 元素）。"""

    title: str
    url: str
    snippet: str = ""
    summary: str = ""
    site_name: str = ""
    published_at: str = ""

    def best_text(self) -> str:
        """优先用摘要（信息更完整），没有摘要时退回片段。"""
        return (self.summary or self.snippet).strip()

    @classmethod
    def from_bocha(cls, raw: dict[str, Any]) -> WebSearchResult:
        return cls(
            title=str(raw.get("name") or raw.get("title") or "").strip(),
            url=str(raw.get("url") or "").strip(),
            snippet=str(raw.get("snippet") or "").strip(),
            summary=str(raw.get("summary") or "").strip(),
            site_name=str(raw.get("siteName") or "").strip(),
            published_at=str(raw.get("datePublished") or "").strip(),
        )


@dataclass(frozen=True)
class WebSearchResponse:
    """一次联网检索的完整返回。"""

    query: str
    results: list[WebSearchResult] = field(default_factory=list)
    cards: list[dict[str, Any]] = field(default_factory=list)
    answer: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.results and not self.cards and not self.answer

    def sources(self, limit: int = 5) -> list[dict[str, str]]:
        """给上层（API / UI）用的来源列表。"""
        return [
            {"title": r.title, "url": r.url, "site": r.site_name}
            for r in self.results[:limit]
        ]


class WebSearchClient(Protocol):
    """联网检索客户端协议（便于替换供应商与测试打桩）。"""

    async def search(self, query: str) -> WebSearchResponse: ...
