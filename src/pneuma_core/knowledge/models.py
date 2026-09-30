"""知识库数据模型：KnowledgeChunk / KnowledgeHit。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KnowledgeChunk:
    """知识库中的最小检索单元。

    title: 所属小节标题（形如「会员制度 · 积分规则」），用于展示与检索加权
    content: 小节正文
    embedding: 向量（None = 尚未生成，此时只能走关键词检索）
    """

    id: str
    doc_id: str
    title: str
    content: str
    source: str = ""
    tags: tuple[str, ...] = ()
    embedding: list[float] | None = None

    def full_text(self) -> str:
        """用于生成向量 / 关键词匹配的文本（标题 + 正文）。"""
        return f"{self.title}\n{self.content}".strip()

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "tags": list(self.tags),
        }
        if self.embedding is not None:
            data["embedding"] = self.embedding
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeChunk:
        return cls(
            id=str(data["id"]),
            doc_id=str(data.get("doc_id", "")),
            title=str(data.get("title", "")),
            content=str(data.get("content", "")),
            source=str(data.get("source", "")),
            tags=tuple(str(t) for t in data.get("tags", [])),
            embedding=list(data["embedding"]) if data.get("embedding") else None,
        )


@dataclass(frozen=True)
class KnowledgeHit:
    """一次检索命中的知识块。

    score: 归一化后的相关度（向量检索为余弦相似度，关键词检索为覆盖率）
    matched_by: "vector" 或 "keyword"
    """

    chunk: KnowledgeChunk
    score: float
    matched_by: str = "vector"


@dataclass(frozen=True)
class DocumentMeta:
    """知识文档的 front matter 元数据。"""

    doc_id: str
    title: str
    source: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
