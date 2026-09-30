"""知识库索引：把 data 目录下的 Markdown 文档切块、向量化并落盘。

索引是一个自包含的 JSON 文件，包含知识块原文与向量，服务启动时直接加载，
无需在启动阶段调用 embedding 接口。

构建（离线，需要 embedding 接口）::

    python scripts/build_knowledge_index.py

若跳过构建，服务会退化为「仅关键词检索」模式（见 retriever.KnowledgeBase）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pneuma_core.knowledge.chunker import DEFAULT_MAX_CHARS, split_markdown
from pneuma_core.knowledge.models import KnowledgeChunk
from pneuma_core.protocols.embedding import EmbeddingService

logger = logging.getLogger(__name__)

INDEX_VERSION = 1

# 单次 embedding 请求的最大条数（多数 OpenAI 兼容端点上限为 25）
DEFAULT_BATCH_SIZE = 25

# 随包分发的示例知识库语料（华润万家）
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "huarun"

# 忽略以下划线开头的文档（用于放说明文件）
_IGNORED_PREFIX = "_"


@dataclass(frozen=True)
class KnowledgeIndex:
    """知识块集合 + 构建元信息。"""

    chunks: list[KnowledgeChunk]
    model: str = ""
    dimension: int = 0
    built_at: str = ""

    @property
    def has_embeddings(self) -> bool:
        return any(chunk.embedding for chunk in self.chunks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": INDEX_VERSION,
            "model": self.model,
            "dimension": self.dimension,
            "built_at": self.built_at,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeIndex:
        return cls(
            chunks=[KnowledgeChunk.from_dict(c) for c in data.get("chunks", [])],
            model=str(data.get("model", "")),
            dimension=int(data.get("dimension", 0)),
            built_at=str(data.get("built_at", "")),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> KnowledgeIndex:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid knowledge index: {path}")
        return cls.from_dict(data)


def load_documents(
    docs_dir: Path, *, max_chars: int = DEFAULT_MAX_CHARS
) -> list[KnowledgeChunk]:
    """把目录下的 ``*.md`` 切块（不做向量化）。"""
    if not docs_dir.is_dir():
        return []

    chunks: list[KnowledgeChunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.startswith(_IGNORED_PREFIX):
            continue
        text = path.read_text(encoding="utf-8")
        chunks.extend(
            split_markdown(
                text, fallback_doc_id=path.stem, max_chars=max_chars
            )
        )
    return chunks


async def build_index(
    docs_dir: Path,
    embedding_service: EmbeddingService,
    *,
    model: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> KnowledgeIndex:
    """读取文档、切块并调用 embedding 接口生成向量索引。

    Args:
        batch_size: 单次请求的最大条数。多数 OpenAI 兼容端点对 batch 有上限
            （例如阿里云百炼为 25），超过会直接报 400。

    Raises:
        ValueError: 目录下没有任何可用文档，或接口没有返回向量时。
    """
    chunks = load_documents(docs_dir, max_chars=max_chars)
    if not chunks:
        raise ValueError(f"No knowledge documents found in {docs_dir}")

    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors.extend(
            await embedding_service.embed_batch([c.full_text() for c in batch])
        )

    embedded: list[KnowledgeChunk] = []
    dimension = 0
    for chunk, vector in zip(chunks, vectors):
        if vector:
            dimension = dimension or len(vector)
        embedded.append(replace(chunk, embedding=vector or None))

    if dimension == 0:
        raise ValueError("Embedding service returned empty vectors")

    return KnowledgeIndex(
        chunks=embedded,
        model=model,
        dimension=dimension,
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
