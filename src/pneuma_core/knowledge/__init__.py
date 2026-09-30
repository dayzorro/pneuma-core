"""本地知识库：文档切块 → 向量索引 → 检索 → 注入提示词。

设计目标是「专业知识问答」：让角色在回答前先检索本地资料，只依据资料作答，
从而避免编造。与长期记忆（memory）不同，知识库是**只读**的、与角色无关的
静态资料，不随对话变化。
"""

from pneuma_core.knowledge.chunker import DEFAULT_MAX_CHARS, split_markdown
from pneuma_core.knowledge.index import (
    DEFAULT_DATA_DIR,
    KnowledgeIndex,
    build_index,
    load_documents,
)
from pneuma_core.knowledge.models import DocumentMeta, KnowledgeChunk, KnowledgeHit
from pneuma_core.knowledge.retriever import (
    DEFAULT_TOP_K,
    KnowledgeBase,
)

__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_TOP_K",
    "DocumentMeta",
    "KnowledgeBase",
    "KnowledgeChunk",
    "KnowledgeHit",
    "KnowledgeIndex",
    "build_index",
    "load_documents",
    "split_markdown",
]
