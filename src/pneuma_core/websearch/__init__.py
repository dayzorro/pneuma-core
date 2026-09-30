"""联网检索：把外部实时信息接进对话。

与前缀知识库（``pneuma_core.knowledge``）分工不同：
    - knowledge：只读的、人工整理的资料，用于「专业知识问答」
    - websearch：实时的外部信息，用于「涉及联网信息」的问题

检索到的结果一方面当轮注入提示词，另一方面会被异步提炼成行业通识认知块，
沉淀进认知库（``pneuma_core.knowledge.insights``）供后续检索复用。
"""

from pneuma_core.websearch.bocha import (
    DEFAULT_BASE_URL,
    DEFAULT_COUNT,
    DEFAULT_ENDPOINT,
    DEFAULT_FRESHNESS,
    BochaSearchClient,
    WebSearchError,
)
from pneuma_core.websearch.models import (
    WebSearchClient,
    WebSearchResponse,
    WebSearchResult,
)
from pneuma_core.websearch.policy import (
    MODE_ALWAYS,
    MODE_AUTO,
    MODE_OFF,
    is_time_sensitive,
    normalize_mode,
    should_search_online,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_COUNT",
    "DEFAULT_ENDPOINT",
    "DEFAULT_FRESHNESS",
    "MODE_ALWAYS",
    "MODE_AUTO",
    "MODE_OFF",
    "BochaSearchClient",
    "WebSearchClient",
    "WebSearchError",
    "WebSearchResponse",
    "WebSearchResult",
    "is_time_sensitive",
    "normalize_mode",
    "should_search_online",
]
