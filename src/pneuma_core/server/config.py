"""Server configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pneuma_core.knowledge import DEFAULT_DATA_DIR as DEFAULT_KNOWLEDGE_DIR
from pneuma_core.websearch.bocha import (
    DEFAULT_ANSWER as DEFAULT_BOCHA_ANSWER,
    DEFAULT_BASE_URL as DEFAULT_BOCHA_BASE_URL,
    DEFAULT_COUNT as DEFAULT_BOCHA_COUNT,
    DEFAULT_ENDPOINT as DEFAULT_BOCHA_ENDPOINT,
    DEFAULT_FRESHNESS as DEFAULT_BOCHA_FRESHNESS,
)
from pneuma_core.websearch.policy import (
    DEFAULT_MODE as DEFAULT_WEB_SEARCH_MODE,
    normalize_mode,
)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8001
DEFAULT_CHARACTER_FILE = "examples/xiaorun-frontdesk.character.yaml"
DEFAULT_DB_PATH = "vault/pneuma.db"
DEFAULT_KNOWLEDGE_INDEX = "vault/knowledge_index.json"
DEFAULT_INSIGHT_STORE = "vault/insights.json"
DEFAULT_INSIGHT_MAX_PER_TURN = 5
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServerConfig:
    """Runtime configuration for the chat service.

    The LLM and embedding endpoints are both OpenAI-compatible. When the
    embedding base URL / API key are not set, they fall back to the LLM ones,
    which covers single-provider setups (e.g. Aliyun DashScope serving both).
    """

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    # 日志级别。设为 debug 可看到每轮的时延埋点明细。
    log_level: str = "info"
    character_file: Path = Path(DEFAULT_CHARACTER_FILE)
    db_path: Path = Path(DEFAULT_DB_PATH)
    llm_base_url: str | None = None
    llm_api_key: str = ""
    llm_model: str = DEFAULT_LLM_MODEL
    llm_timeout: float = 60.0
    # 思考开关的 extra_body 键名（例如 DashScope 的 "enable_thinking"）。
    # 为空时不干预模型思考行为，避免对不支持的供应商报 400；交互场景建议
    # 配成 enable_thinking 以真正关闭思考、降低首字延迟。
    llm_thinking_param: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str = ""
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    history_limit: int = 30
    diagnostic_mode: bool = False
    user_context_dir: Path | None = None
    knowledge_data_dir: Path = DEFAULT_KNOWLEDGE_DIR
    knowledge_index_path: Path = Path(DEFAULT_KNOWLEDGE_INDEX)
    knowledge_top_k: int = 3
    # 联网检索（博查 AI Search）。api_key 为空 = 关闭联网与认知提炼。
    bocha_api_key: str = ""
    bocha_base_url: str = DEFAULT_BOCHA_BASE_URL
    bocha_endpoint: str = DEFAULT_BOCHA_ENDPOINT
    bocha_count: int = DEFAULT_BOCHA_COUNT
    bocha_freshness: str = DEFAULT_BOCHA_FRESHNESS
    bocha_answer: bool = DEFAULT_BOCHA_ANSWER
    web_search_mode: str = DEFAULT_WEB_SEARCH_MODE
    # 联网检索的服务级开关（网页可切换）。False = 无条件不联网，
    # 交互关键路径上完全不触发外网调用。
    web_search_enabled: bool = False
    # 认知库（联网信息提炼出的行业通识认知块）
    insight_store_path: Path = Path(DEFAULT_INSIGHT_STORE)
    insight_max_per_turn: int = DEFAULT_INSIGHT_MAX_PER_TURN

    @classmethod
    def from_env(cls) -> ServerConfig:
        """Build configuration from environment variables.

        Raises:
            ValueError: If ``PNEUMA_LLM_API_KEY`` is not set.
        """
        llm_api_key = os.environ.get("PNEUMA_LLM_API_KEY", "")
        if not llm_api_key:
            raise ValueError("PNEUMA_LLM_API_KEY environment variable is not set")

        llm_base_url = os.environ.get("PNEUMA_LLM_BASE_URL")
        embedding_base_url = (
            os.environ.get("PNEUMA_EMBEDDING_BASE_URL") or llm_base_url
        )
        embedding_api_key = (
            os.environ.get("PNEUMA_EMBEDDING_API_KEY") or llm_api_key
        )

        user_context_dir = os.environ.get("PNEUMA_USER_CONTEXT_DIR")

        return cls(
            host=os.environ.get("PNEUMA_HOST", DEFAULT_HOST),
            port=int(os.environ.get("PNEUMA_PORT", DEFAULT_PORT)),
            log_level=os.environ.get("PNEUMA_LOG_LEVEL", "info").strip().lower() or "info",
            character_file=Path(
                os.environ.get("PNEUMA_CHARACTER_FILE", DEFAULT_CHARACTER_FILE)
            ),
            db_path=Path(os.environ.get("PNEUMA_DB_PATH", DEFAULT_DB_PATH)),
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model=os.environ.get("PNEUMA_LLM_MODEL", DEFAULT_LLM_MODEL),
            llm_timeout=float(os.environ.get("PNEUMA_LLM_TIMEOUT", "60")),
            llm_thinking_param=(
                os.environ.get("PNEUMA_LLM_THINKING_PARAM") or None
            ),
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            embedding_model=os.environ.get(
                "PNEUMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
            ),
            history_limit=int(os.environ.get("PNEUMA_HISTORY_LIMIT", "30")),
            diagnostic_mode=_env_bool("PNEUMA_DIAGNOSTIC", False),
            user_context_dir=Path(user_context_dir) if user_context_dir else None,
            knowledge_data_dir=Path(
                os.environ.get("PNEUMA_KNOWLEDGE_DIR", str(DEFAULT_KNOWLEDGE_DIR))
            ),
            knowledge_index_path=Path(
                os.environ.get(
                    "PNEUMA_KNOWLEDGE_INDEX", DEFAULT_KNOWLEDGE_INDEX
                )
            ),
            knowledge_top_k=int(os.environ.get("PNEUMA_KNOWLEDGE_TOP_K", "3")),
            bocha_api_key=os.environ.get("PNEUMA_BOCHA_API_KEY", "").strip(),
            bocha_base_url=os.environ.get(
                "PNEUMA_BOCHA_BASE_URL", DEFAULT_BOCHA_BASE_URL
            ),
            bocha_endpoint=os.environ.get(
                "PNEUMA_BOCHA_ENDPOINT", DEFAULT_BOCHA_ENDPOINT
            ),
            bocha_count=int(
                os.environ.get("PNEUMA_BOCHA_COUNT", str(DEFAULT_BOCHA_COUNT))
            ),
            bocha_freshness=os.environ.get(
                "PNEUMA_BOCHA_FRESHNESS", DEFAULT_BOCHA_FRESHNESS
            ),
            bocha_answer=_env_bool("PNEUMA_BOCHA_ANSWER", DEFAULT_BOCHA_ANSWER),
            web_search_mode=normalize_mode(
                os.environ.get("PNEUMA_WEB_SEARCH_MODE", DEFAULT_WEB_SEARCH_MODE)
            ),
            web_search_enabled=_env_bool("PNEUMA_WEB_SEARCH_ENABLED", False),
            insight_store_path=Path(
                os.environ.get("PNEUMA_INSIGHT_STORE", DEFAULT_INSIGHT_STORE)
            ),
            insight_max_per_turn=int(
                os.environ.get(
                    "PNEUMA_INSIGHT_MAX_PER_TURN", str(DEFAULT_INSIGHT_MAX_PER_TURN)
                )
            ),
        )
