"""Server configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pneuma_core.knowledge import DEFAULT_DATA_DIR as DEFAULT_KNOWLEDGE_DIR

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8001
DEFAULT_CHARACTER_FILE = "examples/xiaorun-frontdesk.character.yaml"
DEFAULT_DB_PATH = "vault/pneuma.db"
DEFAULT_KNOWLEDGE_INDEX = "vault/knowledge_index.json"
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
    character_file: Path = Path(DEFAULT_CHARACTER_FILE)
    db_path: Path = Path(DEFAULT_DB_PATH)
    llm_base_url: str | None = None
    llm_api_key: str = ""
    llm_model: str = DEFAULT_LLM_MODEL
    llm_timeout: float = 60.0
    embedding_base_url: str | None = None
    embedding_api_key: str = ""
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    history_limit: int = 30
    diagnostic_mode: bool = False
    user_context_dir: Path | None = None
    knowledge_data_dir: Path = DEFAULT_KNOWLEDGE_DIR
    knowledge_index_path: Path = Path(DEFAULT_KNOWLEDGE_INDEX)
    knowledge_top_k: int = 3

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
            character_file=Path(
                os.environ.get("PNEUMA_CHARACTER_FILE", DEFAULT_CHARACTER_FILE)
            ),
            db_path=Path(os.environ.get("PNEUMA_DB_PATH", DEFAULT_DB_PATH)),
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model=os.environ.get("PNEUMA_LLM_MODEL", DEFAULT_LLM_MODEL),
            llm_timeout=float(os.environ.get("PNEUMA_LLM_TIMEOUT", "60")),
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
        )
