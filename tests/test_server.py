"""Tests for the FastAPI chat service."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from pneuma_core.llm.adapter import LLMResponse
from pneuma_core.server.app import create_app
from pneuma_core.server.config import ServerConfig

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


class StubLLM:
    """Deterministic LLM stub for reply / emotion / session-end analysis."""

    def __init__(self) -> None:
        self.requests: list = []

    async def generate(self, request) -> LLMResponse:
        self.requests.append(request)
        prompt = request.system_prompt
        if "PAD" in prompt:
            return LLMResponse(
                content=json.dumps(
                    {
                        "pleasure": 0.5,
                        "arousal": 0.2,
                        "dominance": 0.1,
                        "emotion_label": "喜び",
                        "situation": "楽しい会話",
                    },
                    ensure_ascii=False,
                ),
                model="stub",
                usage={},
            )
        if "記憶管理アシスタント" in prompt:
            return LLMResponse(
                content=json.dumps(
                    {
                        "episodic_memories": [],
                        "semantic_updates": [],
                        "user_context_updates": [],
                        "relationship_changes": [],
                    },
                    ensure_ascii=False,
                ),
                model="stub",
                usage={},
            )
        return LLMResponse(
            content=json.dumps(
                {"speech": "やあ、よろしくね。", "thought": "初対面だ", "action": None},
                ensure_ascii=False,
            ),
            model="stub",
            usage={},
        )


class StubEmbedding:
    async def embed(self, text: str) -> list[float]:
        return [0.1] * 8

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts]


@pytest.fixture
def config(tmp_path: Path) -> ServerConfig:
    return ServerConfig(
        host="127.0.0.1",
        port=8001,
        character_file=EXAMPLES_DIR / "aine.character.yaml",
        db_path=tmp_path / "pneuma.db",
        llm_api_key="test-key",
        llm_model="test-model",
        llm_base_url="http://localhost/v1",
        embedding_api_key="test-key",
        embedding_model="test-embed",
        embedding_base_url="http://localhost/v1",
        diagnostic_mode=True,
    )


@pytest.fixture
async def client(config: ServerConfig):
    """An HTTP client backed by the app with stubbed LLM/embedding."""
    app = create_app(config)
    service = app.state.service
    service._llm = StubLLM()
    service._embedding = StubEmbedding()
    await service.setup()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestServerConfig:
    def test_from_env_requires_llm_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="PNEUMA_LLM_API_KEY"):
                ServerConfig.from_env()

    def test_embedding_falls_back_to_llm(self) -> None:
        env = {
            "PNEUMA_LLM_API_KEY": "k",
            "PNEUMA_LLM_BASE_URL": "https://example.com/v1",
        }
        with patch.dict("os.environ", env, clear=True):
            cfg = ServerConfig.from_env()
        assert cfg.embedding_api_key == "k"
        assert cfg.embedding_base_url == "https://example.com/v1"


class TestServiceEndpoints:
    async def test_healthz(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "character": "アイネ"}

    async def test_character(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/character")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "アイネ"
        assert set(body["personality"]) == {
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        }

    async def test_chat_without_session_returns_409(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.post("/api/chat", json={"message": "こんにちは"})
        assert resp.status_code == 409

    async def test_end_without_session_returns_409(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.post("/api/session/end")
        assert resp.status_code == 409

    async def test_full_flow(self, client: httpx.AsyncClient) -> None:
        # 1. Set who I am
        start = await client.post(
            "/api/session/start",
            json={"user_id": "u1", "user_name": "太郎"},
        )
        assert start.status_code == 200
        start_body = start.json()
        assert start_body["user"]["name"] == "太郎"
        assert start_body["character"]["name"] == "アイネ"
        assert start_body["session_id"]

        # 2. Chat normally
        chat = await client.post("/api/chat", json={"message": "こんにちは"})
        assert chat.status_code == 200
        chat_body = chat.json()
        assert chat_body["reply"] == "やあ、よろしくね。"
        assert chat_body["thought"] == "初対面だ"
        # Non-diagnostic lag: the reply carries the state at turn start
        assert chat_body["emotion"]["label"] == "中立"

        # 3. State reflects the freshly persisted emotion
        state = await client.get("/api/state")
        assert state.status_code == 200
        state_body = state.json()
        assert state_body["session_active"] is True
        assert state_body["emotion"]["label"] == "喜び"
        assert state_body["memory_counts"] == {"episodic": 0, "semantic": 0}

        # 4. End the session (memory consolidation)
        end = await client.post("/api/session/end")
        assert end.status_code == 200
        assert end.json()["success"] is True

        after = await client.get("/api/state")
        assert after.json()["session_active"] is False

    async def test_start_session_ends_previous(
        self, client: httpx.AsyncClient
    ) -> None:
        first = await client.post(
            "/api/session/start", json={"user_id": "u1", "user_name": "太郎"}
        )
        second = await client.post(
            "/api/session/start", json={"user_id": "u2", "user_name": "花子"}
        )
        assert first.json()["session_id"] != second.json()["session_id"]
        assert second.json()["user"]["name"] == "花子"

    async def test_empty_message_rejected(self, client: httpx.AsyncClient) -> None:
        await client.post("/api/session/start", json={"user_id": "u1", "user_name": "T"})
        resp = await client.post("/api/chat", json={"message": ""})
        assert resp.status_code == 422
