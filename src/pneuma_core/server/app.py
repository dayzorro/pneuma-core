"""FastAPI application exposing the chat service over HTTP."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pneuma_core.server.config import ServerConfig
from pneuma_core.server.service import ChatService, SessionNotStartedError


class StartSessionRequest(BaseModel):
    """Identify who the user is before starting to chat."""

    user_id: str = Field(..., min_length=1, description="ユーザーID / user id")
    user_name: str = Field(..., min_length=1, description="表示名 / display name")


class ChatRequest(BaseModel):
    """A single user message."""

    message: str = Field(..., min_length=1)


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Create the FastAPI app with a lazily-configured ChatService."""
    cfg = config or ServerConfig.from_env()
    service = ChatService(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await service.setup()
        yield

    app = FastAPI(
        title="Pneuma Core Chat Service",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.service = service

    @app.get("/healthz", summary="Health check")
    async def healthz() -> dict:
        return {"status": "ok", "character": service.character_info()["name"]}

    @app.get("/api/character", summary="Character definition")
    async def character() -> dict:
        return service.character_info()

    @app.post("/api/session/start", summary="Set the user identity and open a session")
    async def start_session(req: StartSessionRequest) -> dict:
        return await service.start_session(req.user_id, req.user_name)

    @app.post("/api/chat", summary="Send a message and get the reply")
    async def chat(req: ChatRequest) -> dict:
        try:
            return await service.chat(req.message)
        except SessionNotStartedError:
            raise HTTPException(
                status_code=409,
                detail="No active session. Call POST /api/session/start first.",
            )

    @app.post("/api/session/end", summary="End the session and consolidate memories")
    async def end_session() -> dict:
        try:
            return await service.end_session()
        except SessionNotStartedError:
            raise HTTPException(status_code=409, detail="No active session.")

    @app.get("/api/state", summary="Current emotion, memories and relations")
    async def state() -> dict:
        return await service.state()

    return app
