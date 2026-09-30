"""FastAPI application exposing the chat service over HTTP."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from pneuma_core.server.config import ServerConfig
from pneuma_core.server.service import ChatService, SessionNotStartedError
from pneuma_core.server.web import INDEX_HTML


class StartSessionRequest(BaseModel):
    """Identify who the user is before starting to chat."""

    user_id: str = Field(..., min_length=1, description="用户 ID / user id")
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

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return INDEX_HTML

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

    @app.post(
        "/api/chat/stream",
        summary="Send a message and stream the reply (Server-Sent Events)",
    )
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        """以 SSE 流式返回回复。

        事件为 ``data: {"type": ...}\\n\\n``，type 取值：
        ``delta``（speech 增量）、``done``（含 emotion 等的完整返回体）、
        ``error``。
        """

        async def event_source():
            try:
                async for event in service.chat_stream(req.message):
                    payload = json.dumps(event, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
            except SessionNotStartedError:
                payload = json.dumps(
                    {"type": "error", "message": "No active session"},
                    ensure_ascii=False,
                )
                yield f"data: {payload}\n\n"

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
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

    @app.post("/api/admin/shutdown", include_in_schema=False)
    async def shutdown(request: Request) -> dict:
        """Request a graceful shutdown.

        Only reachable from localhost: the server may be exposed to the LAN,
        and an open shutdown endpoint would let anyone stop it.
        """
        client_host = request.client.host if request.client else ""
        if client_host not in ("127.0.0.1", "::1"):
            raise HTTPException(
                status_code=403, detail="Shutdown is only allowed from localhost."
            )
        server = getattr(app.state, "uvicorn_server", None)
        if server is None:
            raise HTTPException(
                status_code=503,
                detail="Shutdown unavailable (server was not started via pneuma_core.server).",
            )
        server.should_exit = True
        return {"status": "shutting_down"}

    return app
