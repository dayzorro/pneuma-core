"""Run the chat service: ``python -m pneuma_core.server``."""

from __future__ import annotations

import uvicorn

from pneuma_core.server.app import create_app
from pneuma_core.server.config import ServerConfig


def main() -> None:
    """Load configuration from the environment and start the HTTP server.

    The uvicorn ``Server`` instance is attached to ``app.state`` so the
    ``/api/admin/shutdown`` route can request a graceful shutdown over HTTP
    (see ``scripts/stop_server.sh``).
    """
    config = ServerConfig.from_env()
    app = create_app(config)
    server = uvicorn.Server(
        uvicorn.Config(app, host=config.host, port=config.port)
    )
    app.state.uvicorn_server = server
    server.run()


if __name__ == "__main__":
    main()
