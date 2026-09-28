"""Run the chat service: ``python -m pneuma_core.server``."""

from __future__ import annotations

import uvicorn

from pneuma_core.server.app import create_app
from pneuma_core.server.config import ServerConfig


def main() -> None:
    """Load configuration from the environment and start the HTTP server."""
    config = ServerConfig.from_env()
    uvicorn.run(create_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
