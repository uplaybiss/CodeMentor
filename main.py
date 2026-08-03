"""EduAgent — command-line entry point.

Run the API server with::

    python main.py            # reads host/port from settings/.env

The FastAPI app itself lives in :mod:`api.main` and can also be served with
``uvicorn api.main:app --reload``.
"""

from __future__ import annotations

import uvicorn

from config.settings import get_settings
from utils.logging import setup_logging


def main() -> None:
    """Start the EduAgent REST API on the configured host/port."""
    settings = get_settings()
    setup_logging(settings.logging)

    uvicorn.run(
        "api.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
        log_level="info",
    )


if __name__ == "__main__":
    main()
