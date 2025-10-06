"""FastAPI application factory for SciNets backend."""
from __future__ import annotations

from fastapi import FastAPI

from backend.app.config import AppConfig, load_config


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Args:
        config: Optional pre-loaded configuration. If omitted, the default
            configuration defined in config.yaml is used.

    Returns:
        FastAPI: Configured FastAPI application.
    """
    resolved_config = config or load_config()
    app = FastAPI(title="SciNets API", version=resolved_config.pipeline.version)

    @app.get("/health", tags=["system"], summary="Service health probe")
    def health() -> dict[str, str]:
        """Return service health information.

        Returns:
            dict[str, str]: Minimal health payload containing pipeline version.
        """
        return {"status": "ok", "pipeline_version": resolved_config.pipeline.version}

    return app
