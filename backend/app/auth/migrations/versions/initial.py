"""Initial migration creating authentication tables."""
from __future__ import annotations

from sqlalchemy.engine import Connection

from backend.app.auth.models import AuthBase


def upgrade(connection: Connection) -> None:
    """Create authentication tables."""

    AuthBase.metadata.create_all(connection)


def downgrade(connection: Connection) -> None:
    """Drop authentication tables."""

    AuthBase.metadata.drop_all(connection)


__all__ = ["upgrade", "downgrade"]
