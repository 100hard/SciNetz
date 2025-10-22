"""Initial migration creating authentication tables."""
from __future__ import annotations

from sqlalchemy.engine import Connection

from backend.app.auth.models import Base


def upgrade(connection: Connection) -> None:
    """Create authentication tables."""

    Base.metadata.create_all(connection)


def downgrade(connection: Connection) -> None:
    """Drop authentication tables."""

    Base.metadata.drop_all(connection)


__all__ = ["upgrade", "downgrade"]
