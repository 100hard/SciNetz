"""Add role column to users table."""
from __future__ import annotations

from sqlalchemy.engine import Connection


def upgrade(connection: Connection) -> None:
    """Add the role column to the users table with a default value."""

    connection.exec_driver_sql(
        "ALTER TABLE users ADD COLUMN role VARCHAR(16) NOT NULL DEFAULT 'user'"
    )
    connection.exec_driver_sql("UPDATE users SET role = 'user' WHERE role IS NULL")


def downgrade(connection: Connection) -> None:
    """Remove the role column from the users table."""

    connection.exec_driver_sql("ALTER TABLE users DROP COLUMN role")


__all__ = ["upgrade", "downgrade"]
