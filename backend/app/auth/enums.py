"""Shared authentication enums."""
from __future__ import annotations

from enum import Enum


class UserRole(str, Enum):
    """Enumerated application roles."""

    ADMIN = "admin"
    USER = "user"


__all__ = ["UserRole"]
