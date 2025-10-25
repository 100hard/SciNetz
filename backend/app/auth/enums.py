"""Shared authentication enums."""
from __future__ import annotations

from enum import Enum


class UserRole(str, Enum):
    """Enumerated application role (single tier)."""

    USER = "user"


__all__ = ["UserRole"]
