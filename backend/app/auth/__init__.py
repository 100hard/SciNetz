"""Authentication package providing user management and JWT utilities."""

from backend.app.auth.service import AuthService
from backend.app.auth.utils import EmailDispatcher, JWTManager

__all__ = ["AuthService", "JWTManager", "EmailDispatcher"]
