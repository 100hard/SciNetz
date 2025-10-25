"""Pydantic schemas for authentication APIs."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from backend.app.auth.enums import UserRole


class _FrozenModel(BaseModel):
    """Base immutable schema."""

    model_config = ConfigDict(frozen=True)


class AuthUser(_FrozenModel):
    """User information exposed through the API."""

    id: str = Field(..., min_length=1)
    email: EmailStr
    is_verified: bool
    role: UserRole
    created_at: datetime
    updated_at: datetime


class SessionStatusResponse(_FrozenModel):
    """Response payload describing authentication state."""

    authenticated: bool
    user: Optional[AuthUser] = None


class TokenPair(_FrozenModel):
    """Access and refresh tokens returned after authentication."""

    access_token: str = Field(..., min_length=1)
    refresh_token: str = Field(..., min_length=1)
    token_type: str = Field("bearer", min_length=1)
    expires_in: int = Field(..., ge=1)


class RegistrationResponse(_FrozenModel):
    """Response returned after user registration."""

    message: str = Field(..., min_length=1)
    user: AuthUser
    requires_verification: bool = True


class LoginResponse(_FrozenModel):
    """Response payload returned after a successful login."""

    message: str = Field(..., min_length=1)
    user: AuthUser
    tokens: TokenPair


class GoogleConfigResponse(_FrozenModel):
    """Exposed Google authentication configuration."""

    client_ids: List[str]


class TokenRefreshResponse(_FrozenModel):
    """Response payload when refreshing authentication tokens."""

    message: str = Field(..., min_length=1)
    tokens: TokenPair


class LogoutResponse(_FrozenModel):
    """Response payload confirming logout."""

    message: str = Field(..., min_length=1)


class VerificationResponse(_FrozenModel):
    """Response payload confirming email verification."""

    message: str = Field(..., min_length=1)


class RegisterRequest(BaseModel):
    """Registration input payload."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=72)


class LoginRequest(BaseModel):
    """Login request payload."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=72)
    user_agent: Optional[str] = Field(default=None, max_length=255)


class GoogleLoginRequest(BaseModel):
    """Google identity login payload."""

    credential: str = Field(..., min_length=1)
    user_agent: Optional[str] = Field(default=None, max_length=255)


class RefreshRequest(BaseModel):
    """Refresh token request payload."""

    refresh_token: str = Field(..., min_length=1)
    user_agent: Optional[str] = Field(default=None, max_length=255)


class LogoutRequest(BaseModel):
    """Logout request payload."""

    refresh_token: str = Field(..., min_length=1)


__all__ = [
    "AuthUser",
    "TokenPair",
    "RegistrationResponse",
    "LoginResponse",
    "SessionStatusResponse",
    "GoogleConfigResponse",
    "TokenRefreshResponse",
    "LogoutResponse",
    "VerificationResponse",
    "RegisterRequest",
    "LoginRequest",
    "GoogleLoginRequest",
    "RefreshRequest",
    "LogoutRequest",
]
