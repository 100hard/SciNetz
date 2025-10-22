"""Tests covering authentication workflows."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from backend.app.auth.enums import UserRole
from backend.app.auth.models import AuthBase
from backend.app.auth.repository import AuthRepository
from backend.app.auth.schemas import LoginRequest, RegisterRequest
from backend.app.auth.service import AuthService, AuthServiceError
from backend.app.auth.utils import JWTError, JWTManager
from backend.app.config import AuthConfig, AuthJWTConfig, AuthSMTPConfig, AuthVerificationConfig


class StubEmailDispatcher:
    """Collect emails instead of sending them over SMTP."""

    def __init__(self) -> None:
        self.messages: List[Tuple[str, str, datetime]] = []

    async def send_verification_email(self, recipient: str, token: str, expires_at: datetime) -> None:
        self.messages.append((recipient, token, expires_at))


async def _setup_repository() -> Tuple[AuthRepository, AsyncEngine]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(AuthBase.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    session = session_factory()
    repo = AuthRepository(session)
    return repo, engine


def test_password_hashing_round_trip() -> None:
    async def _run() -> None:
        repo, engine = await _setup_repository()
        hashed = repo.hash_password("correct horse battery staple")
        assert hashed != "correct horse battery staple"
        assert repo.verify_password("correct horse battery staple", hashed)
        await repo.session.close()
        await engine.dispose()

    asyncio.run(_run())


def test_password_length_restriction() -> None:
    """Passwords longer than bcrypt's limit should be rejected."""

    with pytest.raises(ValidationError):
        RegisterRequest(
            email="toolong@example.com",
            password="x" * 73,
        )


def test_jwt_expiry_enforced() -> None:
    config = AuthJWTConfig(
        secret_key="super-secret-key-that-is-long-enough-123456",
        algorithm="HS256",
        access_token_expires_minutes=1,
        refresh_token_expires_minutes=60,
    )
    manager = JWTManager(config)
    token = manager.create_access_token(
        "user-id",
        expires_delta=timedelta(seconds=1),
        issued_at=datetime.now(timezone.utc) - timedelta(seconds=5),
    )
    with pytest.raises(JWTError):
        manager.decode(token)


def test_email_verification_flow() -> None:
    async def _run() -> None:
        repo, engine = await _setup_repository()
        dispatcher = StubEmailDispatcher()
        config = AuthConfig(
            database_url="sqlite+aiosqlite:///:memory:",
            jwt=AuthJWTConfig(
                secret_key="another-super-secret-key-that-is-long-enough-654321",
                algorithm="HS256",
                access_token_expires_minutes=5,
                refresh_token_expires_minutes=60,
            ),
            verification=AuthVerificationConfig(
                token_ttl_minutes=60,
                link_base_url="https://example.com/verify",
            ),
            smtp=AuthSMTPConfig(
                host="localhost",
                port=1025,
                username="",
                password="",
                use_tls=False,
                from_email="no-reply@example.com",
            ),
        )
        service = AuthService(config, repo, JWTManager(config.jwt), dispatcher)
        register_payload = RegisterRequest(email="user@example.com", password="password123")
        registration, token, expires_at = await service.register_user(register_payload)
        assert registration.user.email == "user@example.com"
        assert registration.user.role is UserRole.USER
        await service.send_verification_email(registration.user.email, token, expires_at)
        assert dispatcher.messages, "Verification email should be enqueued"
        verification_response = await service.verify_email(token)
        assert "Email verified" in verification_response.message
        login_payload = LoginRequest(email="user@example.com", password="password123")
        login_response = await service.login(login_payload)
        assert login_response.tokens.access_token
        assert login_response.user.role is UserRole.USER
        decoded = JWTManager(config.jwt).decode(login_response.tokens.access_token)
        assert decoded.get("role") == UserRole.USER.value
        await service.logout(login_response.tokens.refresh_token)
        with pytest.raises(AuthServiceError):
            await service.verify_email(token)
        await repo.session.close()
        await engine.dispose()

    asyncio.run(_run())
