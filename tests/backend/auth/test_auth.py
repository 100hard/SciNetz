"""Tests for authentication utilities and Google login flow."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Sequence, Tuple

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from backend.app.auth.enums import UserRole
from backend.app.auth.models import AuthBase, User
from backend.app.auth.repository import AuthRepository
from backend.app.auth.router import google_config
from backend.app.auth.schemas import GoogleLoginRequest, RefreshRequest, RegisterRequest
from backend.app.auth.service import AuthService, AuthServiceError
from backend.app.auth.utils import (
    GoogleTokenVerificationError,
    JWTError,
    JWTManager,
)
from backend.app.config import (
    AuthConfig,
    AuthGoogleConfig,
    AuthJWTConfig,
    AuthSMTPConfig,
    AuthVerificationConfig,
)


class StubEmailDispatcher:
    """Collect emails instead of sending them over SMTP."""

    def __init__(self) -> None:
        self.messages: List[Tuple[str, str, datetime]] = []

    async def send_verification_email(self, recipient: str, token: str, expires_at: datetime) -> None:
        self.messages.append((recipient, token, expires_at))


class StubGoogleVerifier:
    """Return precomputed claims for Google credentials."""

    def __init__(self, claims_map: Dict[str, Dict[str, Any]], allowed_audiences: Sequence[str]) -> None:
        self._claims_map = claims_map
        self._allowed_audiences = set(allowed_audiences)

    async def verify(self, credential: str) -> Dict[str, Any]:
        if credential not in self._claims_map:
            raise GoogleTokenVerificationError("invalid credential")
        claims = self._claims_map[credential]
        audience = claims.get("aud")
        if audience not in self._allowed_audiences:
            raise GoogleTokenVerificationError("unexpected audience")
        return claims


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


def test_google_config_endpoint_returns_client_ids() -> None:
    config = AuthConfig(
        database_url="sqlite+aiosqlite:///:memory:",
        jwt=AuthJWTConfig(
            secret_key="config-secret-key-that-is-long-enough-012345",
            algorithm="HS256",
            access_token_expires_minutes=5,
            refresh_token_expires_minutes=60,
        ),
        verification=AuthVerificationConfig(
            enabled=False,
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
        google=AuthGoogleConfig(client_ids=["client-one.apps.googleusercontent.com", "client-two.apps.googleusercontent.com"]),
    )

    response = asyncio.run(google_config(config=config))

    assert response.client_ids == [
        "client-one.apps.googleusercontent.com",
        "client-two.apps.googleusercontent.com",
    ]


def test_google_login_creates_verified_user() -> None:
    async def _run() -> None:
        repo, engine = await _setup_repository()
        dispatcher = StubEmailDispatcher()
        verifier = StubGoogleVerifier(
            {
                "valid-token": {
                    "sub": "abc-123",
                    "email": "user@example.com",
                    "email_verified": True,
                    "aud": "client-id",
                }
            },
            allowed_audiences=["client-id"],
        )
        config = AuthConfig(
            database_url="sqlite+aiosqlite:///:memory:",
            jwt=AuthJWTConfig(
                secret_key="another-super-secret-key-that-is-long-enough-654321",
                algorithm="HS256",
                access_token_expires_minutes=5,
                refresh_token_expires_minutes=60,
            ),
            verification=AuthVerificationConfig(
                enabled=True,
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
            google=AuthGoogleConfig(client_ids=["client-id"]),
        )
        service = AuthService(config, repo, JWTManager(config.jwt), dispatcher, verifier)
        payload = GoogleLoginRequest(credential="valid-token", user_agent="Firefox")
        response = await service.login_with_google(payload)
        assert response.user.email == "user@example.com"
        assert response.user.is_verified is True
        assert response.tokens.access_token
        stored = await repo.get_user_by_email("user@example.com")
        assert stored is not None
        assert stored.is_verified is True
        assert stored.hashed_password is None
        await repo.session.close()
        await engine.dispose()

    asyncio.run(_run())


def test_google_login_requires_verified_email() -> None:
    async def _run() -> None:
        repo, engine = await _setup_repository()
        dispatcher = StubEmailDispatcher()
        verifier = StubGoogleVerifier(
            {
                "unverified-token": {
                    "sub": "def-456",
                    "email": "user@example.com",
                    "email_verified": False,
                    "aud": "client-id",
                }
            },
            allowed_audiences=["client-id"],
        )
        config = AuthConfig(
            database_url="sqlite+aiosqlite:///:memory:",
            jwt=AuthJWTConfig(
                secret_key="no-verification-secret-key-that-is-long-enough-789012",
                algorithm="HS256",
                access_token_expires_minutes=5,
                refresh_token_expires_minutes=60,
            ),
            verification=AuthVerificationConfig(
                enabled=False,
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
            google=AuthGoogleConfig(client_ids=["client-id"]),
        )
        service = AuthService(config, repo, JWTManager(config.jwt), dispatcher, verifier)
        with pytest.raises(AuthServiceError) as excinfo:
            await service.login_with_google(GoogleLoginRequest(credential="unverified-token"))
        assert excinfo.value.reason == "forbidden"
        await repo.session.close()
        await engine.dispose()

    asyncio.run(_run())


def test_google_login_reuses_existing_user() -> None:
    async def _run() -> None:
        repo, engine = await _setup_repository()
        dispatcher = StubEmailDispatcher()
        claims = {
            "shared-token": {
                "sub": "ghi-789",
                "email": "existing@example.com",
                "email_verified": True,
                "aud": "client-id",
            }
        }
        verifier = StubGoogleVerifier(claims, allowed_audiences=["client-id"])
        config = AuthConfig(
            database_url="sqlite+aiosqlite:///:memory:",
            jwt=AuthJWTConfig(
                secret_key="reuse-secret-key-that-is-long-enough-012345",
                algorithm="HS256",
                access_token_expires_minutes=5,
                refresh_token_expires_minutes=60,
            ),
            verification=AuthVerificationConfig(
                enabled=False,
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
            google=AuthGoogleConfig(client_ids=["client-id"]),
        )
        service = AuthService(config, repo, JWTManager(config.jwt), dispatcher, verifier)
        payload = GoogleLoginRequest(credential="shared-token")
        first_response = await service.login_with_google(payload)
        stored_user = await repo.get_user_by_email("existing@example.com")
        assert stored_user is not None
        first_user_id = stored_user.id
        second_response = await service.login_with_google(payload)
        assert second_response.user.id == first_response.user.id
        refreshed_user = await repo.get_user_by_email("existing@example.com")
        assert refreshed_user is not None
        assert refreshed_user.id == first_user_id
        async with repo.session.begin():
            count = await repo.session.execute(select(User))
            users = count.scalars().all()
        assert len(users) == 1
        await repo.session.close()
        await engine.dispose()

    asyncio.run(_run())


def test_refresh_tokens_extends_existing_session() -> None:
    async def _run() -> None:
        repo, engine = await _setup_repository()
        dispatcher = StubEmailDispatcher()
        verifier = StubGoogleVerifier({}, allowed_audiences=["client-id"])
        config = AuthConfig(
            database_url="sqlite+aiosqlite:///:memory:",
            jwt=AuthJWTConfig(
                secret_key="refresh-secret-key-that-is-long-enough-123456",
                algorithm="HS256",
                access_token_expires_minutes=5,
                refresh_token_expires_minutes=60,
            ),
            verification=AuthVerificationConfig(
                enabled=False,
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
            google=AuthGoogleConfig(client_ids=["client-id"]),
        )
        service = AuthService(config, repo, JWTManager(config.jwt), dispatcher, verifier)

        user = await repo.create_user(
            "refresh@example.com",
            password="refreshpass",
            role=UserRole.USER,
            is_verified=True,
        )
        await repo.commit()

        initial_expires = service._now() + timedelta(minutes=5)
        refresh_token = "static-refresh-token"
        await repo.create_refresh_token(
            user,
            refresh_token,
            initial_expires,
            user_agent="initial-agent",
        )
        await repo.commit()

        response = await service.refresh_tokens(
            RefreshRequest(refresh_token=refresh_token, user_agent="updated-agent")
        )

        assert response.tokens.refresh_token == refresh_token
        assert response.tokens.access_token
        stored = await repo.get_refresh_token(refresh_token)
        assert stored is not None
        assert stored.user_agent == "updated-agent"
        stored_expires = stored.expires_at
        if stored_expires.tzinfo is None:
            stored_expires = stored_expires.replace(tzinfo=timezone.utc)
        assert stored_expires > initial_expires

        await repo.session.close()
        await engine.dispose()

    asyncio.run(_run())
