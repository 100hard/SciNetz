"""Repository handling persistence for authentication models."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Optional

from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth.enums import UserRole
from backend.app.auth.models import EmailVerification, RefreshToken, User


class AuthRepository:
    """Provide database access helpers for authentication workflows."""

    def __init__(self, session: AsyncSession, pwd_context: Optional[CryptContext] = None) -> None:
        self._session = session
        self._pwd_context = pwd_context or CryptContext(schemes=["bcrypt"], deprecated="auto")

    @property
    def session(self) -> AsyncSession:
        """Return the underlying SQLAlchemy session."""

        return self._session

    def hash_password(self, password: str) -> str:
        """Hash the provided password using bcrypt."""

        return self._pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: Optional[str]) -> bool:
        """Verify whether a plaintext password matches a stored hash."""

        if not hashed_password:
            return False
        return self._pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def hash_token(token: str) -> str:
        """Return a deterministic hash for sensitive token storage."""

        digest = hashlib.sha256()
        digest.update(token.encode("utf-8"))
        return digest.hexdigest()

    async def create_user(
        self,
        email: str,
        password: Optional[str] = None,
        *,
        role: UserRole = UserRole.USER,
        is_verified: bool = False,
    ) -> User:
        """Persist a new user with the provided credentials."""

        normalized_email = email.strip().lower()
        hashed = self.hash_password(password) if password is not None else None
        user = User(
            email=normalized_email,
            hashed_password=hashed,
            role=role,
            is_verified=is_verified,
        )
        self._session.add(user)
        await self._session.flush()
        return user

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Retrieve a user record by email address."""

        normalized_email = email.strip().lower()
        result = await self._session.execute(select(User).where(User.email == normalized_email))
        return result.scalar_one_or_none()

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Retrieve a user record by identifier."""

        result = await self._session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def create_email_verification(
        self, user: User, token: str, expires_at: datetime
    ) -> EmailVerification:
        """Store a verification token for a user."""

        verification = EmailVerification(
            user=user,
            token_hash=self.hash_token(token),
            expires_at=expires_at,
        )
        self._session.add(verification)
        await self._session.flush()
        return verification

    async def get_email_verification(self, token: str) -> Optional[EmailVerification]:
        """Return an email verification entry for the given token."""

        token_hash = self.hash_token(token)
        result = await self._session.execute(
            select(EmailVerification)
            .options(selectinload(EmailVerification.user))
            .where(EmailVerification.token_hash == token_hash)
        )
        return result.scalar_one_or_none()

    async def mark_email_verified(
        self, verification: EmailVerification, timestamp: Optional[datetime] = None
    ) -> None:
        """Mark the user and token as verified."""

        mark_time = timestamp or datetime.now(timezone.utc)
        verification.consumed_at = mark_time
        verification.user.is_verified = True
        verification.user.updated_at = mark_time
        await self._session.flush()

    async def create_refresh_token(
        self,
        user: User,
        token: str,
        expires_at: datetime,
        user_agent: Optional[str] = None,
    ) -> RefreshToken:
        """Persist a refresh token for a user session."""

        refresh_token = RefreshToken(
            user=user,
            token_hash=self.hash_token(token),
            user_agent=user_agent,
            expires_at=expires_at,
        )
        self._session.add(refresh_token)
        await self._session.flush()
        return refresh_token

    async def get_refresh_token(self, token: str) -> Optional[RefreshToken]:
        """Retrieve a refresh token entry using the raw token."""

        token_hash = self.hash_token(token)
        result = await self._session.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
        return result.scalar_one_or_none()

    async def revoke_refresh_token(
        self, refresh_token: RefreshToken, timestamp: Optional[datetime] = None
    ) -> None:
        """Soft delete a refresh token so it can no longer be used."""

        refresh_token.revoked_at = timestamp or datetime.now(timezone.utc)
        await self._session.flush()

    async def commit(self) -> None:
        """Commit the current transaction."""

        await self._session.commit()

    async def rollback(self) -> None:
        """Rollback the current transaction."""

        await self._session.rollback()


__all__ = ["AuthRepository"]
