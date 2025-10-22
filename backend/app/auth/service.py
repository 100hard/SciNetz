"""Service layer orchestrating authentication workflows."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from backend.app.auth.enums import UserRole
from backend.app.auth.repository import AuthRepository
from backend.app.auth.schemas import (
    AuthUser,
    LoginResponse,
    LogoutResponse,
    RegistrationResponse,
    TokenPair,
    TokenRefreshResponse,
    VerificationResponse,
)
from backend.app.auth.utils import EmailDispatcher, JWTError, JWTManager, generate_refresh_token
from backend.app.config import AuthConfig
from backend.app.auth.schemas import LoginRequest, RefreshRequest, RegisterRequest
from backend.app.auth.models import User


class AuthServiceError(RuntimeError):
    """Raised when authentication operations fail."""

    def __init__(self, message: str, reason: str = "bad_request") -> None:
        super().__init__(message)
        self.reason = reason


class AuthService:
    """Coordinate repository operations and JWT generation."""

    def __init__(
        self,
        config: AuthConfig,
        repository: AuthRepository,
        jwt_manager: JWTManager,
        email_dispatcher: EmailDispatcher,
    ) -> None:
        self._config = config
        self._repository = repository
        self._jwt_manager = jwt_manager
        self._email_dispatcher = email_dispatcher

    @staticmethod
    def _now() -> datetime:
        """Return current UTC timestamp."""

        return datetime.now(timezone.utc)

    @staticmethod
    def _to_auth_user(user: User) -> AuthUser:
        """Convert ORM user model into API schema."""

        return AuthUser(
            id=user.id,
            email=user.email,
            is_verified=user.is_verified,
            role=user.role,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )

    async def register_user(
        self, payload: RegisterRequest
    ) -> Tuple[RegistrationResponse, Optional[str], Optional[datetime]]:
        """Register a new user and create a verification token."""

        existing = await self._repository.get_user_by_email(payload.email)
        if existing is not None:
            raise AuthServiceError("Email already registered", reason="conflict")

        requires_verification = self._config.verification.enabled
        token: Optional[str] = None
        expires_at: Optional[datetime] = None
        now = self._now()
        try:
            user = await self._repository.create_user(
                payload.email, payload.password, role=UserRole.USER
            )
            if requires_verification:
                token = generate_refresh_token(32)
                expires_at = now + self._config.verification.token_ttl
                await self._repository.create_email_verification(user, token, expires_at)
            else:
                user.is_verified = True
                user.updated_at = now
            await self._repository.commit()
        except Exception as exc:  # pragma: no cover - defensive rollback
            await self._repository.rollback()
            raise exc

        message = (
            "Registration successful. Please verify your email."
            if requires_verification
            else "Registration successful."
        )
        response = RegistrationResponse(
            message=message,
            user=self._to_auth_user(user),
            requires_verification=requires_verification,
        )
        return response, token, expires_at

    async def send_verification_email(
        self, email: str, token: str, expires_at: datetime
    ) -> None:
        """Send the verification email via dispatcher."""

        if not self._config.verification.enabled:
            return
        await self._email_dispatcher.send_verification_email(email, token, expires_at)

    async def verify_email(self, token: str) -> VerificationResponse:
        """Validate the verification token and mark the user as verified."""

        verification = await self._repository.get_email_verification(token)
        if verification is None:
            raise AuthServiceError("Invalid verification token", reason="not_found")

        now = self._now()
        if verification.consumed_at is not None:
            raise AuthServiceError("Verification token already used", reason="gone")
        if verification.expires_at < now:
            raise AuthServiceError("Verification token expired", reason="gone")

        try:
            await self._repository.mark_email_verified(verification, now)
            await self._repository.commit()
        except Exception as exc:  # pragma: no cover - defensive rollback
            await self._repository.rollback()
            raise exc

        return VerificationResponse(message="Email verified successfully")

    async def login(self, payload: LoginRequest) -> LoginResponse:
        """Authenticate credentials and return JWT tokens."""

        user = await self._repository.get_user_by_email(payload.email)
        if user is None:
            raise AuthServiceError("Invalid credentials", reason="unauthorized")
        if not user.is_verified:
            raise AuthServiceError("Email verification required", reason="forbidden")
        if not user.is_active:
            raise AuthServiceError("User is disabled", reason="forbidden")
        if not self._repository.verify_password(payload.password, user.hashed_password):
            raise AuthServiceError("Invalid credentials", reason="unauthorized")

        now = self._now()
        access_token = self._jwt_manager.create_access_token(
            subject=user.id,
            additional_claims={"email": user.email, "role": user.role.value},
        )
        refresh_token = generate_refresh_token()
        refresh_expires = now + timedelta(
            minutes=self._config.jwt.refresh_token_expires_minutes
        )

        try:
            await self._repository.create_refresh_token(
                user, refresh_token, refresh_expires, payload.user_agent
            )
            await self._repository.commit()
        except Exception as exc:  # pragma: no cover - defensive rollback
            await self._repository.rollback()
            raise exc

        token_pair = TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=self._config.jwt.access_token_expires_minutes * 60,
        )
        return LoginResponse(
            message="Login successful",
            user=self._to_auth_user(user),
            tokens=token_pair,
        )

    async def refresh_tokens(self, payload: RefreshRequest) -> TokenRefreshResponse:
        """Rotate refresh tokens and issue a new access token."""

        record = await self._repository.get_refresh_token(payload.refresh_token)
        if record is None:
            raise AuthServiceError("Unknown refresh token", reason="unauthorized")

        now = self._now()
        if record.revoked_at is not None or record.expires_at < now:
            raise AuthServiceError("Refresh token expired", reason="unauthorized")

        await self._repository.session.refresh(record, attribute_names=["user"])
        user = record.user
        if not user.is_verified:
            raise AuthServiceError("Email verification required", reason="forbidden")
        if not user.is_active:
            raise AuthServiceError("User is disabled", reason="forbidden")

        new_access_token = self._jwt_manager.create_access_token(
            subject=user.id,
            additional_claims={"email": user.email, "role": user.role.value},
        )
        new_refresh_token = generate_refresh_token()
        new_refresh_expires = now + timedelta(
            minutes=self._config.jwt.refresh_token_expires_minutes
        )

        try:
            await self._repository.revoke_refresh_token(record, now)
            await self._repository.create_refresh_token(
                user, new_refresh_token, new_refresh_expires, payload.user_agent
            )
            await self._repository.commit()
        except Exception as exc:  # pragma: no cover - defensive rollback
            await self._repository.rollback()
            raise exc

        pair = TokenPair(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=self._config.jwt.access_token_expires_minutes * 60,
        )
        return TokenRefreshResponse(message="Token refreshed", tokens=pair)

    async def logout(self, token: str, user_id: str | None = None) -> LogoutResponse:
        """Revoke the supplied refresh token."""

        record = await self._repository.get_refresh_token(token)
        if record is not None:
            if user_id is not None and record.user_id != user_id:
                raise AuthServiceError("Refresh token does not belong to the user", reason="forbidden")
            if record.revoked_at is None:
                try:
                    await self._repository.revoke_refresh_token(record)
                    await self._repository.commit()
                except Exception as exc:  # pragma: no cover - defensive rollback
                    await self._repository.rollback()
                    raise exc
        return LogoutResponse(message="Logged out")

    async def authenticate(self, token: str) -> AuthUser:
        """Validate a bearer token and load the associated user."""

        try:
            payload = self._jwt_manager.decode(token)
        except JWTError as exc:  # pragma: no cover - underlying library tested
            raise AuthServiceError("Invalid access token", reason="unauthorized") from exc

        subject = payload.get("sub")
        if not subject:
            raise AuthServiceError("Invalid access token", reason="unauthorized")

        user = await self._repository.get_user_by_id(subject)
        if user is None or not user.is_verified or not user.is_active:
            raise AuthServiceError("User not authorized", reason="unauthorized")
        return self._to_auth_user(user)
