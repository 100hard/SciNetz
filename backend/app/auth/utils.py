"""Utilities for JWT handling, email dispatch, and Google token verification."""
from __future__ import annotations

import asyncio
import logging
import secrets
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Any, Dict, Optional, Sequence
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl

from aiosmtplib import send
from jose import JWTError, jwt

from backend.app.config import AuthJWTConfig, AuthSMTPConfig, AuthVerificationConfig

try:  # pragma: no cover - import validated via dependency management
    from google.oauth2 import id_token
    from google.auth.transport import requests as google_requests
except Exception as exc:  # pragma: no cover - surfaced during initialization
    GOOGLE_IMPORT_ERROR = exc
    id_token = None  # type: ignore[assignment]
    google_requests = None  # type: ignore[assignment]
else:  # pragma: no cover - exercised indirectly
    GOOGLE_IMPORT_ERROR = None

LOGGER = logging.getLogger(__name__)


class JWTManager:
    """Helper for encoding and decoding JSON Web Tokens."""

    def __init__(self, config: AuthJWTConfig) -> None:
        self._config = config

    def create_access_token(
        self,
        subject: str,
        additional_claims: Optional[Dict[str, Any]] = None,
        expires_delta: Optional[timedelta] = None,
        issued_at: Optional[datetime] = None,
    ) -> str:
        """Create a signed JWT access token."""

        now = issued_at or datetime.now(timezone.utc)
        ttl = expires_delta or timedelta(minutes=self._config.access_token_expires_minutes)
        expire_at = now + ttl
        payload: Dict[str, Any] = {
            "sub": subject,
            "iat": int(now.timestamp()),
            "exp": int(expire_at.timestamp()),
        }
        if additional_claims:
            payload.update(additional_claims)
        return jwt.encode(payload, self._config.secret_key, algorithm=self._config.algorithm)

    def decode(self, token: str) -> Dict[str, Any]:
        """Decode and validate a JWT."""

        return jwt.decode(token, self._config.secret_key, algorithms=[self._config.algorithm])


def generate_refresh_token(size: int = 48) -> str:
    """Generate a cryptographically secure refresh token."""

    return secrets.token_urlsafe(size)


class GoogleTokenVerificationError(RuntimeError):
    """Raised when Google credential validation fails."""


class GoogleTokenVerifier:
    """Verify Google ID tokens against the configured audiences."""

    def __init__(self, client_ids: Sequence[str]) -> None:
        if GOOGLE_IMPORT_ERROR is not None:
            msg = "google-auth dependency is unavailable"
            raise RuntimeError(msg) from GOOGLE_IMPORT_ERROR
        normalized = [client_id.strip() for client_id in client_ids if client_id and client_id.strip()]
        if not normalized:
            msg = "At least one Google client ID must be supplied"
            raise ValueError(msg)
        self._allowed_audiences = set(normalized)

    async def verify(self, credential: str) -> Dict[str, Any]:
        """Validate the provided Google credential and return its claims."""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._verify_sync, credential)

    def _verify_sync(self, credential: str) -> Dict[str, Any]:
        if not credential:
            raise GoogleTokenVerificationError("Missing Google credential")
        request = google_requests.Request()
        try:
            claims = id_token.verify_oauth2_token(credential, request, audience=None)
        except Exception as exc:  # pragma: no cover - upstream library handles edge cases
            raise GoogleTokenVerificationError("Failed to verify Google credential") from exc
        audiences: set[str] = set()
        audience_claim = claims.get("aud")
        if isinstance(audience_claim, str):
            if audience_claim:
                audiences.add(audience_claim)
        elif isinstance(audience_claim, (list, tuple, set)):
            audiences.update(
                str(value).strip()
                for value in audience_claim
                if isinstance(value, str) and value.strip()
            )
        candidate_matches = audiences & self._allowed_audiences
        if not candidate_matches:
            azp_claim = claims.get("azp")
            if isinstance(azp_claim, str) and azp_claim.strip() in self._allowed_audiences:
                return claims
            LOGGER.warning(
                "Rejected Google credential due to audience mismatch",
                extra={
                    "audiences": sorted(audiences),
                    "authorized_party": azp_claim if isinstance(azp_claim, str) else None,
                    "allowed_audiences": sorted(self._allowed_audiences),
                },
            )
            raise GoogleTokenVerificationError("Google credential has an unexpected audience")
        return claims


def build_verification_link(config: AuthVerificationConfig, token: str) -> str:
    """Construct the absolute verification link using the configured base URL."""

    parsed = urlparse(config.link_base_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update({"token": token})
    encoded_query = urlencode(query)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, encoded_query, parsed.fragment))


def build_verification_email(
    smtp_config: AuthSMTPConfig, recipient: str, verification_link: str, expires_at: datetime
) -> EmailMessage:
    """Render the verification email template."""

    message = EmailMessage()
    message["From"] = smtp_config.from_email
    message["To"] = recipient
    message["Subject"] = "Verify your SciNets account"
    message.set_content(
        (
            "Hello,\n\n"
            "Welcome to SciNets. Please confirm your email address by clicking the link below.\n"
            f"Verification link: {verification_link}\n"
            f"This link expires at {expires_at.isoformat()}.\n\n"
            "If you did not sign up, please ignore this email."
        )
    )
    return message


class EmailDispatcher:
    """Send transactional authentication emails."""

    def __init__(self, smtp_config: AuthSMTPConfig, verification_config: AuthVerificationConfig) -> None:
        self._smtp_config = smtp_config
        self._verification_config = verification_config

    async def send_verification_email(self, recipient: str, token: str, expires_at: datetime) -> None:
        """Deliver the verification email to the provided recipient."""

        verification_link = build_verification_link(self._verification_config, token)
        message = build_verification_email(self._smtp_config, recipient, verification_link, expires_at)
        try:
            await send(
                message,
                hostname=self._smtp_config.host,
                port=self._smtp_config.port,
                username=self._smtp_config.username or None,
                password=self._smtp_config.password or None,
                start_tls=self._smtp_config.use_tls,
            )
        except Exception:  # pragma: no cover - network failures are logged
            LOGGER.exception("Failed to send verification email")


__all__ = [
    "JWTManager",
    "EmailDispatcher",
    "build_verification_email",
    "build_verification_link",
    "generate_refresh_token",
    "GoogleTokenVerifier",
    "GoogleTokenVerificationError",
    "JWTError",
]
