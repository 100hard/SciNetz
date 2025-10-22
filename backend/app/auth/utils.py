"""Utilities for JWT handling and email dispatch."""
from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Any, Dict, Optional
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl

from aiosmtplib import send
from jose import JWTError, jwt

from backend.app.config import AuthJWTConfig, AuthSMTPConfig, AuthVerificationConfig

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
    "JWTError",
]
