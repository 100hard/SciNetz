"""Token generation utilities for shareable export links."""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

LOGGER = logging.getLogger(__name__)


class InvalidTokenError(RuntimeError):
    """Raised when a token cannot be verified."""


class ExpiredTokenError(RuntimeError):
    """Raised when a token has expired."""


@dataclass(frozen=True)
class IssuedShareToken:
    """Issued token details returned to the caller."""

    token: str
    metadata_id: str
    expires_at: Optional[datetime]


@dataclass(frozen=True)
class ShareTokenPayload:
    """Decoded token payload containing metadata lookup information."""

    metadata_id: str
    expires_at: Optional[datetime]


class ShareTokenManager:
    """Generate and validate share tokens using HMAC signatures."""

    def __init__(self, *, secret_key: str, clock: Optional[Callable[[], datetime]] = None) -> None:
        if not secret_key:
            msg = "secret_key must be provided for ShareTokenManager"
            raise ValueError(msg)
        self._secret_key = secret_key.encode("utf-8")
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def issue(self, *, metadata_id: str, ttl_hours: Optional[int]) -> IssuedShareToken:
        """Create a signed token for the provided metadata identifier.

        Args:
            metadata_id: Identifier used to look up persisted share metadata.
            ttl_hours: Token lifetime in hours. ``None`` disables expiry.

        Returns:
            IssuedShareToken: Signed token and expiry information.
        """

        if ttl_hours is not None and ttl_hours <= 0:
            msg = "ttl_hours must be positive when provided"
            raise ValueError(msg)
        now = self._ensure_timezone(self._clock())
        expires_at = None
        if ttl_hours is not None:
            expires_at = (now + timedelta(hours=ttl_hours)).replace(microsecond=0)
        payload_bytes = self._encode_payload(metadata_id, expires_at)
        signature_bytes = self._sign(payload_bytes)
        token = f"{self._b64encode(payload_bytes)}.{self._b64encode(signature_bytes)}"
        return IssuedShareToken(token=token, metadata_id=metadata_id, expires_at=expires_at)

    def decode(self, token: str, *, current_time: Optional[datetime] = None) -> ShareTokenPayload:
        """Decode and validate a share token.

        Args:
            token: Token value produced by :meth:`issue`.
            current_time: Optional override for validating expiry.

        Returns:
            ShareTokenPayload: Decoded metadata reference.

        Raises:
            InvalidTokenError: If the token cannot be decoded or the signature is invalid.
            ExpiredTokenError: If the token is valid but expired.
        """

        try:
            payload_part, signature_part = token.split(".", maxsplit=1)
        except ValueError as exc:  # pragma: no cover - defensive
            raise InvalidTokenError("Malformed share token") from exc
        payload_bytes = self._b64decode(payload_part)
        provided_signature = self._b64decode(signature_part)
        expected_signature = self._sign(payload_bytes)
        if not hmac.compare_digest(provided_signature, expected_signature):
            LOGGER.warning("Share token signature mismatch")
            raise InvalidTokenError("Invalid share token signature")
        metadata_id, expires_at = self._decode_payload(payload_bytes)
        now = self._ensure_timezone(current_time or self._clock())
        if expires_at is not None and now >= expires_at:
            raise ExpiredTokenError("Share token has expired")
        return ShareTokenPayload(metadata_id=metadata_id, expires_at=expires_at)

    @staticmethod
    def _ensure_timezone(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _encode_payload(metadata_id: str, expires_at: Optional[datetime]) -> bytes:
        if expires_at is None:
            expiry_str = "never"
        else:
            expiry_str = str(int(expires_at.timestamp()))
        payload = f"{metadata_id}|{expiry_str}"
        return payload.encode("utf-8")

    @staticmethod
    def _decode_payload(payload: bytes) -> tuple[str, Optional[datetime]]:
        try:
            decoded = payload.decode("utf-8")
            metadata_id, expiry_str = decoded.split("|", maxsplit=1)
            if expiry_str == "never":
                expiry = None
            else:
                expiry = datetime.fromtimestamp(int(expiry_str), tz=timezone.utc)
        except Exception as exc:  # pragma: no cover - defensive
            raise InvalidTokenError("Unable to decode share token payload") from exc
        return metadata_id, expiry

    def _sign(self, payload: bytes) -> bytes:
        return hmac.new(self._secret_key, payload, hashlib.sha256).digest()

    @staticmethod
    def _b64encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

    @staticmethod
    def _b64decode(value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        try:
            return base64.urlsafe_b64decode(value + padding)
        except Exception as exc:
            raise InvalidTokenError("Unable to decode share token component") from exc
