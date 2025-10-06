"""Immutable data contracts for SciNets backend."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class _FrozenBaseModel(BaseModel):
    """Base model enforcing immutability after creation."""

    model_config = ConfigDict(frozen=True)


class TextSpan(_FrozenBaseModel):
    """Character offsets referencing evidence snippets."""

    start: int = Field(..., ge=0, description="Inclusive start offset within content.")
    end: int = Field(..., gt=0, description="Exclusive end offset within content.")

    @field_validator("end")
    @classmethod
    def _ensure_end_after_start(cls, value: int, info: ValidationInfo) -> int:
        """Validate that the span end position is greater than the start.

        Args:
            value: The proposed end offset.
            info: Validation context containing other field values.

        Returns:
            int: The validated end offset.

        Raises:
            ValueError: If the end offset is not greater than the start offset.
        """
        start = info.data.get("start", 0)
        if value <= start:
            raise ValueError("text span end must be greater than start")
        return value


class Evidence(_FrozenBaseModel):
    """Evidence metadata linking graph elements back to source text."""

    element_id: str = Field(..., min_length=1)
    text_span: TextSpan
    doc_id: str = Field(..., min_length=1)
    full_sentence: Optional[str] = Field(None, description="Full sentence containing the evidence.")


class ParsedElement(_FrozenBaseModel):
    """Parsed chunk of paper content with section context."""

    doc_id: str = Field(..., min_length=1)
    element_id: str = Field(..., min_length=1)
    section: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    content_hash: str = Field(..., min_length=64, max_length=64)
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., gt=0)

    @field_validator("content_hash")
    @classmethod
    def _validate_content_hash(cls, value: str) -> str:
        """Validate that the hash is a lowercase SHA256 string.

        Args:
            value: The provided hash string.

        Returns:
            str: The validated hash string.

        Raises:
            ValueError: If the hash is not 64 lowercase hexadecimal characters.
        """
        if len(value) != 64:
            raise ValueError("content_hash must be 64 hex characters")
        if not all(c in "0123456789abcdef" for c in value):
            raise ValueError("content_hash must be hexadecimal")
        return value

    @field_validator("end_char")
    @classmethod
    def _ensure_end_after_start(cls, value: int, info: ValidationInfo) -> int:
        """Validate that end_char is strictly greater than start_char.

        Args:
            value: The proposed end character offset.
            info: Validation context containing other field values.

        Returns:
            int: The validated end character offset.

        Raises:
            ValueError: If the end offset is not greater than the start offset.
        """
        start = info.data.get("start_char", 0)
        if value <= start:
            raise ValueError("end_char must be greater than start_char")
        return value


class Triplet(_FrozenBaseModel):
    """Knowledge graph triplet extracted from a document chunk."""

    subject: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    object: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: Evidence
    pipeline_version: str = Field(..., min_length=1)


class Node(_FrozenBaseModel):
    """Canonical graph node with provenance statistics."""

    node_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    aliases: List[str] = Field(default_factory=list)
    section_distribution: Dict[str, int] = Field(default_factory=dict)
    times_seen: int = Field(0, ge=0)
    source_document_ids: List[str] = Field(default_factory=list)


class Edge(_FrozenBaseModel):
    """Directed relation between two canonical nodes."""

    src_id: str = Field(..., min_length=1)
    dst_id: str = Field(..., min_length=1)
    relation: str = Field(..., min_length=1)
    evidence: Evidence
    confidence: float = Field(..., ge=0.0, le=1.0)
    pipeline_version: str = Field(..., min_length=1)
    conflicting: bool = False
    created_at: datetime
    attributes: Dict[str, str] = Field(default_factory=dict)


class PaperMetadata(_FrozenBaseModel):
    """Minimal metadata extracted from a research paper."""

    doc_id: str = Field(..., min_length=1)
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = Field(None, ge=1900, le=2100)
    venue: Optional[str] = None
    doi: Optional[str] = None


__all__ = [
    "TextSpan",
    "Evidence",
    "ParsedElement",
    "Triplet",
    "Node",
    "Edge",
    "PaperMetadata",
]
