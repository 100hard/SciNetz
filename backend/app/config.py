"""Configuration loader for SciNets backend."""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from typing_extensions import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

LOGGER = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded."""


class _FrozenModel(BaseModel):
    """Base model enforcing immutability for config sections."""

    model_config = ConfigDict(frozen=True)


class PipelineConfig(_FrozenModel):
    """Pipeline-level configuration."""

    version: str = Field(..., min_length=1)


class ParsingConfig(_FrozenModel):
    """Parsing and metadata extraction configuration."""

    section_fuzzy_threshold: float = Field(..., ge=0.0, le=1.0)
    metadata_max_pages: int = Field(..., ge=1)
    metadata_known_venues: List[str] = Field(default_factory=list)
    section_aliases: Dict[str, List[str]] = Field(default_factory=dict)



class OpenAIConfig(_FrozenModel):
    """Settings required for the OpenAI LLM adapter."""

    model: str = Field(..., min_length=1)
    api_base: str = Field(..., min_length=1)
    timeout_seconds: float = Field(..., gt=0)
    max_retries: int = Field(..., ge=0)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(..., ge=1)
    prompt_version: str = Field(..., min_length=1)
    initial_output_multiplier: float = Field(..., gt=0)
    backoff_initial_seconds: float = Field(..., gt=0)
    backoff_max_seconds: float = Field(..., gt=0)
    retry_statuses: List[int] = Field(default_factory=list)



class ExtractionConfig(_FrozenModel):
    """Extraction parameters for Phase 3 pipeline."""

    max_triples_per_chunk_base: int = Field(..., ge=1)
    tokens_per_triple: int = Field(..., ge=1)
    max_prompt_entities: int = Field(..., ge=1)
    chunk_size_tokens: int = Field(..., ge=1)
    chunk_overlap_tokens: int = Field(..., ge=0)
    use_entity_inventory: bool = False
    llm_provider: str = Field(..., min_length=1)
    fuzzy_match_threshold: float = Field(..., ge=0.0, le=1.0)
    openai_model: str = Field(..., min_length=1)
    openai_base_url: str = Field(..., min_length=1)
    openai_timeout_seconds: float = Field(..., gt=0)
    openai_prompt_version: str = Field(..., min_length=1)
    openai_max_retries: int = Field(..., ge=0)
    openai_temperature: float = Field(..., ge=0.0, le=2.0)
    openai_max_output_tokens: int = Field(..., ge=1)
    openai_initial_output_multiplier: float = Field(..., gt=0)
    openai_backoff_initial_seconds: float = Field(..., gt=0)
    openai_backoff_max_seconds: float = Field(..., gt=0)
    openai_retry_statuses: List[int] = Field(default_factory=list)
    cache_dir: str = Field(..., min_length=1)
    response_cache_filename: str = Field(..., min_length=1)
    token_cache_filename: str = Field(..., min_length=1)
    entity_types: List[str] = Field(..., min_length=1)

    @property
    def openai(self) -> OpenAIConfig:
        """Return the OpenAI adapter configuration.

        Returns:
            OpenAIConfig: Immutable settings for the OpenAI adapter.
        """

        return OpenAIConfig(
            model=self.openai_model,
            api_base=self.openai_base_url,
            timeout_seconds=self.openai_timeout_seconds,
            max_retries=self.openai_max_retries,
            temperature=self.openai_temperature,
            max_output_tokens=self.openai_max_output_tokens,
            prompt_version=self.openai_prompt_version,
            initial_output_multiplier=self.openai_initial_output_multiplier,
            backoff_initial_seconds=self.openai_backoff_initial_seconds,
            backoff_max_seconds=self.openai_backoff_max_seconds,
            retry_statuses=list(self.openai_retry_statuses),
        )


class QALLMConfig(_FrozenModel):
    """LLM configuration for QA answer synthesis."""

    enabled: bool
    provider: Literal["openai"]
    openai_model: str = Field(..., min_length=1)
    openai_base_url: str = Field(..., min_length=1)
    openai_timeout_seconds: float = Field(..., gt=0)
    openai_prompt_version: str = Field(..., min_length=1)
    openai_max_retries: int = Field(..., ge=0)
    openai_temperature: float = Field(..., ge=0.0, le=2.0)
    openai_max_output_tokens: int = Field(..., ge=1)
    openai_initial_output_multiplier: float = Field(..., gt=0)
    openai_backoff_initial_seconds: float = Field(..., gt=0)
    openai_backoff_max_seconds: float = Field(..., gt=0)
    openai_retry_statuses: List[int] = Field(default_factory=list)

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        normalized = value.lower()
        if normalized != "openai":
            msg = f"Unsupported QA LLM provider: {value}"
            raise ValueError(msg)
        return "openai"

    @property
    def openai(self) -> OpenAIConfig:
        """Return the OpenAI configuration block for QA synthesis.

        Returns:
            OpenAIConfig: Immutable OpenAI adapter configuration.
        """

        return OpenAIConfig(
            model=self.openai_model,
            api_base=self.openai_base_url,
            timeout_seconds=self.openai_timeout_seconds,
            max_retries=self.openai_max_retries,
            temperature=self.openai_temperature,
            max_output_tokens=self.openai_max_output_tokens,
            prompt_version=self.openai_prompt_version,
            initial_output_multiplier=self.openai_initial_output_multiplier,
            backoff_initial_seconds=self.openai_backoff_initial_seconds,
            backoff_max_seconds=self.openai_backoff_max_seconds,
            retry_statuses=list(self.openai_retry_statuses),
        )


class CanonicalizationConfig(_FrozenModel):
    """Canonicalization thresholds and parameters."""

    base_threshold: float = Field(..., ge=0.0, le=1.0)
    polysemy_threshold: float = Field(..., ge=0.0, le=1.0)
    polysemy_section_diversity: int = Field(..., ge=1)
    polysemy_blocklist: List[str] = Field(default_factory=list)
    lexical_similarity_threshold: float = Field(..., ge=0.0, le=1.0)
    lexical_similarity_bonus: float = Field(..., ge=0.0, le=1.0)
    type_match_bonus: float = Field(..., ge=0.0, le=1.0)
    lexical_similarity_floor: float = Field(..., ge=0.0, le=1.0)
    min_shared_token_count: int = Field(..., ge=0)
    alias_token_limit: int = Field(..., ge=1)
    alias_char_limit: int = Field(..., ge=1)
    long_alias_penalty: float = Field(..., ge=0.0, le=1.0)
    sentence_alias_penalty: float = Field(..., ge=0.0, le=1.0)
    cross_type_penalty: float = Field(..., ge=0.0, le=1.0)


class RelationPatternConfig(_FrozenModel):
    """Configuration describing how to normalize relation phrases."""

    canonical: str = Field(..., min_length=1)
    phrases: List[str] = Field(default_factory=list)
    swap: bool = False


class RelationsConfig(_FrozenModel):
    """Relation normalization and metadata configuration."""

    normalization_patterns: List[RelationPatternConfig] = Field(default_factory=list)

    def canonical_relation_names(self) -> List[str]:
        """Return the unique canonical relation identifiers."""

        return sorted({pattern.canonical for pattern in self.normalization_patterns})

    def normalized_patterns(self) -> List[Tuple[str, str, bool]]:
        """Return normalized phrase patterns for relation mapping."""

        patterns: List[Tuple[str, str, bool]] = []
        for entry in self.normalization_patterns:
            for phrase in entry.phrases:
                cleaned = re.sub(r"[^a-z]+", " ", phrase.lower()).strip()
                cleaned = re.sub(r"\s+", " ", cleaned)
                if not cleaned:
                    continue
                patterns.append((cleaned, entry.canonical, entry.swap))
        patterns.sort(key=lambda item: len(item[0]), reverse=True)
        return patterns


class GraphConfig(_FrozenModel):
    """Graph persistence configuration values."""

    relation_semantics: Dict[str, str] = Field(default_factory=dict)
    uri: Optional[str] = Field(default=None)
    username: Optional[str] = Field(default=None)
    password: Optional[str] = Field(default=None)
    entity_batch_size: int = Field(200, ge=1)
    edge_batch_size: int = Field(500, ge=1)

    @field_validator("relation_semantics")
    @classmethod
    def _normalize_semantics(cls, values: Dict[str, str]) -> Dict[str, str]:
        """Normalize and validate relation semantics values."""

        allowed = {"directional", "bidirectional"}
        normalized: Dict[str, str] = {}
        for relation, semantics in values.items():
            normalized_value = semantics.lower()
            if normalized_value not in allowed:
                msg = (
                    "relation semantics for '%s' must be either 'directional' or"
                    " 'bidirectional'"
                ) % relation
                raise ValueError(msg)
            normalized[relation] = normalized_value
        return normalized

    def is_directional(self, relation: str) -> bool:
        """Return whether the given relation should be treated as directional."""

        semantics = self.relation_semantics.get(relation)
        if semantics is None:
            return True
        return semantics == "directional"

    def directional_relations(self) -> List[str]:
        """Return the list of directional relation labels."""

        return [
            relation
            for relation, semantics in self.relation_semantics.items()
            if semantics == "directional"
        ]


class CoMentionConfig(_FrozenModel):
    """Co-mention edge configuration."""

    enabled: bool
    min_occurrences: int = Field(..., ge=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    max_distance_chars: int = Field(..., ge=1)


class QAConfig(_FrozenModel):
    """Question answering configuration."""

    entity_match_threshold: float = Field(..., ge=0.0, le=1.0)
    expand_neighbors: bool
    neighbor_confidence_threshold: float = Field(..., ge=0.0, le=1.0)
    max_hops: int = Field(..., ge=0)
    max_results: int = Field(..., ge=1)
    llm: QALLMConfig


class ExportStorageConfig(_FrozenModel):
    """Object storage configuration for export bundles."""

    bucket: str = Field(..., min_length=1)
    region: str = Field(..., min_length=1)
    prefix: str = Field(..., min_length=1)
    public_endpoint: str | None = Field(default=None, min_length=1)


class ExportConfig(_FrozenModel):
    """Export settings for downloadable artifacts."""

    max_bundle_mb: int = Field(..., ge=1)
    warn_bundle_mb: int = Field(..., ge=1)
    snippet_truncate_length: int = Field(..., ge=1)
    link_ttl_hours: Optional[int] = Field(default=None, ge=1)
    signed_url_ttl_minutes: int = Field(..., ge=1)
    storage: ExportStorageConfig

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "ExportConfig":
        if self.warn_bundle_mb > self.max_bundle_mb:
            msg = "export.warn_bundle_mb cannot exceed export.max_bundle_mb"
            raise ValueError(msg)
        return self


class UIGraphDefaultsConfig(_FrozenModel):
    """Default visualization controls for the UI graph view."""

    relations: List[str] = Field(default_factory=list)
    min_confidence: float = Field(..., ge=0.0, le=1.0)
    sections: List[str] = Field(default_factory=list)
    show_co_mentions: bool = False
    layout: Literal["fcose", "cose-bilkent"] = Field("fcose")


class UIConfig(_FrozenModel):
    """UI-specific configuration values."""

    upload_dir: str = Field(..., min_length=1)
    paper_registry_path: str = Field(..., min_length=1)
    graph_defaults: UIGraphDefaultsConfig
    allowed_origins: List[str] = Field(default_factory=list)


class AuthJWTConfig(_FrozenModel):
    """JWT signing settings."""

    secret_key: str = Field(..., min_length=32)
    algorithm: str = Field("HS256", min_length=1)
    access_token_expires_minutes: int = Field(..., ge=1)
    refresh_token_expires_minutes: int = Field(..., ge=1)

    @property
    def access_token_ttl(self) -> timedelta:
        """Return the configured access token lifetime."""

        return timedelta(minutes=self.access_token_expires_minutes)

    @property
    def refresh_token_ttl(self) -> timedelta:
        """Return the configured refresh token lifetime."""

        return timedelta(minutes=self.refresh_token_expires_minutes)


class AuthVerificationConfig(_FrozenModel):
    """Email verification token settings."""

    enabled: bool = True
    token_ttl_minutes: int = Field(..., ge=1)
    link_base_url: str = Field(..., min_length=1)

    @property
    def token_ttl(self) -> timedelta:
        """Return the verification token lifetime."""

        return timedelta(minutes=self.token_ttl_minutes)


class AuthSMTPConfig(_FrozenModel):
    """SMTP credentials for transactional email delivery."""

    host: str = Field(..., min_length=1)
    port: int = Field(..., ge=1, le=65535)
    username: str = Field(default="")
    password: str = Field(default="")
    use_tls: bool = False
    from_email: str = Field(..., min_length=3)


class AuthGoogleConfig(_FrozenModel):
    """Configuration for Google identity integration."""

    client_ids: List[str] = Field(default_factory=list)

    @field_validator("client_ids")
    @classmethod
    def _validate_client_ids(cls, value: List[str]) -> List[str]:
        normalized = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if not normalized:
            msg = "At least one Google client ID must be configured"
            raise ValueError(msg)
        return normalized


class AuthConfig(_FrozenModel):
    """Top-level authentication configuration."""

    database_url: str = Field(..., min_length=1)
    jwt: AuthJWTConfig
    verification: AuthVerificationConfig
    smtp: AuthSMTPConfig
    google: AuthGoogleConfig


class AppConfig(_FrozenModel):
    """Top-level application configuration composed from config.yaml."""

    pipeline: PipelineConfig
    parsing: ParsingConfig
    extraction: ExtractionConfig
    canonicalization: CanonicalizationConfig
    relations: RelationsConfig
    graph: GraphConfig
    co_mention: CoMentionConfig
    qa: QAConfig
    export: ExportConfig
    ui: UIConfig
    auth: AuthConfig

    @staticmethod
    def default_path() -> Path:
        """Return the default location of the configuration file.

        Returns:
            Path: Absolute path to config.yaml at the repository root.
        """
        return Path(__file__).resolve().parents[2] / "config.yaml"


def _read_yaml(path: Path) -> Dict[str, Any]:
    """Read YAML content from disk.

    Args:
        path: Location of the YAML file.

    Returns:
        Dict[str, Any]: Parsed YAML content.

    Raises:
        ConfigError: If the file cannot be read or parsed.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        LOGGER.error("Configuration file missing at %s", path)
        raise ConfigError("Configuration file not found") from exc
    except yaml.YAMLError as exc:
        LOGGER.error("Invalid YAML syntax in %s", path)
        raise ConfigError("Invalid YAML syntax") from exc
    if not isinstance(data, dict):
        LOGGER.error("Configuration root must be a mapping: %s", path)
        raise ConfigError("Configuration root must be a mapping")
    return data


@lru_cache(maxsize=1)
def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load application configuration from YAML.

    Args:
        path: Optional override path to the YAML file.

    Returns:
        AppConfig: Parsed configuration object.

    Raises:
        ConfigError: If the configuration cannot be loaded or validated.
    """
    config_path = path or AppConfig.default_path()
    raw_content = _read_yaml(config_path)
    try:
        return AppConfig(**raw_content)
    except ValidationError as exc:
        LOGGER.error("Invalid configuration values: %s", exc)
        raise ConfigError("Configuration validation failed") from exc
