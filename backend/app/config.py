"""Configuration loader for SciNets backend."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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


class ExtractionConfig(_FrozenModel):
    """Extraction parameters for Phase 3 pipeline."""

    max_triples_per_chunk_base: int = Field(..., ge=1)
    tokens_per_triple: int = Field(..., ge=1)
    chunk_size_tokens: int = Field(..., ge=1)
    chunk_overlap_tokens: int = Field(..., ge=0)


class CanonicalizationConfig(_FrozenModel):
    """Canonicalization thresholds and parameters."""

    base_threshold: float = Field(..., ge=0.0, le=1.0)
    polysemy_threshold: float = Field(..., ge=0.0, le=1.0)
    polysemy_section_diversity: int = Field(..., ge=1)


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


class ExportConfig(_FrozenModel):
    """Export settings for downloadable artifacts."""

    max_size_mb: int = Field(..., ge=1)
    warn_threshold_mb: int = Field(..., ge=1)
    snippet_truncate_length: int = Field(..., ge=1)


class AppConfig(_FrozenModel):
    """Top-level application configuration composed from config.yaml."""

    pipeline: PipelineConfig
    parsing: ParsingConfig
    extraction: ExtractionConfig
    canonicalization: CanonicalizationConfig
    co_mention: CoMentionConfig
    qa: QAConfig
    export: ExportConfig

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
