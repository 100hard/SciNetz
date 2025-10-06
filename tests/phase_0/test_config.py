from __future__ import annotations

import yaml

from backend.app.config import AppConfig, load_config


def test_config_loads_expected_structure(tmp_path) -> None:
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.pipeline.version == "1.0.0"
    assert config.parsing.section_fuzzy_threshold == 0.85
    assert "NeurIPS" in config.parsing.metadata_known_venues
    assert config.extraction.max_triples_per_chunk_base == 15
    assert config.canonicalization.base_threshold == 0.86
    assert config.canonicalization.polysemy_section_diversity == 3
    assert config.qa.entity_match_threshold == 0.83
    assert config.export.max_size_mb == 5


def test_config_strict_fields_match_yaml() -> None:
    config_path = AppConfig.default_path()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = load_config()
    assert raw["pipeline"]["version"] == config.pipeline.version
    assert raw["co_mention"]["confidence"] == config.co_mention.confidence
    assert raw["parsing"]["metadata_max_pages"] == config.parsing.metadata_max_pages
