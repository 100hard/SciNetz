from __future__ import annotations

import yaml

from backend.app.config import AppConfig, load_config


def test_config_loads_expected_structure(tmp_path) -> None:
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.pipeline.version == "1.0.0"
    assert config.parsing.section_fuzzy_threshold == 0.85
    assert "Nature" in config.parsing.metadata_known_venues
    assert config.extraction.max_triples_per_chunk_base == 15
    assert config.extraction.max_prompt_entities == 12
    assert config.extraction.llm_provider == "openai"
    assert config.extraction.fuzzy_match_threshold == 0.9
    assert config.extraction.openai_model == "gpt-4o-mini"
    assert config.extraction.openai_base_url == "https://api.openai.com/v1"
    assert config.extraction.openai_timeout_seconds == 60
    assert config.extraction.openai_prompt_version == "phase3-v3"
    settings = config.extraction.openai
    assert config.extraction.cache_dir == "data/cache"
    assert config.extraction.response_cache_filename == "openai_responses.json"
    assert config.extraction.token_cache_filename == "openai_token_budget.json"
    assert settings.model == "gpt-4o-mini"
    assert settings.prompt_version == "phase3-v3"
    assert settings.max_retries == 2
    assert settings.temperature == 0.0
    assert settings.max_output_tokens == 3200
    assert settings.initial_output_multiplier == 2
    assert 429 in settings.retry_statuses
    assert config.canonicalization.base_threshold == 0.88
    assert config.canonicalization.polysemy_section_diversity == 3
    assert config.canonicalization.lexical_similarity_floor == 0.55
    assert config.canonicalization.min_shared_token_count == 1
    assert config.canonicalization.alias_token_limit == 5
    assert config.qa.entity_match_threshold == 0.83
    assert config.export.max_size_mb == 5
    assert config.extraction.use_entity_inventory is False
    assert "model" in config.canonicalization.polysemy_blocklist
    assert "uses" in config.relations.canonical_relation_names()
    graph_defaults = config.ui.graph_defaults
    assert graph_defaults.min_confidence == 0.5
    assert graph_defaults.relations[:2] == ["defined-as", "uses"]
    assert graph_defaults.sections == ["Results", "Methods"]
    assert graph_defaults.show_co_mentions is False
    assert config.graph.entity_batch_size == 200
    assert config.graph.edge_batch_size == 500


def test_config_strict_fields_match_yaml() -> None:
    config_path = AppConfig.default_path()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = load_config()
    assert raw["pipeline"]["version"] == config.pipeline.version
    assert raw["co_mention"]["confidence"] == config.co_mention.confidence
    assert raw["parsing"]["metadata_max_pages"] == config.parsing.metadata_max_pages
    assert (
        raw["ui"]["graph_defaults"]["min_confidence"]
        == config.ui.graph_defaults.min_confidence
    )
