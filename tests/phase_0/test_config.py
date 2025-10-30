from __future__ import annotations

import yaml

from backend.app.config import AppConfig, load_config


def test_config_loads_expected_structure(tmp_path) -> None:
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.pipeline.version == "1.0.0"
    assert config.parsing.section_fuzzy_threshold == 0.85
    assert "Nature" in config.parsing.metadata_known_venues
    assert config.parsing.docling_artifacts_path == "data/cache/docling"
    assert config.parsing.accelerator.device == "auto"
    assert config.parsing.accelerator.num_threads == 8
    assert config.parsing.rapidocr.backend == "torch"
    assert config.parsing.rapidocr.text_score == 0.5
    assert config.parsing.rapidocr.use_det is True
    assert config.parsing.rapidocr.use_cls is True
    assert config.parsing.rapidocr.use_rec is True
    assert config.parsing.rapidocr.model_cache_dir == "data/cache/rapidocr"
    assert config.parsing.rapidocr.warmup_on_startup is True
    assert config.parsing.threaded_pdf.enabled is True
    assert config.parsing.threaded_pdf.ocr_batch_size == 4
    assert config.parsing.threaded_pdf.layout_batch_size == 4
    assert config.parsing.threaded_pdf.table_batch_size == 2
    assert config.parsing.threaded_pdf.batch_timeout_seconds == 2.0
    assert config.parsing.threaded_pdf.queue_max_size == 32
    assert config.extraction.max_triples_per_chunk_base == 15
    assert config.extraction.max_prompt_entities == 12
    assert config.extraction.concurrent_workers == 3
    assert config.extraction.llm_provider == "openai"
    assert config.extraction.fuzzy_match_threshold == 0.9
    assert config.extraction.openai_model == "gpt-4o-mini"
    assert config.extraction.openai_base_url == "https://api.openai.com/v1"
    assert config.extraction.openai_timeout_seconds == 60
    assert config.extraction.openai_prompt_version == "phase3-v4"
    settings = config.extraction.openai
    assert config.extraction.cache_dir == "data/cache"
    assert config.extraction.response_cache_filename == "openai_responses.json"
    assert config.extraction.token_cache_filename == "openai_token_budget.json"
    assert settings.model == "gpt-4o-mini"
    assert settings.prompt_version == "phase3-v4"
    assert settings.max_retries == 2
    assert settings.temperature == 0.0
    assert settings.max_output_tokens == 3200
    assert settings.initial_output_multiplier == 1.5
    assert 429 in settings.retry_statuses
    assert config.extraction.default_domain == "ml"
    domains = {domain.name: domain for domain in config.extraction.domains}
    assert {"ml", "biology", "physics"} <= set(domains)
    biology = domains["biology"]
    assert "Protein" in biology.entity_types
    assert biology.inventory_model == "en_core_sci_md"
    assert biology.fuzzy_match_threshold < config.extraction.fuzzy_match_threshold
    assert any("Cas9" in term for term in biology.vocabulary)
    physics = domains["physics"]
    assert "Phenomenon" in physics.entity_types
    assert physics.prompt_version.startswith("phase3-")
    assert config.canonicalization.base_threshold == 0.88
    assert config.canonicalization.polysemy_section_diversity == 3
    assert config.canonicalization.lexical_similarity_floor == 0.55
    assert config.canonicalization.min_shared_token_count == 1
    assert config.canonicalization.alias_token_limit == 5
    assert config.qa.entity_match_threshold == 0.83
    assert config.canonicalization.embedding_model == "intfloat/e5-base"
    assert config.canonicalization.embedding_device == "cpu"
    assert config.canonicalization.embedding_batch_size == 16
    assert config.canonicalization.preload_embeddings is True
    assert config.export.max_bundle_mb == 5
    assert config.export.warn_bundle_mb == 3
    assert config.export.link_ttl_hours is None
    assert config.export.signed_url_ttl_minutes == 10
    assert config.export.storage.bucket == "scinets-test-exports"
    assert config.export.storage.region == "us-east-1"
    assert config.export.storage.prefix == "exports"
    assert config.export.storage.public_endpoint == "http://localhost:9000"
    assert config.extraction.use_entity_inventory is False
    assert "model" in config.canonicalization.polysemy_blocklist
    relation_names = config.relations.canonical_relation_names()
    assert "uses" in relation_names
    assert "introduces-concept-of" in relation_names
    assert "benchmarked-against" in relation_names
    graph_defaults = config.ui.graph_defaults
    assert graph_defaults.min_confidence == 0.5
    assert graph_defaults.relations[:2] == ["defined-as", "uses"]
    assert graph_defaults.sections == ["Results", "Methods"]
    assert graph_defaults.show_co_mentions is False
    assert config.ui.polling.active_interval_seconds == 12
    assert config.ui.polling.idle_interval_seconds == 60
    assert config.qa.llm.enabled is True
    assert config.qa.llm.provider == "openai"
    qa_openai = config.qa.llm.openai
    assert qa_openai.prompt_version == "qa-v1"
    assert qa_openai.max_output_tokens == 600
    assert qa_openai.temperature == 0.1
    assert 429 in qa_openai.retry_statuses
    assert config.graph.relation_semantics["binds-to"] == "bidirectional"
    assert config.graph.relation_semantics["interacts-with"] == "bidirectional"
    assert config.graph.entity_batch_size == 200
    assert config.graph.edge_batch_size == 500
    assert config.observability.root_dir == "data/observability"
    assert config.observability.run_manifests_filename == "runs.jsonl"
    assert config.observability.audit_results_filename == "edge_audits.jsonl"
    assert config.observability.semantic_drift_filename == "semantic_drift.jsonl"
    assert config.observability.quality_alerts_filename == "quality_alerts.jsonl"
    quality = config.observability.quality
    assert quality.noise_control_target == 0.85
    assert quality.noise_control_warning == 0.9
    assert quality.duplicate_rate_target == 0.1
    assert quality.duplicate_rate_warning == 0.05
    assert quality.semantic_drift_drop_threshold == 0.25
    assert quality.semantic_drift_relation_threshold == 3


def test_config_strict_fields_match_yaml() -> None:
    config_path = AppConfig.default_path()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = load_config()
    assert raw["pipeline"]["version"] == config.pipeline.version
    assert raw["co_mention"]["confidence"] == config.co_mention.confidence
    assert raw["parsing"]["metadata_max_pages"] == config.parsing.metadata_max_pages
    assert raw["parsing"]["docling_artifacts_path"] == config.parsing.docling_artifacts_path
    assert raw["parsing"]["rapidocr"]["backend"] == config.parsing.rapidocr.backend
    assert raw["parsing"]["threaded_pdf"]["enabled"] == config.parsing.threaded_pdf.enabled
    assert (
        raw["ui"]["graph_defaults"]["min_confidence"]
        == config.ui.graph_defaults.min_confidence
    )
    assert raw["ui"]["polling"]["active_interval_seconds"] == config.ui.polling.active_interval_seconds
    assert (
        raw["observability"]["audit_results_filename"]
        == config.observability.audit_results_filename
    )
    assert (
        raw["observability"]["semantic_drift_filename"]
        == config.observability.semantic_drift_filename
    )
    assert (
        raw["observability"]["quality_alerts_filename"]
        == config.observability.quality_alerts_filename
    )
    assert (
        raw["observability"]["quality"]["noise_control_target"]
        == config.observability.quality.noise_control_target
    )
    assert (
        raw["observability"]["quality"]["noise_control_warning"]
        == config.observability.quality.noise_control_warning
    )
    assert (
        raw["observability"]["quality"]["duplicate_rate_target"]
        == config.observability.quality.duplicate_rate_target
    )
    assert (
        raw["observability"]["quality"]["duplicate_rate_warning"]
        == config.observability.quality.duplicate_rate_warning
    )
    assert (
        raw["observability"]["quality"]["semantic_drift_drop_threshold"]
        == config.observability.quality.semantic_drift_drop_threshold
    )
    assert (
        raw["observability"]["quality"]["semantic_drift_relation_threshold"]
        == config.observability.quality.semantic_drift_relation_threshold
    )


def test_google_client_ids_override_from_env(monkeypatch) -> None:
    load_config.cache_clear()
    monkeypatch.setenv("SCINETS_SKIP_ENV_FILE", "1")
    monkeypatch.delenv("NEXT_PUBLIC_GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.setenv(
        "SCINETS_AUTH_GOOGLE_CLIENT_IDS",
        "client-one.apps.googleusercontent.com, client-two.apps.googleusercontent.com",
    )
    try:
        config = load_config()
        assert config.auth.google.client_ids == [
            "client-one.apps.googleusercontent.com",
            "client-two.apps.googleusercontent.com",
        ]
    finally:
        load_config.cache_clear()


def test_google_client_ids_fallbacks_to_frontend_env(monkeypatch) -> None:
    load_config.cache_clear()
    monkeypatch.setenv("SCINETS_SKIP_ENV_FILE", "1")
    monkeypatch.setenv(
        "NEXT_PUBLIC_GOOGLE_CLIENT_ID",
        "client-only.apps.googleusercontent.com",
    )
    try:
        config = load_config()
        assert config.auth.google.client_ids == [
            "client-only.apps.googleusercontent.com",
        ]
    finally:
        load_config.cache_clear()


def test_google_client_ids_append_frontend_id_when_overridden(monkeypatch) -> None:
    load_config.cache_clear()
    monkeypatch.setenv("SCINETS_SKIP_ENV_FILE", "1")
    monkeypatch.setenv(
        "SCINETS_AUTH_GOOGLE_CLIENT_IDS",
        "client-one.apps.googleusercontent.com",
    )
    monkeypatch.setenv(
        "NEXT_PUBLIC_GOOGLE_CLIENT_ID",
        "client-two.apps.googleusercontent.com",
    )
    try:
        config = load_config()
        assert config.auth.google.client_ids == [
            "client-one.apps.googleusercontent.com",
            "client-two.apps.googleusercontent.com",
        ]
    finally:
        load_config.cache_clear()


def test_google_client_ids_loaded_from_env_file(monkeypatch, tmp_path) -> None:
    load_config.cache_clear()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Sample .env file\nNEXT_PUBLIC_GOOGLE_CLIENT_ID=client-from-env-file.apps.googleusercontent.com\n",
        encoding="utf-8",
    )
    for key in (
        "SCINETS_AUTH_GOOGLE_CLIENT_IDS",
        "GOOGLE_CLIENT_IDS",
        "GOOGLE_CLIENT_ID",
        "NEXT_PUBLIC_GOOGLE_CLIENT_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("SCINETS_SKIP_ENV_FILE", raising=False)
    monkeypatch.setenv("SCINETS_ENV_FILE", str(env_file))
    try:
        config = load_config()
        assert config.auth.google.client_ids == [
            "client-from-env-file.apps.googleusercontent.com",
        ]
    finally:
        load_config.cache_clear()
        monkeypatch.delenv("SCINETS_ENV_FILE", raising=False)
        monkeypatch.delenv("NEXT_PUBLIC_GOOGLE_CLIENT_ID", raising=False)


def test_env_file_overrides_blank_environment_value(monkeypatch, tmp_path) -> None:
    load_config.cache_clear()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "NEXT_PUBLIC_GOOGLE_CLIENT_ID=client-from-env-file.apps.googleusercontent.com\n",
        encoding="utf-8",
    )
    for key in (
        "SCINETS_AUTH_GOOGLE_CLIENT_IDS",
        "GOOGLE_CLIENT_IDS",
        "GOOGLE_CLIENT_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NEXT_PUBLIC_GOOGLE_CLIENT_ID", "   ")
    monkeypatch.setenv("SCINETS_ENV_FILE", str(env_file))
    try:
        config = load_config()
        assert config.auth.google.client_ids == [
            "client-from-env-file.apps.googleusercontent.com",
        ]
    finally:
        load_config.cache_clear()
        monkeypatch.delenv("SCINETS_ENV_FILE", raising=False)
        monkeypatch.delenv("NEXT_PUBLIC_GOOGLE_CLIENT_ID", raising=False)


