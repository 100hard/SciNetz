"""Tests for entity canonicalization logic."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pytest

import backend.app.canonicalization.entity_canonicalizer as canonicalizer_module
from backend.app.canonicalization import (
    CanonicalizationResult,
    CanonicalizationPipeline,
    E5EmbeddingBackend,
    EntityAggregator,
    EntityCandidate,
    EntityCanonicalizer,
)
from backend.app.config import AppConfig, load_config
from backend.app.contracts import Evidence, TextSpan, Triplet
from backend.app.extraction import ExtractionResult


class StubEmbeddingBackend:
    """Embedding backend that returns predefined unit vectors."""

    def __init__(self, vectors: Dict[str, Iterable[float]]) -> None:
        self._vectors = {key: np.array(value, dtype=np.float32) for key, value in vectors.items()}
        for key, vector in self._vectors.items():
            norm = np.linalg.norm(vector)
            if norm == 0:
                msg = f"Embedding for {key} must be non-zero"
                raise ValueError(msg)
            self._vectors[key] = vector / norm

    def embed(self, text: str) -> np.ndarray:
        normalized = text
        if normalized not in self._vectors:
            msg = f"No embedding for text {text}"
            raise KeyError(msg)
        return self._vectors[normalized]


@pytest.fixture(name="config")
def fixture_config() -> AppConfig:
    return load_config()


@pytest.fixture(name="storage_dirs")
def fixture_storage_dirs(tmp_path: Path) -> Dict[str, Path]:
    embeddings = tmp_path / "embeddings"
    reports = tmp_path / "canonicalization"
    return {"embeddings": embeddings, "reports": reports}


def test_e5_backend_formats_queries_and_caches(monkeypatch) -> None:
    calls: Dict[str, object] = {"encode": 0}

    class DummyModel:
        def __init__(self, model_name: str, device: str | None = None) -> None:
            calls["model_name"] = model_name
            calls["device"] = device

        def encode(self, sentences, **kwargs):
            calls["encode"] = int(calls["encode"]) + 1
            calls["sentences"] = list(sentences)
            calls["kwargs"] = kwargs
            return np.array([[3.0, 4.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(canonicalizer_module, "SentenceTransformer", DummyModel)
    backend = E5EmbeddingBackend(model_name="intfloat/e5-base", device="cpu")

    vector = backend.embed("Reinforcement Learning")
    assert np.isclose(np.linalg.norm(vector), 1.0)
    assert calls["sentences"] == ["query: Reinforcement Learning"]

    cached = backend.embed("Reinforcement Learning")
    assert calls["encode"] == 1
    np.testing.assert_allclose(vector, cached)




def test_e5_embedding_backend_reuses_global_model(monkeypatch) -> None:
    class DummyModel:
        instances = 0

        def __init__(self, model_name: str, device: str | None = None) -> None:
            DummyModel.instances += 1
            self.model_name = model_name
            self.device = device

        def encode(self, sentences, **_kwargs):
            return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(canonicalizer_module, "SentenceTransformer", DummyModel)
    E5EmbeddingBackend._GLOBAL_MODELS.clear()

    try:
        backend_one = E5EmbeddingBackend(model_name="intfloat/e5-base", device="cpu")
        backend_two = E5EmbeddingBackend(model_name="intfloat/e5-base", device="cpu")

        vec_one = backend_one.embed("alpha")
        vec_two = backend_two.embed("beta")

        assert DummyModel.instances == 1
        assert backend_one._get_model() is backend_two._get_model()
        assert np.isclose(np.linalg.norm(vec_one), 1.0)
        assert np.isclose(np.linalg.norm(vec_two), 1.0)
    finally:
        E5EmbeddingBackend._GLOBAL_MODELS.clear()


def test_canonicalizer_prefers_e5_backend_by_default(monkeypatch, config, storage_dirs) -> None:


    class DummyBackend(canonicalizer_module.EmbeddingBackend):
        instances = 0

        @classmethod
        def is_available(cls) -> bool:
            return True

        def __init__(self) -> None:
            DummyBackend.instances += 1

        def embed(self, text: str) -> np.ndarray:
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr(canonicalizer_module, "E5EmbeddingBackend", DummyBackend)

    canonicalizer = EntityCanonicalizer(
        config,
        embedding_dir=storage_dirs["embeddings"],
        report_dir=storage_dirs["reports"],
    )

    assert isinstance(canonicalizer._embedding_backend, DummyBackend)
    assert DummyBackend.instances == 1


def _candidate(
    name: str,
    *,
    times_seen: int,
    section_distribution: Dict[str, int],
    doc_ids: Iterable[str],
    entity_type: str = "Method",
) -> EntityCandidate:
    return EntityCandidate(
        name=name,
        type=entity_type,
        times_seen=times_seen,
        section_distribution=section_distribution,
        source_document_ids=list(doc_ids),
    )


def _triplet(
    subject: str,
    predicate: str,
    obj: str,
    *,
    doc_id: str,
    element_id: str,
) -> Triplet:
    return Triplet(
        subject=subject,
        predicate=predicate,
        object=obj,
        confidence=0.9,
        evidence=Evidence(
            element_id=element_id,
            doc_id=doc_id,
            text_span=TextSpan(start=0, end=9),
            full_sentence="Sentence.",
        ),
        pipeline_version="1.0.0",
    )


def test_synonyms_merge_into_single_node(config, storage_dirs) -> None:
    backend = StubEmbeddingBackend(
        {
            "reinforcement learning": [1.0, 0.0, 0.0],
            "rl": [1.0, 0.0, 0.0],
        }
    )
    canonicalizer = EntityCanonicalizer(
        config,
        embedding_backend=backend,
        embedding_dir=storage_dirs["embeddings"],
        report_dir=storage_dirs["reports"],
    )
    candidates = [
        _candidate(
            "Reinforcement Learning",
            times_seen=5,
            section_distribution={"Methods": 5},
            doc_ids=["doc1"],
        ),
        _candidate(
            "reinforcement learning",
            times_seen=3,
            section_distribution={"Methods": 3},
            doc_ids=["doc2"],
        ),
        _candidate(
            "RL",
            times_seen=2,
            section_distribution={"Results": 2},
            doc_ids=["doc3"],
        ),
    ]

    result = canonicalizer.canonicalize(candidates)

    assert len(result.nodes) == 1
    node = result.nodes[0]
    assert node.name == "Reinforcement Learning"
    assert sorted(node.aliases) == ["RL", "reinforcement learning"]
    assert node.times_seen == 10
    assert set(node.source_document_ids) == {"doc1", "doc2", "doc3"}
    assert result.merge_map["Reinforcement Learning"] == node.node_id
    assert result.merge_map["RL"] == node.node_id


def test_blocklisted_terms_require_matching_sections(config, storage_dirs) -> None:
    backend = StubEmbeddingBackend({"model": [1.0, 0.0, 0.0]})
    canonicalizer = EntityCanonicalizer(
        config,
        embedding_backend=backend,
        embedding_dir=storage_dirs["embeddings"],
        report_dir=storage_dirs["reports"],
    )
    candidates = [
        _candidate(
            "Model",
            times_seen=4,
            section_distribution={"Methods": 4},
            doc_ids=["doc-a"],
        ),
        _candidate(
            "Model",
            times_seen=3,
            section_distribution={"Results": 3},
            doc_ids=["doc-b"],
        ),
    ]

    result = canonicalizer.canonicalize(candidates)

    assert len(result.nodes) == 2


def test_distinct_terms_do_not_merge(config, storage_dirs) -> None:
    backend = StubEmbeddingBackend(
        {
            "bert": [1.0, 0.0, 0.0],
            "gpt-3": [0.0, 1.0, 0.0],
        }
    )
    canonicalizer = EntityCanonicalizer(
        config,
        embedding_backend=backend,
        embedding_dir=storage_dirs["embeddings"],
        report_dir=storage_dirs["reports"],
    )
    candidates = [
        _candidate("BERT", times_seen=4, section_distribution={"Methods": 4}, doc_ids=["doc1"]),
        _candidate("GPT-3", times_seen=6, section_distribution={"Methods": 6}, doc_ids=["doc2"]),
    ]

    result = canonicalizer.canonicalize(candidates)

    assert len(result.nodes) == 2
    names = sorted(node.name for node in result.nodes)
    assert names == ["BERT", "GPT-3"]


def test_polysemous_entities_require_section_overlap(config, storage_dirs) -> None:
    backend = StubEmbeddingBackend(
        {
            "transformer": [1.0, 0.0, 0.0],
        }
    )
    canonicalizer = EntityCanonicalizer(
        config,
        embedding_backend=backend,
        embedding_dir=storage_dirs["embeddings"],
        report_dir=storage_dirs["reports"],
    )
    candidates = [
        _candidate(
            "Transformer",
            times_seen=5,
            section_distribution={"Methods": 3, "Results": 2, "Introduction": 1},
            doc_ids=["doc1"],
        ),
        _candidate(
            "Transformer",
            times_seen=4,
            section_distribution={"Introduction": 4, "Background": 2, "Discussion": 1},
            doc_ids=["doc2"],
        ),
    ]

    result = canonicalizer.canonicalize(candidates)

    assert len(result.nodes) == 2
    keys = sorted(result.merge_map)
    assert keys == ["Transformer::Introduction|Background", "Transformer::Methods|Results"]
    ids = {result.merge_map[key] for key in keys}
    assert len(ids) == 2


def test_entity_aggregator_accumulates_entities() -> None:
    aggregator = EntityAggregator()
    extraction_a = ExtractionResult(
        triplets=[
            _triplet(
                "Reinforcement Learning",
                "uses",
                "Q-learning",
                doc_id="doc1",
                element_id="e1",
            )
        ],
        section_distribution={
            "Reinforcement Learning": {"Methods": 2},
            "Q-learning": {"Results": 1},
        },
    )
    extraction_b = ExtractionResult(
        triplets=[
            _triplet(
                "Reinforcement Learning",
                "compared-to",
                "Dynamic Programming",
                doc_id="doc2",
                element_id="e2",
            )
        ],
        section_distribution={
            "Reinforcement Learning": {"Results": 1},
            "Dynamic Programming": {"Discussion": 1},
        },
    )

    aggregator.extend([extraction_a, extraction_b])
    candidates = aggregator.build_candidates()

    rl_candidate = next(candidate for candidate in candidates if candidate.name == "Reinforcement Learning")
    assert rl_candidate.times_seen == 3
    assert rl_candidate.section_distribution == {"Methods": 2, "Results": 1}
    assert rl_candidate.source_document_ids == ["doc1", "doc2"]
    assert rl_candidate.type == "Unknown"


def test_entity_aggregator_uses_type_resolver() -> None:
    def resolver(name: str) -> str | None:
        if name.lower().startswith("reinforcement"):
            return "Method"
        return None

    aggregator = EntityAggregator(type_resolver=resolver)
    aggregator.ingest(
        ExtractionResult(
            triplets=[
                _triplet(
                    "Reinforcement Learning",
                    "uses",
                    "Q-learning",
                    doc_id="doc1",
                    element_id="e1",
                )
            ],
            section_distribution={
                "Reinforcement Learning": {"Methods": 2},
                "Q-learning": {"Results": 1},
            },
        )
    )

    candidate = next(candidate for candidate in aggregator.build_candidates() if candidate.name == "Reinforcement Learning")
    assert candidate.type == "Method"


def test_entity_aggregator_prefers_llm_type_votes() -> None:
    aggregator = EntityAggregator(type_resolver=lambda _: "Fallback")
    aggregator.ingest(
        ExtractionResult(
            triplets=[
                _triplet(
                    "Reinforcement Learning",
                    "uses",
                    "Q-learning",
                    doc_id="doc1",
                    element_id="e1",
                )
            ],
            section_distribution={
                "Reinforcement Learning": {"Methods": 1},
                "Q-learning": {"Results": 1},
            },
            entity_type_votes={
                "Reinforcement Learning": {"Method": 2},
                "Q-learning": {"Method": 1},
            },
        )
    )

    candidate = next(candidate for candidate in aggregator.build_candidates() if candidate.name == "Reinforcement Learning")
    assert candidate.type == "Method"


def test_canonicalization_pipeline_end_to_end(config, storage_dirs) -> None:
    backend = StubEmbeddingBackend(
        {
            "reinforcement learning": [1.0, 0.0, 0.0],
            "rl": [1.0, 0.0, 0.0],
            "q-learning": [0.0, 1.0, 0.0],
            "dynamic programming": [0.0, 0.0, 1.0],
        }
    )
    clock = lambda: datetime(2024, 1, 1, tzinfo=timezone.utc)
    canonicalizer = EntityCanonicalizer(
        config,
        embedding_backend=backend,
        embedding_dir=storage_dirs["embeddings"],
        report_dir=storage_dirs["reports"],
        clock=clock,
    )
    pipeline = CanonicalizationPipeline(
        config=config,
        canonicalizer=canonicalizer,
        aggregator_factory=lambda: EntityAggregator(default_type="Method"),
    )

    extraction_primary = ExtractionResult(
        triplets=[
            _triplet(
                "Reinforcement Learning",
                "uses",
                "Q-learning",
                doc_id="doc1",
                element_id="e1",
            )
        ],
        section_distribution={
            "Reinforcement Learning": {"Methods": 2},
            "Q-learning": {"Results": 1},
        },
    )
    extraction_alias = ExtractionResult(
        triplets=[
            _triplet(
                "RL",
                "compared-to",
                "Dynamic Programming",
                doc_id="doc2",
                element_id="e2",
            )
        ],
        section_distribution={
            "RL": {"Methods": 1},
            "Dynamic Programming": {"Discussion": 1},
        },
    )

    result = pipeline.run([extraction_primary, extraction_alias])

    rl_node = next(node for node in result.nodes if node.name == "Reinforcement Learning")
    assert rl_node.type == "Method"
    assert rl_node.times_seen == 3
    assert rl_node.aliases == ["RL"]
    assert rl_node.section_distribution["Methods"] == 3
    assert result.merge_map_path.exists()
    assert result.merge_report_path.exists()
    assert result.merge_report_path.name == "merge_report_20240101T000000Z.json"


def test_canonicalizer_is_idempotent(config, storage_dirs) -> None:
    backend = StubEmbeddingBackend(
        {
            "reinforcement learning": [1.0, 0.0, 0.0],
            "rl": [1.0, 0.0, 0.0],
        }
    )
    canonicalizer = EntityCanonicalizer(
        config,
        embedding_backend=backend,
        embedding_dir=storage_dirs["embeddings"],
        report_dir=storage_dirs["reports"],
    )
    candidates = [
        _candidate(
            "Reinforcement Learning",
            times_seen=5,
            section_distribution={"Methods": 5},
            doc_ids=["doc1"],
        ),
        _candidate(
            "RL",
            times_seen=2,
            section_distribution={"Methods": 2},
            doc_ids=["doc2"],
        ),
    ]

    first = canonicalizer.canonicalize(candidates)
    first_embeddings = sorted(path.name for path in storage_dirs["embeddings"].glob("*.npy"))

    second = canonicalizer.canonicalize(candidates)
    second_embeddings = sorted(path.name for path in storage_dirs["embeddings"].glob("*.npy"))

    def _normalize_nodes(result: CanonicalizationResult) -> list[dict[str, object]]:
        simplified = []
        for node in result.nodes:
            simplified.append(
                {
                    "id": node.node_id,
                    "name": node.name,
                    "type": node.type,
                    "aliases": tuple(sorted(node.aliases)),
                    "times_seen": node.times_seen,
                    "sections": tuple(sorted(node.section_distribution.items())),
                    "docs": tuple(sorted(node.source_document_ids)),
                }
            )
        return sorted(simplified, key=lambda item: item["id"])

    assert _normalize_nodes(first) == _normalize_nodes(second)
    assert first.merge_map == second.merge_map
    assert first_embeddings == second_embeddings


def test_canonicalizer_rejects_sentence_like_aliases(config, storage_dirs) -> None:
    backend = StubEmbeddingBackend(
        {
            "convolutional neural network": [1.0, 0.0, 0.0],
            "this novel convolutional neural network architecture dramatically improves accuracy": [1.0, 0.0, 0.0],
        }
    )
    tuned_config = config.model_copy(
        update={
            "canonicalization": config.canonicalization.model_copy(
                update={
                    "alias_token_limit": 6,
                    "alias_char_limit": 80,
                    "long_alias_penalty": 0.05,
                    "sentence_alias_penalty": 0.08,
                }
            ),
        }
    )
    canonicalizer = EntityCanonicalizer(
        tuned_config,
        embedding_backend=backend,
        embedding_dir=storage_dirs["embeddings"],
        report_dir=storage_dirs["reports"],
    )
    candidates = [
        _candidate(
            "Convolutional Neural Network",
            times_seen=4,
            section_distribution={"Methods": 4},
            doc_ids=["docA"],
        ),
        _candidate(
            "This novel convolutional neural network architecture dramatically improves accuracy",
            times_seen=2,
            section_distribution={"Methods": 2},
            doc_ids=["docB"],
        ),
    ]

    result = canonicalizer.canonicalize(candidates)

    assert len(result.nodes) == 2
    for node in result.nodes:
        assert not node.aliases


def test_canonicalizer_merges_concise_aliases_under_stricter_rules(config, storage_dirs) -> None:
    backend = StubEmbeddingBackend(
        {
            "reinforcement learning": [1.0, 0.0, 0.0],
            "rl": [1.0, 0.0, 0.0],
        }
    )
    tuned_config = config.model_copy(
        update={
            "canonicalization": config.canonicalization.model_copy(
                update={
                    "alias_token_limit": 6,
                    "alias_char_limit": 80,
                    "long_alias_penalty": 0.05,
                    "sentence_alias_penalty": 0.08,
                }
            ),
        }
    )
    canonicalizer = EntityCanonicalizer(
        tuned_config,
        embedding_backend=backend,
        embedding_dir=storage_dirs["embeddings"],
        report_dir=storage_dirs["reports"],
    )
    candidates = [
        _candidate(
            "Reinforcement Learning",
            times_seen=5,
            section_distribution={"Methods": 5},
            doc_ids=["doc1"],
        ),
        _candidate(
            "RL",
            times_seen=2,
            section_distribution={"Methods": 2},
            doc_ids=["doc2"],
        ),
    ]

    result = canonicalizer.canonicalize(candidates)

    assert any(node.name == "Reinforcement Learning" and node.aliases == ["RL"] for node in result.nodes)


def test_canonicalizer_rejects_context_only_aliases(config, storage_dirs) -> None:
    backend = StubEmbeddingBackend(
        {
            "word embedding vectors": [1.0, 0.0, 0.0],
            "english wikipedia": [1.0, 0.0, 0.0],
            "adam": [1.0, 0.0, 0.0],
            "4 days": [1.0, 0.0, 0.0],
        }
    )
    canonicalizer = EntityCanonicalizer(
        config,
        embedding_backend=backend,
        embedding_dir=storage_dirs["embeddings"],
        report_dir=storage_dirs["reports"],
    )
    candidates = [
        _candidate(
            "Word Embedding Vectors",
            times_seen=6,
            section_distribution={"Methods": 6, "Results": 3},
            doc_ids=["doc1"],
        ),
        _candidate(
            "English Wikipedia",
            times_seen=3,
            section_distribution={"Methods": 3},
            doc_ids=["doc2"],
            entity_type="Dataset",
        ),
        _candidate(
            "Adam",
            times_seen=2,
            section_distribution={"Methods": 2},
            doc_ids=["doc3"],
            entity_type="Method",
        ),
        _candidate(
            "4 days",
            times_seen=1,
            section_distribution={"Results": 1},
            doc_ids=["doc4"],
            entity_type="Metric",
        ),
    ]

    result = canonicalizer.canonicalize(candidates)

    target_nodes = {node.name: node for node in result.nodes}
    assert "Word Embedding Vectors" in target_nodes
    assert "English Wikipedia" in target_nodes
    assert "Adam" in target_nodes
    assert "4 days" in target_nodes
    assert not target_nodes["Word Embedding Vectors"].aliases
