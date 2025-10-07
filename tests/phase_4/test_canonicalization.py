"""Tests for entity canonicalization logic."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pytest

from backend.app.canonicalization import EntityCandidate, EntityCanonicalizer
from backend.app.config import AppConfig, load_config


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
