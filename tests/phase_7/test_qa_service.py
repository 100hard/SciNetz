"""Unit tests for the graph-first QA service orchestration."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence

from backend.app.canonicalization.entity_canonicalizer import HashingEmbeddingBackend
from backend.app.config import AppConfig, load_config
from backend.app.qa import AnswerMode, QAService
from backend.app.qa.entity_resolution import (
    CandidateNode,
    QARepositoryProtocol,
    ResolvedCandidate,
    ResolvedEntity,
)
from backend.app.qa.repository import NeighborRecord, PathRecord


class _StubExtractor:
    """Deterministic extractor that returns configured entity mentions."""

    def __init__(self, known_entities: Sequence[str]) -> None:
        self._known = [name.lower() for name in known_entities]
        self._original = list(known_entities)

    def extract(self, question: str) -> List[str]:
        lowered = question.lower()
        return [original for original, token in zip(self._original, self._known) if token in lowered]


class _StubRepository(QARepositoryProtocol):
    """In-memory repository returning predetermined results."""

    def __init__(
        self,
        candidates: Sequence[CandidateNode],
        *,
        paths: Optional[Dict[tuple[str, str], Sequence[PathRecord]]] = None,
        neighbors: Optional[Dict[str, Sequence[NeighborRecord]]] = None,
    ) -> None:
        self._candidates = list(candidates)
        self._paths = paths or {}
        self._neighbors = neighbors or {}

    def fetch_nodes_by_exact_match(self, mention: str) -> Sequence[CandidateNode]:
        lowered = mention.lower()
        matches: List[CandidateNode] = []
        for node in self._candidates:
            if node.name.lower() == lowered:
                matches.append(node)
                continue
            if any(alias.lower() == lowered for alias in node.aliases):
                matches.append(node)
        return matches

    def fetch_candidate_nodes(self, limit: int) -> Sequence[CandidateNode]:
        return self._candidates[:limit]

    def fetch_paths(
        self,
        *,
        start_id: str,
        end_id: str,
        max_hops: int,
        min_confidence: float,
        limit: int,
        allowed_relations: Optional[Sequence[str]] = None,
    ) -> Sequence[PathRecord]:
        del max_hops, min_confidence, limit, allowed_relations
        return self._paths.get((start_id, end_id), [])

    def fetch_neighbors(
        self,
        node_id: str,
        *,
        min_confidence: float,
        limit: int,
        allowed_relations: Optional[Sequence[str]] = None,
    ) -> Sequence[NeighborRecord]:
        del min_confidence, limit, allowed_relations
        return self._neighbors.get(node_id, [])


def _with_config_updates(config: AppConfig, *, qa: Optional[Mapping[str, object]] = None) -> AppConfig:
    """Return a new immutable config instance with provided QA overrides."""

    qa_config = config.qa.model_copy(update=dict(qa or {}))
    return config.model_copy(update={"qa": qa_config})


def _make_candidate(node_id: str, name: str) -> CandidateNode:
    return CandidateNode(
        node_id=node_id,
        name=name,
        aliases=[f"alias-{name}"],
        times_seen=3,
        section_distribution={"Results": 2},
    )


def _path_record(conflicting: bool) -> PathRecord:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    nodes: Sequence[Mapping[str, object]] = (
        {"node_id": "alpha", "name": "Model Alpha", "section_distribution": {"Results": 2}},
        {"node_id": "beta", "name": "Model Beta", "section_distribution": {"Results": 1}},
    )
    relationships: Sequence[Mapping[str, object]] = (
        {
            "relation_norm": "outperforms",
            "relation_verbatim": "outperforms",
            "confidence": 0.92,
            "created_at": now,
            "conflicting": conflicting,
            "evidence": {
                "doc_id": "doc-alpha-beta",
                "element_id": "alpha:0",
                "text_span": {"start": 0, "end": 42},
                "full_sentence": "Model Alpha outperforms Model Beta.",
            },
            "attributes": {"method": "llm"},
        },
    )
    return PathRecord(nodes=nodes, relationships=relationships)


def _neighbor_record(source: str, target: str, relation: str, doc_id: str) -> NeighborRecord:
    now = datetime(2024, 1, 2, tzinfo=timezone.utc)
    source_node: Mapping[str, object] = {
        "node_id": source,
        "name": f"Node {source}",
        "section_distribution": {"Methods": 1},
    }
    target_node: Mapping[str, object] = {
        "node_id": target,
        "name": f"Node {target}",
        "section_distribution": {"Results": 1},
    }
    relationship: Mapping[str, object] = {
        "relation_norm": relation,
        "relation_verbatim": relation,
        "confidence": 0.8,
        "created_at": now,
        "conflicting": False,
        "evidence": {
            "doc_id": doc_id,
            "element_id": f"{source}:0",
            "text_span": {"start": 0, "end": 10},
            "full_sentence": "Some supporting statement.",
        },
        "attributes": {"method": "llm"},
    }
    return NeighborRecord(source=source_node, target=target_node, relationship=relationship)


def test_qa_service_detects_conflicting_paths() -> None:
    config = load_config()
    candidates = [_make_candidate("alpha", "Model Alpha"), _make_candidate("beta", "Model Beta")]
    repository = _StubRepository(candidates, paths={("alpha", "beta"): [_path_record(True)]})
    extractor = _StubExtractor(["Model Alpha", "Model Beta"])
    qa_service = QAService(
        config=config,
        repository=repository,
        embedding_backend=HashingEmbeddingBackend(),
        extractor=extractor,  # type: ignore[arg-type]
    )

    response = qa_service.answer("Does Model Alpha conflict with Model Beta?")

    assert response.mode == AnswerMode.CONFLICTING
    assert response.paths
    assert response.paths[0].edges[0].conflicting is True


def test_qa_service_handles_blank_questions() -> None:
    config = load_config()
    repository = _StubRepository([])
    qa_service = QAService(
        config=config,
        repository=repository,
        embedding_backend=HashingEmbeddingBackend(),
        extractor=_StubExtractor([]),  # type: ignore[arg-type]
    )

    response = qa_service.answer("   ")

    assert response.mode == AnswerMode.INSUFFICIENT
    assert response.summary == "Insufficient evidence to answer the question."
    assert response.paths == []
    assert response.fallback_edges == []


def test_qa_service_returns_related_findings_when_no_paths() -> None:
    config = load_config()
    config = _with_config_updates(config, qa={"expand_neighbors": False})
    candidates = [_make_candidate("omega", "Model Omega")]
    neighbor = _neighbor_record("omega", "delta", "uses", "doc-omega")
    repository = _StubRepository(candidates, neighbors={"omega": [neighbor]})
    extractor = _StubExtractor(["Model Omega"])
    qa_service = QAService(
        config=config,
        repository=repository,
        embedding_backend=HashingEmbeddingBackend(),
        extractor=extractor,  # type: ignore[arg-type]
    )

    response = qa_service.answer("What is known about Model Omega?")

    assert response.mode == AnswerMode.INSUFFICIENT
    assert response.fallback_edges
    assert "Related findings:" in response.summary
    assert "doc-omega" in response.summary
    assert "0.80" in response.summary


def test_collect_neighbors_aggregates_all_candidates() -> None:
    config = load_config()
    candidates = [_make_candidate("alpha", "Model Alpha"), _make_candidate("gamma", "Model Gamma")]
    neighbor_map = {
        "alpha": [_neighbor_record("alpha", "delta", "uses", "doc-alpha")],
        "gamma": [_neighbor_record("gamma", "epsilon", "evaluated-on", "doc-gamma")],
    }
    repository = _StubRepository(candidates, neighbors=neighbor_map)
    qa_service = QAService(
        config=config,
        repository=repository,
        embedding_backend=HashingEmbeddingBackend(),
        extractor=_StubExtractor([]),  # type: ignore[arg-type]
    )

    resolved = [
        ResolvedEntity(
            mention="Model Ambiguous",
            candidates=[
                ResolvedCandidate(
                    node_id="alpha",
                    name="Model Alpha",
                    aliases=("alias-alpha",),
                    times_seen=3,
                    section_distribution={"Results": 2},
                    similarity=0.95,
                ),
                ResolvedCandidate(
                    node_id="gamma",
                    name="Model Gamma",
                    aliases=("alias-gamma",),
                    times_seen=2,
                    section_distribution={"Methods": 1},
                    similarity=0.94,
                ),
            ],
        )
    ]

    edges = qa_service._collect_neighbors(resolved)

    doc_ids = {edge.evidence.doc_id for edge in edges}
    assert doc_ids == {"doc-alpha", "doc-gamma"}
    assert all(edge.confidence >= config.qa.neighbor_confidence_threshold for edge in edges)
