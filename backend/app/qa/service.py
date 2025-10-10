"""Graph-first QA service orchestrating resolution and path search."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from itertools import combinations
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field

from backend.app.canonicalization.entity_canonicalizer import (
    E5EmbeddingBackend,
    EmbeddingBackend,
    HashingEmbeddingBackend,
)
from backend.app.config import AppConfig
from backend.app.qa.entity_resolution import (
    EntityResolver,
    QuestionEntityExtractor,
    QARepositoryProtocol,
    ResolvedCandidate,
    ResolvedEntity,
)
from backend.app.qa.repository import NeighborRecord, PathRecord

LOGGER = logging.getLogger(__name__)

_SECTION_WEIGHTS = {
    "Results": 1.2,
    "Discussion": 1.1,
    "Methods": 1.1,
}

class AnswerMode(str, Enum):
    """Enumeration describing QA response modes."""

    DIRECT = "direct"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"


class EvidenceModel(BaseModel):
    """Evidence payload returned to clients."""

    doc_id: str = Field(..., min_length=1)
    element_id: str = Field(..., min_length=1)
    text_span: Mapping[str, int]
    full_sentence: Optional[str] = None


class PathEdgeModel(BaseModel):
    """Single edge in a discovered reasoning path."""

    src_id: str
    src_name: str
    dst_id: str
    dst_name: str
    relation: str
    relation_verbatim: str
    confidence: float
    created_at: datetime
    conflicting: bool
    evidence: EvidenceModel
    attributes: Mapping[str, str]


class PathModel(BaseModel):
    """Reasoning path connecting resolved entities."""

    edges: Sequence[PathEdgeModel]
    confidence_product: float
    section_score: float
    score: float
    latest_timestamp: datetime


class CandidateModel(BaseModel):
    """Candidate entity returned for each mention."""

    node_id: str
    name: str
    aliases: Sequence[str]
    times_seen: int
    section_distribution: Mapping[str, int]
    similarity: float
    selected: bool


class ResolvedEntityModel(BaseModel):
    """Resolved question mention and candidates."""

    mention: str
    candidates: Sequence[CandidateModel]


class QAResponse(BaseModel):
    """Response payload returned by the QA endpoint."""

    mode: AnswerMode
    summary: str
    resolved_entities: Sequence[ResolvedEntityModel]
    paths: Sequence[PathModel]
    fallback_edges: Sequence[PathEdgeModel]


@dataclass(frozen=True)
class _PathScore:
    path: PathModel
    conflicting: bool


class QAService:
    """High-level QA service responsible for resolving and answering questions."""

    def __init__(
        self,
        *,
        config: AppConfig,
        repository: QARepositoryProtocol,
        embedding_backend: Optional[EmbeddingBackend] = None,
        extractor: Optional[QuestionEntityExtractor] = None,
        max_neighbor_results: int = 10,
    ) -> None:
        self._config = config
        self._repository = repository
        self._embedding_backend = embedding_backend or self._create_embedding_backend()
        self._extractor = extractor or QuestionEntityExtractor()
        self._resolver = EntityResolver(
            repository=repository,
            embedding_backend=self._embedding_backend,
            similarity_threshold=self._config.qa.entity_match_threshold,
        )
        self._max_neighbor_results = max_neighbor_results
        relation_names = self._config.relations.canonical_relation_names()
        self._allowed_relations: Optional[Tuple[str, ...]] = (
            tuple(relation_names) if relation_names else None
        )

    def answer(self, question: str) -> QAResponse:
        """Answer the supplied question using the knowledge graph."""

        question_text = question.strip()
        if not question_text:
            return QAResponse(
                mode=AnswerMode.INSUFFICIENT,
                summary="Insufficient evidence to answer the question.",
                resolved_entities=[],
                paths=[],
                fallback_edges=[],
            )

        mentions = self._extractor.extract(question_text)
        resolved_entities = [self._resolver.resolve(mention) for mention in mentions]
        resolved_models = [self._to_model(entity) for entity in resolved_entities]
        primary_candidates = [entity.candidates[0] for entity in resolved_entities if entity.candidates]

        if not primary_candidates:
            summary = "Unable to resolve any entities for the question."
            return QAResponse(
                mode=AnswerMode.INSUFFICIENT,
                summary=summary,
                resolved_entities=resolved_models,
                paths=[],
                fallback_edges=[],
            )

        path_scores = self._discover_paths(resolved_entities)
        paths = [score.path for score in path_scores]
        has_conflict = any(score.conflicting for score in path_scores)

        if paths:
            summary = self._summarize_path(paths[0])
            mode = AnswerMode.CONFLICTING if has_conflict else AnswerMode.DIRECT
            return QAResponse(
                mode=mode,
                summary=summary,
                resolved_entities=resolved_models,
                paths=paths[: self._config.qa.max_results],
                fallback_edges=[],
            )

        fallback_edges = self._collect_neighbors(primary_candidates)
        summary = "Insufficient evidence to answer. Showing related findings."
        return QAResponse(
            mode=AnswerMode.INSUFFICIENT,
            summary=summary,
            resolved_entities=resolved_models,
            paths=[],
            fallback_edges=fallback_edges,
        )

    def _discover_paths(self, entities: Sequence[ResolvedEntity]) -> List[_PathScore]:
        results: List[_PathScore] = []
        seen_edges: set[Tuple[str, ...]] = set()
        candidate_pairs = [
            (first, second)
            for first, second in combinations(entities, 2)
            if first.candidates and second.candidates
        ]
        for first, second in candidate_pairs:
            for candidate_a in first.candidates:
                for candidate_b in second.candidates:
                    path_records = self._repository.fetch_paths(
                        start_id=candidate_a.node_id,
                        end_id=candidate_b.node_id,
                        max_hops=self._config.qa.max_hops,
                        min_confidence=self._config.qa.neighbor_confidence_threshold,
                        limit=self._config.qa.max_results,
                        allowed_relations=self._allowed_relations,
                    )
                    for record in path_records:
                        path = self._build_path(record)
                        edge_signature = tuple(
                            f"{edge.src_id}->{edge.dst_id}:{edge.relation}:{edge.evidence.doc_id}:{edge.evidence.element_id}"
                            for edge in path.edges
                        )
                        if edge_signature in seen_edges:
                            continue
                        seen_edges.add(edge_signature)
                        conflicting = any(edge.conflicting for edge in path.edges)
                        results.append(_PathScore(path=path, conflicting=conflicting))
        results.sort(key=lambda item: (-item.path.score, -item.path.latest_timestamp.timestamp()))
        return results

    def _collect_neighbors(self, candidates: Sequence[ResolvedCandidate]) -> List[PathEdgeModel]:
        edges: List[PathEdgeModel] = []
        for candidate in candidates:
            neighbor_records = self._repository.fetch_neighbors(
                candidate.node_id,
                min_confidence=self._config.qa.neighbor_confidence_threshold,
                limit=self._max_neighbor_results,
                allowed_relations=self._allowed_relations,
            )
            for record in neighbor_records:
                edge = self._neighbor_to_edge(record)
                edges.append(edge)
        unique: Dict[Tuple[str, str, str], PathEdgeModel] = {}
        for edge in edges:
            key = (edge.src_id, edge.dst_id, edge.relation)
            if key not in unique:
                unique[key] = edge
        sorted_edges = sorted(
            unique.values(),
            key=lambda edge: (-edge.confidence, edge.created_at.timestamp()),
        )
        return sorted_edges[: self._config.qa.max_results]

    def _build_path(self, record: PathRecord) -> PathModel:
        node_lookup = [self._normalize_node(node) for node in record.nodes]
        edges: List[PathEdgeModel] = []
        confidence_product = 1.0
        section_score = 1.0
        latest_ts = datetime.fromtimestamp(0)
        for idx, rel in enumerate(record.relationships):
            start_node = node_lookup[idx]
            end_node = node_lookup[idx + 1]
            created_at = _parse_datetime(rel.get("created_at"))
            latest_ts = max(latest_ts, created_at)
            confidence = float(rel.get("confidence", 0.0))
            confidence_product *= confidence
            section_score *= self._section_weight(start_node) * self._section_weight(end_node)
            edge = PathEdgeModel(
                src_id=str(start_node["node_id"]),
                src_name=str(start_node["name"]),
                dst_id=str(end_node["node_id"]),
                dst_name=str(end_node["name"]),
                relation=str(rel.get("relation_norm", "")),
                relation_verbatim=str(rel.get("relation_verbatim", rel.get("relation_norm", ""))),
                confidence=confidence,
                created_at=created_at,
                conflicting=bool(rel.get("conflicting", False)),
                evidence=self._extract_evidence(rel.get("evidence", {})),
                attributes=self._extract_attributes(rel.get("attributes", {})),
            )
            edges.append(edge)
        score = confidence_product * section_score
        return PathModel(
            edges=edges,
            confidence_product=confidence_product,
            section_score=section_score,
            score=score,
            latest_timestamp=latest_ts,
        )

    def _neighbor_to_edge(self, record: NeighborRecord) -> PathEdgeModel:
        created_at = _parse_datetime(record.relationship.get("created_at"))
        return PathEdgeModel(
            src_id=str(record.source.get("node_id")),
            src_name=str(record.source.get("name")),
            dst_id=str(record.target.get("node_id")),
            dst_name=str(record.target.get("name")),
            relation=str(record.relationship.get("relation_norm", "")),
            relation_verbatim=str(
                record.relationship.get("relation_verbatim", record.relationship.get("relation_norm", ""))
            ),
            confidence=float(record.relationship.get("confidence", 0.0)),
            created_at=created_at,
            conflicting=bool(record.relationship.get("conflicting", False)),
            evidence=self._extract_evidence(record.relationship.get("evidence", {})),
            attributes=self._extract_attributes(record.relationship.get("attributes", {})),
        )

    def _extract_evidence(self, payload: Mapping[str, object]) -> EvidenceModel:
        text_span = payload.get("text_span", {})
        return EvidenceModel(
            doc_id=str(payload.get("doc_id", "")),
            element_id=str(payload.get("element_id", "")),
            text_span={
                "start": int(text_span.get("start", 0)),
                "end": int(text_span.get("end", 0)),
            },
            full_sentence=payload.get("full_sentence"),
        )

    @staticmethod
    def _extract_attributes(payload: Mapping[str, object]) -> Mapping[str, str]:
        return {str(key): str(value) for key, value in payload.items()}

    def _normalize_node(self, node: Mapping[str, object]) -> Mapping[str, object]:
        return {
            "node_id": str(node.get("node_id")),
            "name": str(node.get("name", "")),
            "section_distribution": dict(node.get("section_distribution", {})),
        }

    def _section_weight(self, node: Mapping[str, object]) -> float:
        distribution = node.get("section_distribution", {})
        if not isinstance(distribution, Mapping):
            return 1.0
        weights = [
            weight
            for section, weight in _SECTION_WEIGHTS.items()
            if float(distribution.get(section, 0)) > 0
        ]
        if not weights:
            return 1.0
        return float(np.mean(weights))

    def _summarize_path(self, path: PathModel) -> str:
        segments = [
            f"{edge.src_name} {edge.relation} {edge.dst_name} (doc {edge.evidence.doc_id}, conf={edge.confidence:.2f})"
            for edge in path.edges
        ]
        return "; ".join(segments)

    def _to_model(self, entity: ResolvedEntity) -> ResolvedEntityModel:
        candidates = []
        for idx, candidate in enumerate(entity.candidates):
            candidates.append(
                CandidateModel(
                    node_id=candidate.node_id,
                    name=candidate.name,
                    aliases=list(candidate.aliases),
                    times_seen=candidate.times_seen,
                    section_distribution=dict(candidate.section_distribution),
                    similarity=candidate.similarity,
                    selected=idx == 0,
                )
            )
        return ResolvedEntityModel(mention=entity.mention, candidates=candidates)

    @staticmethod
    def _create_embedding_backend() -> EmbeddingBackend:
        if E5EmbeddingBackend.is_available():  # type: ignore[attr-defined]
            try:
                return E5EmbeddingBackend()
            except Exception:  # noqa: BLE001 - dependency issues fallback
                LOGGER.exception("Failed to initialize E5 embeddings; falling back to hashing backend")
        return HashingEmbeddingBackend()


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_native"):
        native = value.to_native()  # type: ignore[call-arg]
        if isinstance(native, datetime):
            return native
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return datetime(
            int(getattr(value, "year")),
            int(getattr(value, "month")),
            int(getattr(value, "day")),
            int(getattr(value, "hour", 0)),
            int(getattr(value, "minute", 0)),
            int(getattr(value, "second", 0)),
        )
    return datetime.fromtimestamp(0)
