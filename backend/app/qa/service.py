"""Graph-first QA service orchestrating resolution and path search."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from itertools import combinations
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field

from backend.app.canonicalization.entity_canonicalizer import (
    E5EmbeddingBackend,
    EmbeddingBackend,
    HashingEmbeddingBackend,
)
from backend.app.config import AppConfig
from backend.app.qa.answer_synthesis import (
    AnswerContextEdge,
    AnswerSynthesisRequest,
    LLMAnswerSynthesizer,
)
from backend.app.qa.entity_resolution import (
    EntityResolver,
    QuestionEntityExtractor,
    QARepositoryProtocol,
    ResolvedCandidate,
    ResolvedEntity,
)
from backend.app.qa.intent import IntentClassification, QAIntent, QueryIntentClassifier
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
    llm_answer: Optional[str] = None


@dataclass(frozen=True)
class _PathScore:
    path: PathModel
    conflicting: bool


@dataclass(frozen=True)
class QAStreamEvent:
    """Event emitted during streaming QA responses."""

    type: str
    payload: object


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
        answer_synthesizer: Optional[LLMAnswerSynthesizer] = None,
        intent_classifier: Optional[QueryIntentClassifier] = None,
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
        self._synthesizer = answer_synthesizer
        self._pipeline_version = self._config.pipeline.version
        self._intent_classifier = intent_classifier or QueryIntentClassifier(self._config.qa.intent)
        self._summary_edge_limit = max(1, self._config.qa.intent.max_summary_edges)
        self._allow_llm_without_evidence = self._config.qa.llm.allow_fallback_without_evidence

    def answer(self, question: str) -> QAResponse:
        """Answer the supplied question using the knowledge graph."""

        final_response: Optional[QAResponse] = None
        for event in self._iter_events(question):
            if event.type == "final":
                payload = event.payload
                if isinstance(payload, QAResponse):
                    final_response = payload
                else:
                    serialized = self._serialize_payload(payload)
                    final_response = QAResponse(**serialized)  # type: ignore[arg-type]
                break
        if final_response is None:
            msg = "QAService failed to produce a final response"
            raise RuntimeError(msg)
        return final_response

    def stream_answer(self, question: str) -> Iterator[bytes]:
        """Return a byte iterator producing SSE-formatted QA events."""

        return (self._format_stream_event(event) for event in self._iter_events(question))

    def _empty_response(self) -> QAResponse:
        return QAResponse(
            mode=AnswerMode.INSUFFICIENT,
            summary="Insufficient evidence to answer the question.",
            resolved_entities=[],
            paths=[],
            fallback_edges=[],
            llm_answer=None,
        )

    def _classify(self, question: str) -> IntentClassification:
        classifier = getattr(self, "_intent_classifier", None)
        if classifier is None:
            return IntentClassification(intent=QAIntent.FACTOID)
        try:
            return classifier.classify(question)
        except Exception:  # noqa: BLE001 - intent detection must fail softly
            LOGGER.exception("QA intent classification failed; defaulting to factoid mode")
            return IntentClassification(intent=QAIntent.FACTOID)

    def _iter_events(self, question: str) -> Iterator[QAStreamEvent]:
        question_text = question.strip()
        if not question_text:
            yield QAStreamEvent("final", self._empty_response())
            return

        classification = self._classify(question_text)
        yield QAStreamEvent("classification", classification)

        mentions = self._extractor.extract(question_text)
        resolved_entities = [self._resolver.resolve(mention) for mention in mentions]
        resolved_models = [self._to_model(entity) for entity in resolved_entities]
        yield QAStreamEvent("entities", resolved_models)

        if classification.intent == QAIntent.PAPER_SUMMARY:
            yield from self._answer_paper_summary_stream(
                question_text,
                classification,
                resolved_entities,
                resolved_models,
            )
            return
        if classification.intent == QAIntent.CLUSTER_SUMMARY:
            yield from self._answer_cluster_summary_stream(
                question_text,
                resolved_entities,
                resolved_models,
            )
            return
        if classification.intent == QAIntent.ENTITY_SUMMARY:
            yield from self._answer_entity_summary_stream(
                question_text,
                resolved_entities,
                resolved_models,
            )
            return
        yield from self._answer_factoid_stream(
            question_text,
            resolved_entities,
            resolved_models,
        )

    def _answer_factoid_stream(
        self,
        question_text: str,
        resolved_entities: Sequence[ResolvedEntity],
        resolved_models: Sequence[ResolvedEntityModel],
    ) -> Iterator[QAStreamEvent]:
        primary_candidates = [
            entity.candidates[0] for entity in resolved_entities if entity.candidates
        ]
        if not primary_candidates:
            summary = "Unable to resolve any entities for the question."
            response = QAResponse(
                mode=AnswerMode.INSUFFICIENT,
                summary=summary,
                resolved_entities=resolved_models,
                paths=[],
                fallback_edges=[],
                llm_answer=None,
            )
            yield QAStreamEvent("final", response)
            return

        path_scores = self._discover_paths(resolved_entities)
        top_scores = path_scores[: self._config.qa.max_results]
        paths = [score.path for score in top_scores]
        has_conflict = any(score.conflicting for score in top_scores)

        if paths:
            yield QAStreamEvent("paths", paths)
            mode = AnswerMode.CONFLICTING if has_conflict else AnswerMode.DIRECT
            summary = self._summarize_path(paths[0])
            llm_answer = None
            llm_generator = self._stream_llm_answer(
                question_text,
                mode,
                paths,
                [],
                resolved_entities,
            )
            if llm_generator is not None:
                llm_answer = yield from llm_generator
            if llm_answer:
                yield QAStreamEvent("llm_answer", llm_answer)
            response = QAResponse(
                mode=mode,
                summary=summary,
                resolved_entities=resolved_models,
                paths=paths,
                fallback_edges=[],
                llm_answer=llm_answer,
            )
            yield QAStreamEvent("final", response)
            return

        fallback_edges = self._collect_neighbors(resolved_entities)
        if fallback_edges:
            yield QAStreamEvent("fallback", fallback_edges)
            summary = self._summarize_neighbors(fallback_edges)
        else:
            summary = self._summarize_entity_profiles(resolved_entities)
        llm_answer = None
        llm_generator = self._stream_llm_answer(
            question_text,
            AnswerMode.INSUFFICIENT,
            [],
            fallback_edges,
            resolved_entities,
        )
        if llm_generator is not None:
            llm_answer = yield from llm_generator
        llm_answer = self._maybe_suppress_insufficient(llm_answer, fallback_edges)
        if llm_answer:
            yield QAStreamEvent("llm_answer", llm_answer)
        response = QAResponse(
            mode=AnswerMode.INSUFFICIENT,
            summary=summary,
            resolved_entities=resolved_models,
            paths=[],
            fallback_edges=fallback_edges,
            llm_answer=llm_answer,
        )
        yield QAStreamEvent("final", response)

    def _answer_entity_summary_stream(
        self,
        question_text: str,
        resolved_entities: Sequence[ResolvedEntity],
        resolved_models: Sequence[ResolvedEntityModel],
    ) -> Iterator[QAStreamEvent]:
        primary = self._select_primary_candidate(resolved_entities)
        if primary is None:
            yield from self._answer_factoid_stream(question_text, resolved_entities, resolved_models)
            return
        edges = self._collect_summary_neighbors([primary])
        if not edges:
            summary = self._summarize_entity_profiles(resolved_entities)
            llm_answer = None
            if self._allow_llm_without_evidence:
                generator = self._stream_llm_answer(
                    question_text,
                    AnswerMode.INSUFFICIENT,
                    [],
                    [],
                    resolved_entities,
                )
                if generator is not None:
                    llm_answer = yield from generator
            response = QAResponse(
                mode=AnswerMode.INSUFFICIENT,
                summary=summary,
                resolved_entities=resolved_models,
                paths=[],
                fallback_edges=[],
                llm_answer=llm_answer,
            )
            if llm_answer:
                yield QAStreamEvent("llm_answer", llm_answer)
            yield QAStreamEvent("final", response)
            return
        yield QAStreamEvent("fallback", edges)
        summary = f"Summary for {primary.name} derived from {len(edges)} graph findings."
        llm_answer = None
        generator = self._stream_llm_answer(
            question_text,
            AnswerMode.DIRECT,
            [],
            edges,
            resolved_entities,
        )
        if generator is not None:
            llm_answer = yield from generator
        llm_answer = self._maybe_suppress_insufficient(llm_answer, edges)
        if llm_answer:
            yield QAStreamEvent("llm_answer", llm_answer)
        response = QAResponse(
            mode=AnswerMode.DIRECT,
            summary=summary,
            resolved_entities=resolved_models,
            paths=[],
            fallback_edges=edges,
            llm_answer=llm_answer,
        )
        yield QAStreamEvent("final", response)
        return

    def _answer_cluster_summary_stream(
        self,
        question_text: str,
        resolved_entities: Sequence[ResolvedEntity],
        resolved_models: Sequence[ResolvedEntityModel],
    ) -> Iterator[QAStreamEvent]:
        candidates = self._unique_candidates(resolved_entities)
        if not candidates:
            yield from self._answer_factoid_stream(question_text, resolved_entities, resolved_models)
            return
        edges = self._collect_summary_neighbors(candidates)
        if not edges:
            summary = self._summarize_entity_profiles(resolved_entities)
            llm_answer = None
            if self._allow_llm_without_evidence:
                generator = self._stream_llm_answer(
                    question_text,
                    AnswerMode.INSUFFICIENT,
                    [],
                    [],
                    resolved_entities,
                )
                if generator is not None:
                    llm_answer = yield from generator
            response = QAResponse(
                mode=AnswerMode.INSUFFICIENT,
                summary=summary,
                resolved_entities=resolved_models,
                paths=[],
                fallback_edges=[],
                llm_answer=llm_answer,
            )
            if llm_answer:
                yield QAStreamEvent("llm_answer", llm_answer)
            yield QAStreamEvent("final", response)
            return
        yield QAStreamEvent("fallback", edges)
        cluster_names = ", ".join(candidate.name for candidate in candidates[:3])
        summary = (
            f"Cluster summary for {cluster_names} derived from {len(edges)} graph findings."
        )
        llm_answer = None
        generator = self._stream_llm_answer(
            question_text,
            AnswerMode.DIRECT,
            [],
            edges,
            resolved_entities,
        )
        if generator is not None:
            llm_answer = yield from generator
        llm_answer = self._maybe_suppress_insufficient(llm_answer, edges)
        if llm_answer:
            yield QAStreamEvent("llm_answer", llm_answer)
        response = QAResponse(
            mode=AnswerMode.DIRECT,
            summary=summary,
            resolved_entities=resolved_models,
            paths=[],
            fallback_edges=edges,
            llm_answer=llm_answer,
        )
        yield QAStreamEvent("final", response)

    def _answer_paper_summary_stream(
        self,
        question_text: str,
        classification: IntentClassification,
        resolved_entities: Sequence[ResolvedEntity],
        resolved_models: Sequence[ResolvedEntityModel],
    ) -> Iterator[QAStreamEvent]:
        document_ids = classification.document_ids
        if not document_ids:
            summary = "Unable to identify which document to summarize."
            response = QAResponse(
                mode=AnswerMode.INSUFFICIENT,
                summary=summary,
                resolved_entities=resolved_models,
                paths=[],
                fallback_edges=[],
                llm_answer=None,
            )
            yield QAStreamEvent("final", response)
            return
        edges = self._collect_document_edges(document_ids)
        if not edges:
            joined = ", ".join(document_ids)
            summary = f"No extracted findings available for documents: {joined}."
            llm_answer = None
            if self._allow_llm_without_evidence:
                generator = self._stream_llm_answer(
                    question_text,
                    AnswerMode.INSUFFICIENT,
                    [],
                    [],
                    resolved_entities,
                )
                if generator is not None:
                    llm_answer = yield from generator
            response = QAResponse(
                mode=AnswerMode.INSUFFICIENT,
                summary=summary,
                resolved_entities=resolved_models,
                paths=[],
                fallback_edges=[],
                llm_answer=llm_answer,
            )
            if llm_answer:
                yield QAStreamEvent("llm_answer", llm_answer)
            yield QAStreamEvent("final", response)
            return
        yield QAStreamEvent("fallback", edges)
        joined = ", ".join(document_ids)
        summary = f"Summary for documents {joined} derived from {len(edges)} graph findings."
        llm_answer = None
        generator = self._stream_llm_answer(
            question_text,
            AnswerMode.DIRECT,
            [],
            edges,
            resolved_entities,
        )
        if generator is not None:
            llm_answer = yield from generator
        llm_answer = self._maybe_suppress_insufficient(llm_answer, edges)
        if llm_answer:
            yield QAStreamEvent("llm_answer", llm_answer)
        response = QAResponse(
            mode=AnswerMode.DIRECT,
            summary=summary,
            resolved_entities=resolved_models,
            paths=[],
            fallback_edges=edges,
            llm_answer=llm_answer,
        )
        yield QAStreamEvent("final", response)

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

    def _collect_neighbors(self, entities: Sequence[ResolvedEntity]) -> List[PathEdgeModel]:
        if not self._config.qa.expand_neighbors:
            LOGGER.debug(
                "QA expand_neighbors disabled in config; still collecting 1-hop neighbors"
            )

        unique_candidates = self._unique_candidates(entities)

        if not unique_candidates:
            return []

        edges: List[PathEdgeModel] = []
        thresholds: List[float] = [self._config.qa.neighbor_confidence_threshold]
        secondary_threshold = max(0.35, thresholds[0] - 0.25)
        if secondary_threshold < thresholds[0] - 1e-6:
            thresholds.append(secondary_threshold)

        for threshold in thresholds:
            edges.clear()
            for candidate in unique_candidates:
                neighbor_records = self._repository.fetch_neighbors(
                    candidate.node_id,
                    min_confidence=threshold,
                    limit=self._max_neighbor_results,
                    allowed_relations=self._allowed_relations,
                )
                for record in neighbor_records:
                    edge = self._neighbor_to_edge(record)
                    edges.append(edge)
            if edges:
                if threshold != thresholds[0]:
                    LOGGER.debug(
                        "Expanded QA neighbor search using relaxed confidence %.2f (initial %.2f)",
                        threshold,
                        thresholds[0],
                    )
                break
        unique: Dict[Tuple[str, str, str], PathEdgeModel] = {}
        for edge in edges:
            key = (edge.src_id, edge.dst_id, edge.relation)
            if key not in unique:
                unique[key] = edge
        sorted_edges = sorted(
            unique.values(),
            key=lambda edge: (-edge.confidence, -edge.created_at.timestamp()),
        )
        return sorted_edges[: self._config.qa.max_results]

    def _collect_summary_neighbors(
        self,
        candidates: Sequence[ResolvedCandidate],
    ) -> List[PathEdgeModel]:
        if not candidates:
            return []
        limit = self._summary_edge_limit
        candidate_count = len(candidates)
        per_candidate_limit = max(1, (limit + candidate_count - 1) // candidate_count)
        unique: Dict[Tuple[str, str, str, str, str], PathEdgeModel] = {}
        for candidate in candidates:
            neighbor_records = self._repository.fetch_neighbors(
                candidate.node_id,
                min_confidence=self._config.qa.neighbor_confidence_threshold,
                limit=per_candidate_limit,
                allowed_relations=(),
            )
            for record in neighbor_records:
                edge = self._neighbor_to_edge(record)
                key = (
                    edge.src_id,
                    edge.dst_id,
                    edge.relation,
                    edge.evidence.doc_id,
                    edge.evidence.element_id,
                )
                existing = unique.get(key)
                if existing is None or edge.confidence > existing.confidence:
                    unique[key] = edge
        edges = sorted(
            unique.values(),
            key=lambda edge: (-edge.confidence, -edge.created_at.timestamp()),
        )
        return edges[:limit]

    def _prepare_synthesis_request(
        self,
        question: str,
        mode: AnswerMode,
        paths: Sequence[PathModel],
        fallback_edges: Sequence[PathEdgeModel],
        resolved_entities: Sequence[ResolvedEntity],
    ) -> Optional[Tuple[AnswerSynthesisRequest, LLMAnswerSynthesizer]]:
        synthesizer = self._synthesizer
        if synthesizer is None or not getattr(synthesizer, "enabled", False):
            return None
        question_text = question.strip()
        if not question_text:
            return None
        edge_count = sum(len(path.edges) for path in paths) + len(fallback_edges)
        if edge_count == 0 and not self._allow_llm_without_evidence:
            return None

        path_context = self._build_path_context(paths)
        neighbor_context = [self._edge_to_context(edge) for edge in fallback_edges]

        has_graph_evidence = bool(path_context or neighbor_context)
        if not has_graph_evidence and not self._allow_llm_without_evidence:
            return None

        scope_documents = self._collect_scope_documents(path_context, neighbor_context, resolved_entities)
        request = AnswerSynthesisRequest(
            question=question_text,
            mode=mode.value,
            graph_paths=path_context,
            neighbor_edges=neighbor_context,
            scope_documents=scope_documents,
            pipeline_version=self._pipeline_version,
            has_graph_evidence=has_graph_evidence,
            allow_off_graph_answer=self._allow_llm_without_evidence,
        )
        return request, synthesizer

    def _collect_document_edges(self, document_ids: Sequence[str]) -> List[PathEdgeModel]:
        if not document_ids:
            return []
        limit = self._summary_edge_limit
        doc_count = len(document_ids)
        per_doc_limit = max(1, (limit + doc_count - 1) // doc_count)
        unique: Dict[Tuple[str, str, str, str, str], PathEdgeModel] = {}
        for doc_id in document_ids:
            if not doc_id:
                continue
            neighbor_records = self._repository.fetch_document_edges(
                doc_id,
                min_confidence=self._config.qa.neighbor_confidence_threshold,
                limit=per_doc_limit,
                allowed_relations=(),
            )
            for record in neighbor_records:
                edge = self._neighbor_to_edge(record)
                key = (
                    edge.src_id,
                    edge.dst_id,
                    edge.relation,
                    edge.evidence.doc_id,
                    edge.evidence.element_id,
                )
                if key not in unique:
                    unique[key] = edge
        edges = sorted(
            unique.values(),
            key=lambda edge: (-edge.confidence, -edge.created_at.timestamp()),
        )
        return edges[:limit]

    def _synthesize_answer(
        self,
        question: str,
        mode: AnswerMode,
        paths: Sequence[PathModel],
        fallback_edges: Sequence[PathEdgeModel],
        resolved_entities: Sequence[ResolvedEntity],
    ) -> Optional[str]:
        prepared = self._prepare_synthesis_request(
            question, mode, paths, fallback_edges, resolved_entities
        )
        if prepared is None:
            return None
        request, synthesizer = prepared
        result = synthesizer.synthesize(request)
        if result is None:
            return None
        answer = result.answer.strip()
        return answer or None

    def _stream_llm_answer(
        self,
        question: str,
        mode: AnswerMode,
        paths: Sequence[PathModel],
        fallback_edges: Sequence[PathEdgeModel],
        resolved_entities: Sequence[ResolvedEntity],
    ) -> Optional[Iterator[QAStreamEvent]]:
        prepared = self._prepare_synthesis_request(
            question, mode, paths, fallback_edges, resolved_entities
        )
        if prepared is None:
            return None
        request, synthesizer = prepared

        def _generator() -> Iterator[QAStreamEvent]:
            stream_iter = synthesizer.stream(request)
            if stream_iter is None:
                result = synthesizer.synthesize(request)
                if result is None:
                    return None
                answer_text = result.answer.strip()
                return answer_text or None
            chunks: List[str] = []
            try:
                for chunk in stream_iter:
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    yield QAStreamEvent("llm_delta", chunk)
            except Exception:  # noqa: BLE001 - streaming failures should not break QA
                LOGGER.exception("QA LLM stream generator failed; falling back to None")
                return None
            final_answer = "".join(chunks).strip()
            return final_answer or None

        return _generator()

    def _summarize_entity_profiles(self, entities: Sequence[ResolvedEntity]) -> str:
        lines: List[str] = []
        for entity in entities:
            if not entity.candidates:
                continue
            top = max(entity.candidates, key=lambda candidate: candidate.similarity)
            sections = ", ".join(sorted(section for section in top.section_distribution.keys())) or "unknown sections"
            lines.append(
                f"{top.name} (similarity {top.similarity:.2f}, seen {top.times_seen}x, sections: {sections})"
            )
        if not lines:
            return "Insufficient evidence to answer the question."
        joined = "; ".join(lines)
        return f"Insufficient evidence from graph relations. Known entity profiles: {joined}."

    def _build_path_context(self, paths: Sequence[PathModel]) -> List[List[AnswerContextEdge]]:
        context: List[List[AnswerContextEdge]] = []
        for path in paths:
            edges = [self._edge_to_context(edge) for edge in path.edges]
            if edges:
                context.append(edges)
        return context

    def _edge_to_context(self, edge: PathEdgeModel) -> AnswerContextEdge:
        return AnswerContextEdge(
            subject=edge.src_name,
            predicate=edge.relation,
            obj=edge.dst_name,
            doc_id=edge.evidence.doc_id,
            element_id=edge.evidence.element_id,
            confidence=edge.confidence,
            relation_verbatim=edge.relation_verbatim,
            sentence=edge.evidence.full_sentence,
        )

    def _collect_scope_documents(
        self,
        path_context: Sequence[Sequence[AnswerContextEdge]],
        neighbor_context: Sequence[AnswerContextEdge],
        resolved_entities: Sequence[ResolvedEntity],
    ) -> List[str]:
        doc_ids = {
            edge.doc_id
            for path in path_context
            for edge in path
            if edge.doc_id
        }
        doc_ids.update(edge.doc_id for edge in neighbor_context if edge.doc_id)

        if not doc_ids:
            for entity in resolved_entities:
                for candidate in entity.candidates:
                    doc_id = getattr(candidate, "doc_id", None)
                    if doc_id:
                        doc_ids.add(str(doc_id))
        return sorted(doc_ids)

    def _summarize_neighbors(self, edges: Sequence[PathEdgeModel]) -> str:
        segments = [
            f"{edge.src_name} {edge.relation} {edge.dst_name} (doc {edge.evidence.doc_id}, conf={edge.confidence:.2f})"
            for edge in edges[: self._config.qa.max_results]
        ]
        if not segments:
            return "Insufficient evidence to answer the question."
        details = "; ".join(segments)
        return f"Insufficient evidence to answer. Related findings: {details}"

    def _unique_candidates(self, entities: Sequence[ResolvedEntity]) -> List[ResolvedCandidate]:
        unique: Dict[str, ResolvedCandidate] = {}
        for entity in entities:
            for candidate in entity.candidates:
                existing = unique.get(candidate.node_id)
                if existing is None or candidate.similarity > existing.similarity:
                    unique[candidate.node_id] = candidate
        return list(unique.values())

    def _select_primary_candidate(
        self, entities: Sequence[ResolvedEntity]
    ) -> Optional[ResolvedCandidate]:
        best: Optional[ResolvedCandidate] = None
        best_score = -1.0
        for entity in entities:
            for candidate in entity.candidates:
                if candidate.similarity > best_score:
                    best = candidate
                    best_score = candidate.similarity
        return best

    @staticmethod
    def _maybe_suppress_insufficient(
        answer: Optional[str],
        fallback_edges: Sequence[PathEdgeModel],
    ) -> Optional[str]:
        if not answer:
            return None
        if not fallback_edges:
            return answer
        normalized = answer.strip().lower()
        if normalized == "insufficient evidence to answer.":
            return None
        return answer

    def _serialize_payload(self, payload: object) -> object:
        if isinstance(payload, BaseModel):
            return payload.model_dump(mode="json")
        if isinstance(payload, IntentClassification):
            return {
                "intent": payload.intent.value,
                "document_ids": list(payload.document_ids),
            }
        if isinstance(payload, (list, tuple)):
            return [self._serialize_payload(item) for item in payload]
        if isinstance(payload, Mapping):
            return {str(key): self._serialize_payload(value) for key, value in payload.items()}
        return payload

    def _format_stream_event(self, event: QAStreamEvent) -> bytes:
        payload = self._serialize_payload(event.payload)
        body = json.dumps({"type": event.type, "payload": payload}, ensure_ascii=False)
        return f"event: {event.type}\ndata: {body}\n\n".encode("utf-8")

    def _build_path(self, record: PathRecord) -> PathModel:
        node_lookup = [self._normalize_node(node) for node in record.nodes]
        edges: List[PathEdgeModel] = []
        confidence_product = 1.0
        section_score = 1.0
        latest_ts = datetime.fromtimestamp(0, tz=timezone.utc)
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

    def _extract_evidence(self, payload: Mapping[str, object] | object) -> EvidenceModel:
        if isinstance(payload, str):
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError:
                LOGGER.warning("Failed to decode evidence payload from JSON in QA service")
                decoded = {}
            payload = decoded if isinstance(decoded, Mapping) else {}
        if not isinstance(payload, Mapping):
            if payload is not None:
                LOGGER.warning(
                    "Unexpected evidence payload type in QA service: %s", type(payload)
                )
            payload = {}
        raw_span = payload.get("text_span")
        if isinstance(raw_span, Mapping):
            start = int(raw_span.get("start", 0))
            end = int(raw_span.get("end", 0))
        else:
            start = int(payload.get("text_span_start", 0) or 0)
            end = int(payload.get("text_span_end", 0) or 0)
        return EvidenceModel(
            doc_id=str(payload.get("doc_id", "")),
            element_id=str(payload.get("element_id", "")),
            text_span={"start": start, "end": end},
            full_sentence=payload.get("full_sentence"),
        )

    @staticmethod
    def _extract_attributes(payload: Mapping[str, object] | object) -> Mapping[str, str]:
        if isinstance(payload, str):
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError:
                LOGGER.warning("Failed to decode attributes payload from JSON in QA service")
                return {}
            payload = decoded if isinstance(decoded, Mapping) else {}
        if not isinstance(payload, Mapping):
            if payload is not None:
                LOGGER.warning(
                    "Unexpected attributes payload type in QA service: %s", type(payload)
                )
            return {}
        return {str(key): str(value) for key, value in payload.items() if key}

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
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if hasattr(value, "to_native"):
        native = value.to_native()  # type: ignore[call-arg]
        if isinstance(native, datetime):
            return native if native.tzinfo else native.replace(tzinfo=timezone.utc)
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return datetime(
            int(getattr(value, "year")),
            int(getattr(value, "month")),
            int(getattr(value, "day")),
            int(getattr(value, "hour", 0)),
            int(getattr(value, "minute", 0)),
            int(getattr(value, "second", 0)),
            tzinfo=timezone.utc,
        )
    return datetime.fromtimestamp(0, tz=timezone.utc)
