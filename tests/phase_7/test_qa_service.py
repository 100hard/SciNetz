"""Unit tests for the graph-first QA service orchestration."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterator, List, Mapping, Optional, Sequence

from backend.app.canonicalization.entity_canonicalizer import HashingEmbeddingBackend
from backend.app.config import AppConfig, load_config
from backend.app.qa import AnswerMode, QAService
from backend.app.qa.answer_synthesis import AnswerSynthesisRequest, AnswerSynthesisResult
from backend.app.qa.entity_resolution import (
    CandidateNode,
    QARepositoryProtocol,
    ResolvedCandidate,
    ResolvedEntity,
)
from backend.app.qa.repository import NeighborRecord, PathRecord
from backend.app.qa.intent import IntentClassification, QAIntent


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
        document_neighbors: Optional[Dict[str, Sequence[NeighborRecord]]] = None,
    ) -> None:
        self._candidates = list(candidates)
        self._paths = paths or {}
        self._neighbors = neighbors or {}
        self._document_neighbors = document_neighbors or {}

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

    def fetch_candidates_for_mention(
        self,
        mention: str,
        limit: int,
        *,
        tokens: Sequence[str] | None = None,
    ) -> Sequence[CandidateNode]:
        del tokens
        mention_lower = mention.lower()
        matches = [
            node
            for node in self._candidates
            if mention_lower in node.name.lower()
            or any(mention_lower in alias.lower() for alias in node.aliases)
        ]
        if matches:
            return matches[:limit]
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

    def fetch_document_edges(
        self,
        doc_id: str,
        *,
        min_confidence: float,
        limit: int,
        allowed_relations: Optional[Sequence[str]] = None,
    ) -> Sequence[NeighborRecord]:
        del min_confidence, limit, allowed_relations
        return self._document_neighbors.get(doc_id, [])


class _StubSynthesizer:
    """Collects synthesis requests and returns a canned answer."""

    def __init__(
        self,
        answer: str = "Synthesized answer.",
        *,
        stream_chunks: Optional[Sequence[str]] = None,
    ) -> None:
        self.enabled = True
        self.answer = answer
        self.requests: List[object] = []
        self.stream_requests: List[object] = []
        self._stream_chunks = list(stream_chunks) if stream_chunks is not None else None

    def synthesize(self, request: object) -> AnswerSynthesisResult:
        self.requests.append(request)
        return AnswerSynthesisResult(answer=self.answer, raw_response={"mock": True})

    def stream(self, request: object) -> Optional[Iterator[str]]:
        self.stream_requests.append(request)
        if self._stream_chunks is None:
            return None

        def _generator() -> Iterator[str]:
            for chunk in self._stream_chunks:
                yield chunk

        return _generator()


class _StubIntentClassifier:
    """Returns a preconfigured classification for any question."""

    def __init__(self, classification: IntentClassification) -> None:
        self._classification = classification
        self.questions: List[str] = []

    def classify(self, question: str) -> IntentClassification:
        self.questions.append(question)
        return self._classification


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
    assert response.llm_answer is None


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
    assert response.llm_answer is None


def test_qa_service_returns_related_findings_when_no_paths() -> None:
    config = load_config()
    candidates = [_make_candidate("omega", "Model Omega")]
    neighbor = _neighbor_record("omega", "delta", "uses", "doc-omega")
    repository = _StubRepository(candidates, neighbors={"omega": [neighbor]})
    extractor = _StubExtractor(["Model Omega"])
    classifier = _StubIntentClassifier(IntentClassification(intent=QAIntent.FACTOID))
    qa_service = QAService(
        config=config,
        repository=repository,
        embedding_backend=HashingEmbeddingBackend(),
        extractor=extractor,  # type: ignore[arg-type]
        intent_classifier=classifier,  # type: ignore[arg-type]
    )

    response = qa_service.answer("What is known about Model Omega?")

    assert response.mode == AnswerMode.INSUFFICIENT
    assert response.fallback_edges
    assert "Related findings:" in response.summary
    assert "doc-omega" in response.summary
    assert "0.80" in response.summary
    assert response.llm_answer is None


def test_qa_service_surfaces_entity_profiles_when_no_neighbors() -> None:
    config = load_config()
    candidates = [_make_candidate("sigma", "Cell Sigma")]
    repository = _StubRepository(candidates, neighbors={})
    extractor = _StubExtractor(["Cell Sigma"])
    classifier = _StubIntentClassifier(IntentClassification(intent=QAIntent.FACTOID))
    qa_service = QAService(
        config=config,
        repository=repository,
        embedding_backend=HashingEmbeddingBackend(),
        extractor=extractor,  # type: ignore[arg-type]
        intent_classifier=classifier,  # type: ignore[arg-type]
    )

    response = qa_service.answer("What is Cell Sigma?")

    assert response.summary.startswith("Insufficient evidence from graph relations.")
    assert "Cell Sigma" in response.summary
    assert response.fallback_edges == []
    assert response.llm_answer is None


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
def test_qa_service_invokes_synthesizer_when_evidence_present() -> None:
    config = load_config()
    candidates = [_make_candidate("alpha", "Model Alpha"), _make_candidate("beta", "Model Beta")]
    repository = _StubRepository(candidates, paths={("alpha", "beta"): [_path_record(False)]})
    extractor = _StubExtractor(["Model Alpha", "Model Beta"])
    synthesizer = _StubSynthesizer(
        answer="Model Alpha outperforms Model Beta [doc-alpha-beta:alpha:0].",
    )
    qa_service = QAService(
        config=config,
        repository=repository,
        embedding_backend=HashingEmbeddingBackend(),
        extractor=extractor,  # type: ignore[arg-type]
        answer_synthesizer=synthesizer,  # type: ignore[arg-type]
    )

    response = qa_service.answer("Does Model Alpha outperform Model Beta?")

    assert response.mode == AnswerMode.DIRECT
    assert response.llm_answer == "Model Alpha outperforms Model Beta [doc-alpha-beta:alpha:0]."
    assert synthesizer.requests  # ensure synthesizer invoked


def test_qa_service_invokes_synthesizer_without_graph_evidence_when_allowed() -> None:
    config = load_config()
    candidates = [_make_candidate("alpha", "Model Alpha")]
    repository = _StubRepository(candidates, neighbors={})
    extractor = _StubExtractor(["Model Alpha"])
    synthesizer = _StubSynthesizer(answer="Unverified inference: model guess.")
    classifier = _StubIntentClassifier(IntentClassification(intent=QAIntent.ENTITY_SUMMARY))
    qa_service = QAService(
        config=config,
        repository=repository,
        embedding_backend=HashingEmbeddingBackend(),
        extractor=extractor,  # type: ignore[arg-type]
        answer_synthesizer=synthesizer,  # type: ignore[arg-type]
        intent_classifier=classifier,  # type: ignore[arg-type]
    )

    response = qa_service.answer("What is Model Alpha?")

    assert response.llm_answer == "Unverified inference: model guess."
    assert len(synthesizer.requests) == 1
    request = synthesizer.requests[0]
    assert isinstance(request, AnswerSynthesisRequest)
    assert request.has_graph_evidence is False
    assert request.allow_off_graph_answer is True


def test_qa_service_entity_summary_invokes_synthesizer() -> None:
    config = load_config()
    candidates = [_make_candidate("omega", "Model Omega")]
    neighbor = _neighbor_record("omega", "delta", "uses", "doc-omega")
    repository = _StubRepository(candidates, neighbors={"omega": [neighbor]})
    extractor = _StubExtractor(["Model Omega"])
    synthesizer = _StubSynthesizer(answer="Summary for Model Omega.")
    classifier = _StubIntentClassifier(IntentClassification(intent=QAIntent.ENTITY_SUMMARY))
    qa_service = QAService(
        config=config,
        repository=repository,
        embedding_backend=HashingEmbeddingBackend(),
        extractor=extractor,  # type: ignore[arg-type]
        answer_synthesizer=synthesizer,  # type: ignore[arg-type]
        intent_classifier=classifier,  # type: ignore[arg-type]
    )

    response = qa_service.answer("What is Model Omega?")

    assert response.mode == AnswerMode.DIRECT
    assert response.paths == []
    assert response.fallback_edges
    assert response.llm_answer == "Summary for Model Omega."
    assert "Model Omega" in response.summary


def test_qa_service_paper_summary_uses_document_edges() -> None:
    config = load_config()
    neighbor = _neighbor_record("alpha", "beta", "uses", "doc-omega")
    repository = _StubRepository([], document_neighbors={"doc-omega": [neighbor]})
    extractor = _StubExtractor([])
    synthesizer = _StubSynthesizer(answer="Document summary.")
    classifier = _StubIntentClassifier(
        IntentClassification(intent=QAIntent.PAPER_SUMMARY, document_ids=("doc-omega",))
    )
    qa_service = QAService(
        config=config,
        repository=repository,
        embedding_backend=HashingEmbeddingBackend(),
        extractor=extractor,  # type: ignore[arg-type]
        answer_synthesizer=synthesizer,  # type: ignore[arg-type]
        intent_classifier=classifier,  # type: ignore[arg-type]
    )

    response = qa_service.answer("Summarize the paper doc-omega.")

    assert response.mode == AnswerMode.DIRECT
    assert response.paths == []
    assert response.fallback_edges
    assert response.llm_answer == "Document summary."
    assert "doc-omega" in response.summary


def test_qa_service_entity_summary_includes_noncanonical_relations() -> None:
    config = load_config()
    candidates = [_make_candidate("theta", "Pathogen Theta")]
    custom_neighbor = _neighbor_record("theta", "sigma", "novel-interacts-with", "doc-theta")
    repository = _StubRepository(candidates, neighbors={"theta": [custom_neighbor]})
    extractor = _StubExtractor(["Pathogen Theta"])
    classifier = _StubIntentClassifier(IntentClassification(intent=QAIntent.ENTITY_SUMMARY))
    qa_service = QAService(
        config=config,
        repository=repository,
        embedding_backend=HashingEmbeddingBackend(),
        extractor=extractor,  # type: ignore[arg-type]
        intent_classifier=classifier,  # type: ignore[arg-type]
    )

    response = qa_service.answer("What is Pathogen Theta?")

    assert response.mode == AnswerMode.DIRECT
    assert response.fallback_edges
    relations = {edge.relation for edge in response.fallback_edges}
    assert "novel-interacts-with" in relations


def test_qa_service_suppresses_insufficient_llm_answer_when_evidence_exists() -> None:
    config = load_config()
    candidates = [_make_candidate("eta", "Protein Eta")]
    neighbor = _neighbor_record("eta", "zeta", "interacts-with", "doc-eta")
    repository = _StubRepository(candidates, neighbors={"eta": [neighbor]})
    extractor = _StubExtractor(["Protein Eta"])
    synthesizer = _StubSynthesizer(answer="Insufficient evidence to answer.")
    classifier = _StubIntentClassifier(IntentClassification(intent=QAIntent.ENTITY_SUMMARY))
    qa_service = QAService(
        config=config,
        repository=repository,
        embedding_backend=HashingEmbeddingBackend(),
        extractor=extractor,  # type: ignore[arg-type]
        answer_synthesizer=synthesizer,  # type: ignore[arg-type]
        intent_classifier=classifier,  # type: ignore[arg-type]
    )

    response = qa_service.answer("What is Protein Eta?")

    assert response.mode == AnswerMode.DIRECT
    assert response.fallback_edges
    assert response.llm_answer is None


def test_qa_service_stream_answer_emits_events() -> None:
    config = load_config()
    candidates = [_make_candidate("alpha", "Model Alpha"), _make_candidate("beta", "Model Beta")]
    repository = _StubRepository(candidates, paths={("alpha", "beta"): [_path_record(False)]})
    extractor = _StubExtractor(["Model Alpha", "Model Beta"])
    qa_service = QAService(
        config=config,
        repository=repository,
        embedding_backend=HashingEmbeddingBackend(),
        extractor=extractor,  # type: ignore[arg-type]
    )

    chunks = list(qa_service.stream_answer("Does Model Alpha outperform Model Beta?"))
    payload = "".join(chunk.decode("utf-8") for chunk in chunks)

    assert "event: classification" in payload
    assert "event: entities" in payload
    assert "event: paths" in payload
    assert "event: final" in payload
    assert '"mode": "direct"' in payload


def test_qa_service_stream_answer_emits_llm_delta_events_when_supported() -> None:
    config = load_config()
    candidates = [_make_candidate("alpha", "Model Alpha"), _make_candidate("beta", "Model Beta")]
    repository = _StubRepository(candidates, paths={("alpha", "beta"): [_path_record(False)]})
    extractor = _StubExtractor(["Model Alpha", "Model Beta"])
    synthesizer = _StubSynthesizer(stream_chunks=["Partial ", "answer"])
    qa_service = QAService(
        config=config,
        repository=repository,
        embedding_backend=HashingEmbeddingBackend(),
        extractor=extractor,  # type: ignore[arg-type]
        answer_synthesizer=synthesizer,  # type: ignore[arg-type]
    )

    chunks = list(qa_service.stream_answer("Does Model Alpha outperform Model Beta?"))
    payload = "".join(chunk.decode("utf-8") for chunk in chunks)

    assert "event: llm_delta" in payload
    assert "Partial " in payload
    assert "answer" in payload
    assert "event: llm_answer" in payload
    assert "Partial answer" in payload
    assert len(synthesizer.stream_requests) == 1
    assert synthesizer.requests == []
