"""Unit tests for the Neo4j graph writer."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import pytest

from backend.app.config import AppConfig, load_config
from backend.app.contracts import Evidence, Node, TextSpan
from backend.app.graph import GraphWriter


@dataclass
class _InMemoryNode:
    node_id: str
    name: str
    type: str
    aliases: List[str]
    section_distribution: Dict[str, int]
    times_seen: int
    source_document_ids: List[str]
    pipeline_version: str


@dataclass
class _InMemoryEdge:
    src_id: str
    dst_id: str
    relation_norm: str
    relation_verbatim: str
    evidence: Dict[str, Any]
    confidence: float
    pipeline_version: str
    times_seen: int
    created_at: datetime
    attributes: Dict[str, str]
    conflicting: bool


@dataclass
class _Result:
    records: List[Dict[str, Any]]

    def __iter__(self):
        return iter(self.records)

    def single(self):
        return self.records[0] if self.records else None


class _InMemoryTransaction:
    """Minimal transaction stub replicating the write queries."""

    def __init__(self, store: "_InMemoryDriver") -> None:
        self._store = store

    def run(self, query: str, **params: Any) -> None:
        if "UNWIND $entities" in query:
            return self._apply_entities(params["entities"])
        elif "UNWIND $edges" in query:
            return self._apply_edges(params["edges"])
        elif "WHERE node.node_id IN $ids" in query:
            return self._load_nodes(params["ids"])
        elif "UNWIND $entries AS entry" in query and "MATCH (node:Entity)" in query:
            return self._lookup_aliases(params["entries"])
        elif "CALL db.labels" in query:
            return _Result(records=[{"present": 1}])
        else:  # pragma: no cover - defensive guard
            raise NotImplementedError(query)

    def _apply_entities(self, entities: Iterable[MutableMapping[str, Any]]) -> _Result:
        for payload in entities:
            node = self._store.nodes.get(payload["node_id"])
            if node is None:
                node = _InMemoryNode(
                    node_id=payload["node_id"],
                    name=payload["name"],
                    type=payload["type"],
                    aliases=list(payload["aliases"]),
                    section_distribution=dict(payload["section_distribution"]),
                    times_seen=payload["times_seen"],
                    source_document_ids=list(payload["source_document_ids"]),
                    pipeline_version=payload["pipeline_version"],
                )
            else:
                node.name = payload["name"]
                node.type = payload["type"]
                node.pipeline_version = payload["pipeline_version"]
                node.aliases = list(payload["aliases"])
                node.section_distribution = dict(payload["section_distribution"])
                node.times_seen = payload["times_seen"]
                node.source_document_ids = list(payload["source_document_ids"])
            self._store.nodes[payload["node_id"]] = node
        return _Result(records=[])

    def _apply_edges(self, edges: Iterable[MutableMapping[str, Any]]) -> _Result:
        records: List[Dict[str, Any]] = []
        for payload in edges:
            key = (payload["src_id"], payload["dst_id"], payload["relation_norm"])
            edge = self._store.edges.get(key)
            evidence = self._normalise_evidence(payload["evidence"])
            attributes = self._normalise_attributes(payload["attributes"])
            reverse_key = (payload["dst_id"], payload["src_id"], payload["relation_norm"])
            reverse_exists = bool(payload["directional"] and reverse_key in self._store.edges)
            if edge is None:
                edge = _InMemoryEdge(
                    src_id=payload["src_id"],
                    dst_id=payload["dst_id"],
                    relation_norm=payload["relation_norm"],
                    relation_verbatim=payload["relation_verbatim"],
                    evidence=evidence,
                    confidence=payload["confidence"],
                    pipeline_version=payload["pipeline_version"],
                    times_seen=payload["times_seen"],
                    created_at=payload["created_at"],
                    attributes=attributes,
                    conflicting=False,
                )
            else:
                edge.relation_verbatim = payload["relation_verbatim"]
                edge.pipeline_version = payload["pipeline_version"]
                if payload["attributes_provided"]:
                    edge.attributes = attributes
                edge.evidence = evidence
                edge.times_seen += payload["times_seen"]
                edge.confidence = max(edge.confidence, payload["confidence"])
            self._store.edges[key] = edge

            if reverse_exists:
                self._store.edges[key].conflicting = True
                self._store.edges[reverse_key].conflicting = True

            records.append(
                {
                    "src_id": payload["src_id"],
                    "dst_id": payload["dst_id"],
                    "relation_norm": payload["relation_norm"],
                    "directional": payload["directional"],
                    "has_reverse": reverse_exists,
                    "evidence": payload["evidence"],
                    "reverse_evidence": (
                        json.dumps(self._store.edges[reverse_key].evidence, sort_keys=True)
                        if reverse_exists
                        else None
                    ),
                }
            )
        return _Result(records=records)

    def _lookup_aliases(
        self, entries: Iterable[Mapping[str, str]]
    ) -> _Result:
        records: List[Dict[str, Any]] = []
        for entry in entries:
            alias = entry.get("alias", "")
            requested = entry.get("node_id", "")
            if not alias or not requested:
                continue
            alias_lower = str(alias).strip().lower()
            for node in self._store.nodes.values():
                candidate_aliases = {node.name.lower(), *(alias.lower() for alias in node.aliases)}
                if alias_lower not in candidate_aliases:
                    continue
                records.append(
                    {
                        "alias": alias_lower,
                        "requested_id": requested,
                        "node_id": node.node_id,
                        "name": node.name,
                        "aliases": list(node.aliases),
                        "docs": list(node.source_document_ids),
                    }
                )
        return _Result(records=records)

    def _load_nodes(self, ids: Iterable[str]) -> _Result:
        records: List[Dict[str, Any]] = []
        for node_id in ids:
            node = self._store.nodes.get(node_id)
            if node is None:
                continue
            sections = dict(node.section_distribution)
            records.append(
                {
                    "node_id": node.node_id,
                    "aliases": list(node.aliases),
                    "docs": list(node.source_document_ids),
                    "legacy_sections": None,
                    "section_keys": list(sections.keys()),
                    "section_values": list(sections.values()),
                    "times_seen": node.times_seen,
                }
            )
        return _Result(records=records)

    @staticmethod
    def _normalise_evidence(payload: Any) -> Dict[str, Any]:
        """Convert serialized evidence payloads into nested dictionaries.

        Args:
            payload: Evidence payload emitted by the graph writer.

        Returns:
            Dict[str, Any]: Evidence dictionary containing a nested text span map.
        """

        text_span: Dict[str, int]
        if isinstance(payload, str):
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError:
                decoded = {}
            payload = decoded if isinstance(decoded, Mapping) else {}
        if not isinstance(payload, Mapping):
            payload = {}
        raw_span = payload.get("text_span")
        if isinstance(raw_span, Mapping):
            start = int(raw_span.get("start", 0) or 0)
            end = int(raw_span.get("end", 0) or 0)
        else:
            start = int(payload.get("text_span_start", 0) or 0)
            end = int(payload.get("text_span_end", 0) or 0)
        text_span = {"start": start, "end": end}

        result: Dict[str, Any] = {
            "doc_id": payload.get("doc_id"),
            "element_id": payload.get("element_id"),
            "text_span": text_span,
        }
        full_sentence = payload.get("full_sentence")
        if full_sentence is not None:
            result["full_sentence"] = full_sentence
        return result

    @staticmethod
    def _normalise_attributes(payload: Any) -> Dict[str, str]:
        if isinstance(payload, str):
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError:
                decoded = {}
            payload = decoded if isinstance(decoded, Mapping) else {}
        if not isinstance(payload, Mapping):
            return {}
        return {str(key): str(value) for key, value in payload.items() if key}


class _InMemorySession:
    """Context manager returning transactions for the in-memory driver."""

    def __init__(self, store: "_InMemoryDriver") -> None:
        self._store = store

    def __enter__(self) -> "_InMemorySession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - interface contract
        return None

    def execute_write(self, func: Callable[..., Any], payload: Iterable[MutableMapping[str, Any]]) -> None:
        tx = _InMemoryTransaction(self._store)
        func(tx, payload)

    def execute_read(self, func: Callable[..., Any], *args: Any) -> Any:
        tx = _InMemoryTransaction(self._store)
        return func(tx, *args)


class _InMemoryDriver:
    """Lightweight replacement for the Neo4j driver used in unit tests."""

    def __init__(self) -> None:
        self.nodes: Dict[str, _InMemoryNode] = {}
        self.edges: Dict[Tuple[str, str, str], _InMemoryEdge] = {}

    def session(self) -> _InMemorySession:
        return _InMemorySession(self)


@pytest.fixture(name="config")
def fixture_config() -> AppConfig:
    return load_config()


@pytest.fixture(name="driver")
def fixture_driver() -> _InMemoryDriver:
    return _InMemoryDriver()


@pytest.fixture(name="writer")
def fixture_writer(driver: _InMemoryDriver, config: AppConfig) -> GraphWriter:
    return GraphWriter(driver=driver, config=config, entity_batch_size=50, edge_batch_size=50)


def test_entity_upsert_is_idempotent(writer: GraphWriter, driver: _InMemoryDriver) -> None:
    node = Node(
        node_id="node-1",
        name="Reinforcement Learning",
        type="Method",
        aliases=["RL"],
        section_distribution={"Methods": 3, "Results": 1},
        times_seen=4,
        source_document_ids=["doc1", "doc2"],
    )

    writer.upsert_entity(node)
    writer.upsert_entity(node)
    writer.flush()

    assert len(driver.nodes) == 1
    stored = driver.nodes[node.node_id]
    assert stored.times_seen == 4
    assert stored.aliases == ["RL"]
    assert stored.section_distribution == {"Methods": 3, "Results": 1}
    assert sorted(stored.source_document_ids) == ["doc1", "doc2"]


def test_entity_upsert_increments_with_new_documents(
    writer: GraphWriter, driver: _InMemoryDriver
) -> None:
    node = Node(
        node_id="node-2",
        name="Graph Neural Network",
        type="Model",
        aliases=["GNN"],
        section_distribution={"Methods": 1},
        times_seen=1,
        source_document_ids=["doc-1"],
    )
    writer.upsert_entity(node)
    writer.flush()

    updated = Node(
        node_id="node-2",
        name="Graph Neural Network",
        type="Model",
        aliases=["GNN"],
        section_distribution={"Methods": 2},
        times_seen=2,
        source_document_ids=["doc-2"],
    )
    writer.upsert_entity(updated)
    writer.flush()

    stored = driver.nodes[node.node_id]
    assert stored.times_seen == 3
    assert stored.section_distribution == {"Methods": 3}
    assert sorted(stored.source_document_ids) == ["doc-1", "doc-2"]


def test_upsert_edge_requires_evidence(writer: GraphWriter, driver: _InMemoryDriver) -> None:
    span = TextSpan(start=0, end=10)
    evidence = Evidence(element_id="el-1", text_span=span, doc_id="doc-1")

    writer.upsert_entity(
        Node(
            node_id="node-a",
            name="Model A",
            type="Model",
            aliases=[],
            section_distribution={"Intro": 1},
            times_seen=1,
            source_document_ids=["doc-1"],
        )
    )
    writer.upsert_entity(
        Node(
            node_id="node-b",
            name="Model B",
            type="Model",
            aliases=[],
            section_distribution={"Intro": 1},
            times_seen=1,
            source_document_ids=["doc-1"],
        )
    )
    writer.flush()

    with pytest.raises(ValueError):
        writer.upsert_edge(
            src_id="node-a",
            dst_id="node-b",
            relation_norm="compared-to",
            relation_verbatim="compared to",
            evidence=None,  # type: ignore[arg-type]
            confidence=0.8,
        )

    writer.upsert_edge(
        src_id="node-a",
        dst_id="node-b",
        relation_norm="compared-to",
        relation_verbatim="compared to",
        evidence=evidence,
        confidence=0.8,
    )
    writer.flush()

    key = ("node-a", "node-b", "compared-to")
    assert key in driver.edges


def test_edge_evidence_serialization(writer: GraphWriter, driver: _InMemoryDriver) -> None:
    node_src = Node(
        node_id="node-src",
        name="Source",
        type="Concept",
        aliases=[],
        section_distribution={"Intro": 1},
        times_seen=1,
        source_document_ids=["doc-10"],
    )
    node_dst = Node(
        node_id="node-dst",
        name="Destination",
        type="Concept",
        aliases=[],
        section_distribution={"Intro": 1},
        times_seen=1,
        source_document_ids=["doc-10"],
    )
    writer.upsert_entity(node_src)
    writer.upsert_entity(node_dst)
    writer.flush()

    evidence = Evidence(
        element_id="el-1",
        text_span=TextSpan(start=5, end=25),
        doc_id="doc-10",
        full_sentence="Source relates to destination.",
    )
    writer.upsert_edge(
        src_id=node_src.node_id,
        dst_id=node_dst.node_id,
        relation_norm="relates-to",
        relation_verbatim="relates to",
        evidence=evidence,
        confidence=0.6,
    )
    writer.flush()

    key = (node_src.node_id, node_dst.node_id, "relates-to")
    stored = driver.edges[key]
    assert stored.evidence["doc_id"] == evidence.doc_id
    assert stored.evidence["element_id"] == evidence.element_id
    assert stored.evidence["text_span"]["start"] == evidence.text_span.start
    assert stored.evidence["text_span"]["end"] == evidence.text_span.end
    assert stored.evidence["full_sentence"] == evidence.full_sentence


def test_edge_attributes_are_serialized_to_json(writer: GraphWriter) -> None:
    evidence = Evidence(
        element_id="el-attr",
        text_span=TextSpan(start=1, end=2),
        doc_id="doc-attr",
    )
    payload = writer._edge_to_parameters(
        src_id="src",
        dst_id="dst",
        relation_norm="uses",
        relation_verbatim="uses",
        evidence=evidence,
        confidence=0.5,
        attributes={"section": "Results", "method": "llm"},
        created_at=datetime.now(timezone.utc),
        times_seen=1,
    )
    assert isinstance(payload["attributes"], str)
    decoded = json.loads(payload["attributes"])
    assert decoded == {"method": "llm", "section": "Results"}


def test_edge_upsert_counts_and_conflicts(
    writer: GraphWriter,
    driver: _InMemoryDriver,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="backend.app.graph.writer")
    node_a = Node(
        node_id="node-a",
        name="Model A",
        type="Model",
        aliases=["A"],
        section_distribution={"Methods": 2},
        times_seen=2,
        source_document_ids=["doc-1"],
    )
    node_b = Node(
        node_id="node-b",
        name="Model B",
        type="Model",
        aliases=["B"],
        section_distribution={"Methods": 1},
        times_seen=1,
        source_document_ids=["doc-2"],
    )
    writer.upsert_entity(node_a)
    writer.upsert_entity(node_b)
    writer.flush()

    evidence_one = Evidence(
        element_id="el-1",
        text_span=TextSpan(start=0, end=5),
        doc_id="doc-1",
        full_sentence="Model A outperforms Model B.",
    )
    writer.upsert_edge(
        src_id=node_a.node_id,
        dst_id=node_b.node_id,
        relation_norm="outperforms",
        relation_verbatim="outperforms",
        evidence=evidence_one,
        confidence=0.9,
        attributes={"metric": "accuracy"},
        times_seen=2,
    )
    writer.upsert_edge(
        src_id=node_a.node_id,
        dst_id=node_b.node_id,
        relation_norm="outperforms",
        relation_verbatim="clearly outperforms",
        evidence=evidence_one,
        confidence=0.7,
    )
    writer.flush()

    forward_key = (node_a.node_id, node_b.node_id, "outperforms")
    forward_edge = driver.edges[forward_key]
    assert forward_edge.times_seen == 3
    assert forward_edge.confidence == pytest.approx(0.9)
    assert forward_edge.conflicting is False
    assert forward_edge.attributes == {"metric": "accuracy"}

    evidence_two = Evidence(
        element_id="el-2",
        text_span=TextSpan(start=5, end=10),
        doc_id="doc-2",
        full_sentence="Model B beats Model A.",
    )
    writer.upsert_edge(
        src_id=node_b.node_id,
        dst_id=node_a.node_id,
        relation_norm="outperforms",
        relation_verbatim="beats",
        evidence=evidence_two,
        confidence=0.6,
    )
    writer.flush()

    reverse_key = (node_b.node_id, node_a.node_id, "outperforms")
    assert driver.edges[forward_key].conflicting is True
    assert driver.edges[reverse_key].conflicting is True

    conflict_logs = [
        record.message
        for record in caplog.records
        if "conflicts with an existing reverse edge" in record.message
    ]
    assert conflict_logs
    latest = conflict_logs[-1]
    assert "new_doc_id=doc-2" in latest
    assert "new_element_id=el-2" in latest
    assert "existing_doc_id=doc-1" in latest
    assert "existing_element_id=el-1" in latest


def test_bidirectional_relations_do_not_conflict(
    writer: GraphWriter, driver: _InMemoryDriver
) -> None:
    node_a = Node(
        node_id="node-x",
        name="Dataset X",
        type="Dataset",
        aliases=["X"],
        section_distribution={"Results": 1},
        times_seen=1,
        source_document_ids=["doc-10"],
    )
    node_b = Node(
        node_id="node-y",
        name="Dataset Y",
        type="Dataset",
        aliases=["Y"],
        section_distribution={"Results": 1},
        times_seen=1,
        source_document_ids=["doc-11"],
    )
    writer.upsert_entity(node_a)
    writer.upsert_entity(node_b)
    writer.flush()

    evidence = Evidence(
        element_id="el-compare",
        text_span=TextSpan(start=0, end=5),
        doc_id="doc-compare",
        full_sentence="Dataset X is compared to Dataset Y.",
    )
    writer.upsert_edge(
        src_id=node_a.node_id,
        dst_id=node_b.node_id,
        relation_norm="compared-to",
        relation_verbatim="compared to",
        evidence=evidence,
        confidence=0.6,
    )
    writer.upsert_edge(
        src_id=node_b.node_id,
        dst_id=node_a.node_id,
        relation_norm="compared-to",
        relation_verbatim="compared to",
        evidence=evidence,
        confidence=0.6,
    )
    writer.flush()

    forward_key = (node_a.node_id, node_b.node_id, "compared-to")
    reverse_key = (node_b.node_id, node_a.node_id, "compared-to")
    assert driver.edges[forward_key].conflicting is False
    assert driver.edges[reverse_key].conflicting is False


def test_get_documents_by_aliases_consumes_results_before_session_close(
    config: AppConfig,
) -> None:
    alias_map = {"Transformer": "node-42"}

    class _AliasReadResult:
        def __init__(self, session: "_AliasReadSession", rows: List[Dict[str, Any]]) -> None:
            self._session = session
            self._rows = rows

        def __iter__(self) -> Iterable[Dict[str, Any]]:
            if getattr(self._session, "closed", False):
                raise RuntimeError("result out of scope")
            return iter(self._rows)

    class _AliasReadTransaction:
        def __init__(self, session: "_AliasReadSession") -> None:
            self._session = session

        def run(
            self,
            query: str,
            entries: Iterable[Mapping[str, str]],
        ) -> _AliasReadResult:
            rows = []
            for entry in entries:
                node_id = entry["node_id"]
                if node_id == "node-42":
                    rows.append(
                        {
                            "requested_id": node_id,
                            "alias": entry["alias"],
                            "node_id": "existing-42",
                            "name": "Transformer",
                            "aliases": ["Transformer"],
                            "docs": ["doc-a", "doc-b", "doc-a"],
                        }
                    )
            return _AliasReadResult(self._session, rows)

    class _AliasReadSession:
        def __init__(self) -> None:
            self.closed = False

        def __enter__(self) -> "_AliasReadSession":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.closed = True

        def execute_read(self, callback: Callable[[Any], Any]) -> Any:
            transaction = _AliasReadTransaction(self)
            return callback(transaction)

    class _AliasReadDriver:
        def session(self) -> _AliasReadSession:
            return _AliasReadSession()

    writer = GraphWriter(
        driver=_AliasReadDriver(),
        config=config,
        entity_batch_size=10,
        edge_batch_size=10,
    )

    documents = writer.get_documents_by_aliases(alias_map)
    assert documents == {"node-42": frozenset({"doc-a", "doc-b"})}


def test_resolve_aliases_returns_existing_metadata(
    writer: GraphWriter,
    driver: _InMemoryDriver,
) -> None:
    existing = Node(
        node_id="node-existing",
        name="Transformer",
        type="Model",
        aliases=["Attention Model"],
        section_distribution={"Methods": 2},
        times_seen=3,
        source_document_ids=["doc-old"],
    )
    writer.upsert_entity(existing)
    writer.flush()

    matches = writer.resolve_aliases({"transformer": "node-candidate"})
    assert "node-candidate" in matches
    match = matches["node-candidate"]
    assert match.existing_node_id == "node-existing"
    assert match.name == "Transformer"
    assert match.aliases == frozenset({"Attention Model", "transformer"})
    assert match.docs == frozenset({"doc-old"})

    documents = writer.get_documents_by_aliases({"transformer": "node-candidate"})
    assert documents == {"node-candidate": frozenset({"doc-old"})}
