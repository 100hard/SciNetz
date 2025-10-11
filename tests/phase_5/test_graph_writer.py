"""Unit tests for the Neo4j graph writer."""
from __future__ import annotations

import json
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


class _InMemoryTransaction:
    """Minimal transaction stub replicating the write queries."""

    def __init__(self, store: "_InMemoryDriver") -> None:
        self._store = store

    def run(self, query: str, **params: Any) -> None:
        if "UNWIND $entities" in query:
            self._apply_entities(params["entities"])
        elif "UNWIND $edges" in query:
            self._apply_edges(params["edges"])
        else:  # pragma: no cover - defensive guard
            raise NotImplementedError(query)

    def _apply_entities(self, entities: Iterable[MutableMapping[str, Any]]) -> None:
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

    def _apply_edges(self, edges: Iterable[MutableMapping[str, Any]]) -> None:
        for payload in edges:
            key = (payload["src_id"], payload["dst_id"], payload["relation_norm"])
            edge = self._store.edges.get(key)
            evidence = self._normalise_evidence(payload["evidence"])
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
                    attributes=dict(payload["attributes"]),
                    conflicting=False,
                )
            else:
                edge.relation_verbatim = payload["relation_verbatim"]
                edge.pipeline_version = payload["pipeline_version"]
                if payload["attributes_provided"]:
                    edge.attributes = dict(payload["attributes"])
                edge.evidence = evidence
                edge.times_seen += payload["times_seen"]
                edge.confidence = max(edge.confidence, payload["confidence"])
            self._store.edges[key] = edge

            reverse_key = (payload["dst_id"], payload["src_id"], payload["relation_norm"])
            if payload["directional"] and reverse_key in self._store.edges:
                self._store.edges[key].conflicting = True
                self._store.edges[reverse_key].conflicting = True

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


def test_edge_upsert_counts_and_conflicts(writer: GraphWriter, driver: _InMemoryDriver) -> None:
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
