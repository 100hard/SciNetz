from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

import numpy as np

from backend.app.config import load_config
from backend.app.qa.entity_resolution import (
    CandidateNode,
    EntityResolver,
    QARepositoryProtocol,
)


class SimpleEmbeddingBackend:
    """Deterministic embedding backend that captures basic string similarity."""

    def embed(self, text: str) -> np.ndarray:
        lowered = text.lower()
        features = np.array(
            [
                len(lowered),
                sum(1 for char in lowered if char in "aeiou"),
                sum(1 for char in lowered if char in "brtgp"),
                lowered.count("e"),
            ],
            dtype=np.float32,
        )
        norm = np.linalg.norm(features)
        if norm == 0:
            return features
        return features / norm


@dataclass
class _StubRepo(QARepositoryProtocol):
    nodes: Sequence[CandidateNode]

    def fetch_nodes_by_exact_match(self, mention: str) -> Sequence[CandidateNode]:
        lowered = mention.lower()
        results: List[CandidateNode] = []
        for node in self.nodes:
            if node.name.lower() == lowered:
                results.append(node)
                continue
            if any(alias.lower() == lowered for alias in node.aliases):
                results.append(node)
        return results

    def fetch_candidate_nodes(self, limit: int) -> Sequence[CandidateNode]:
        return list(self.nodes)[:limit]

    def fetch_paths(
        self,
        *,
        start_id: str,
        end_id: str,
        max_hops: int,
        min_confidence: float,
        limit: int,
        allowed_relations: Optional[Sequence[str]] = None,
    ) -> Sequence[Mapping[str, object]]:
        return []

    def fetch_neighbors(
        self,
        node_id: str,
        *,
        min_confidence: float,
        limit: int,
        allowed_relations: Optional[Sequence[str]] = None,
    ) -> Sequence[Mapping[str, object]]:
        return []


def test_entity_resolver_matches_typo() -> None:
    config = load_config()
    nodes = [
        CandidateNode(
            node_id="n1",
            name="BERT",
            aliases=["Bidirectional Encoder Representations"],
            times_seen=5,
            section_distribution={"Methods": 3},
        ),
        CandidateNode(
            node_id="n2",
            name="GPT-2",
            aliases=[],
            times_seen=2,
            section_distribution={"Results": 2},
        ),
    ]
    resolver = EntityResolver(
        repository=_StubRepo(nodes),
        embedding_backend=SimpleEmbeddingBackend(),
        similarity_threshold=config.qa.entity_match_threshold,
    )

    resolved = resolver.resolve("BURT")

    assert resolved.candidates
    assert resolved.candidates[0].name == "BERT"
    assert resolved.candidates[0].similarity >= config.qa.entity_match_threshold


def test_entity_resolver_resolves_alias() -> None:
    config = load_config()
    nodes = [
        CandidateNode(
            node_id="n1",
            name="Reinforcement Learning",
            aliases=["RL"],
            times_seen=10,
            section_distribution={"Results": 4},
        ),
    ]
    resolver = EntityResolver(
        repository=_StubRepo(nodes),
        embedding_backend=SimpleEmbeddingBackend(),
        similarity_threshold=config.qa.entity_match_threshold,
    )

    resolved = resolver.resolve("RL")

    assert resolved.candidates
    assert resolved.candidates[0].name == "Reinforcement Learning"
    assert resolved.candidates[0].similarity == 1.0
