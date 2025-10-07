"""Entity canonicalization logic for Phase 4 of the pipeline."""
from __future__ import annotations

import json
import logging
import math
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set
from uuid import UUID, uuid5

import numpy as np

from ..config import AppConfig, load_config
from ..contracts import Node

LOGGER = logging.getLogger(__name__)

_POLYSEMY_BLOCKLIST = {
    "transformer",
    "regression",
    "model",
    "network",
    "system",
    "algorithm",
    "method",
    "approach",
    "framework",
    "attention",
}

_BLOCKLIST_THRESHOLD = 0.94
_NAMESPACE_UUID = UUID("2d6bb3b6-4be3-4f98-a18a-0a49de2a5d88")
_TOP_SECTION_OVERLAP_THRESHOLD = 0.5
_DISTRIBUTION_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class EntityCandidate:
    """Aggregated representation of an entity prior to canonicalization."""

    name: str
    type: str
    times_seen: int
    section_distribution: Mapping[str, int]
    source_document_ids: Sequence[str]


@dataclass(frozen=True, slots=True)
class CanonicalizationResult:
    """Output of canonicalization consisting of nodes and merge metadata."""

    nodes: Sequence[Node]
    merge_map: Mapping[str, str]
    merge_map_path: Path
    merge_report_path: Path


class EmbeddingBackend:
    """Protocol for embedding text into dense vectors."""

    def embed(self, text: str) -> np.ndarray:  # pragma: no cover - interface method
        raise NotImplementedError


class HashingEmbeddingBackend(EmbeddingBackend):
    """Deterministic embedding backend using hashing for reproducibility."""

    def __init__(self, dimensions: int = 16) -> None:
        if dimensions <= 0:
            msg = "Embedding dimensions must be positive"
            raise ValueError(msg)
        self._dimensions = dimensions

    def embed(self, text: str) -> np.ndarray:
        """Embed text into a deterministic vector.

        Args:
            text: Input string to embed.

        Returns:
            np.ndarray: Unit-normalized embedding vector.
        """

        normalized = unicodedata.normalize("NFKC", text)
        digest = sha256(normalized.encode("utf-8")).digest()
        hashed = np.frombuffer(digest, dtype=np.uint8)
        if hashed.size < self._dimensions:
            repeats = math.ceil(self._dimensions / hashed.size)
            hashed = np.tile(hashed, repeats)
        vector = hashed[: self._dimensions].astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return np.zeros(self._dimensions, dtype=np.float32)
        return vector / norm

def _normalize_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name)
    normalized = " ".join(normalized.split())
    normalized = normalized.strip("\t\n\r ")
    normalized = normalized.strip("!\"#$%&'()*+, -./:;<=>?@[\\]^_`{|}~")
    return normalized.lower()


def _top_sections(distribution: Mapping[str, int]) -> List[str]:
    filtered = [(section, count) for section, count in distribution.items() if count > 0]
    filtered.sort(key=lambda item: (-item[1], item[0]))
    return [section for section, _ in filtered[:2]]


def _section_overlap_ratio(a: Mapping[str, int], b: Mapping[str, int]) -> float:
    top_a = set(_top_sections(a))
    top_b = set(_top_sections(b))
    if not top_a or not top_b:
        return 0.0
    intersection = len(top_a & top_b)
    union = len(top_a | top_b)
    if union == 0:
        return 0.0
    return intersection / union


def _normalize_distribution(distribution: Mapping[str, int]) -> Dict[str, float]:
    total = sum(count for count in distribution.values() if count > 0)
    if total == 0:
        return {}
    return {section: count / total for section, count in distribution.items() if count > 0}


def _distributions_match(a: Mapping[str, int], b: Mapping[str, int]) -> bool:
    norm_a = _normalize_distribution(a)
    norm_b = _normalize_distribution(b)
    keys = set(norm_a) | set(norm_b)
    for key in keys:
        if abs(norm_a.get(key, 0.0) - norm_b.get(key, 0.0)) > _DISTRIBUTION_TOLERANCE:
            return False
    return True


class _Cluster:
    """Internal representation of a merged entity cluster."""

    def __init__(self, candidate: "_PreparedCandidate") -> None:
        self.members: List["_PreparedCandidate"] = [candidate]
        self.embeddings: List[np.ndarray] = [candidate.embedding]
        self.alias_counts: Dict[str, int] = {candidate.original.name: candidate.original.times_seen}
        self.section_distribution: Dict[str, int] = dict(candidate.original.section_distribution)
        self.source_documents: Set[str] = set(candidate.original.source_document_ids)
        self.type_counts: Dict[str, int] = {candidate.original.type: candidate.original.times_seen}
        self.normalized_names: Set[str] = {candidate.normalized_name}
        self.total_times_seen = candidate.original.times_seen
        self.blocklisted = candidate.blocklisted

    def similarity(self, embedding: np.ndarray) -> float:
        scores = [float(np.dot(embedding, member)) for member in self.embeddings]
        return max(scores, default=0.0)

    def add(self, candidate: "_PreparedCandidate") -> None:
        self.members.append(candidate)
        self.embeddings.append(candidate.embedding)
        self.total_times_seen += candidate.original.times_seen
        self.alias_counts[candidate.original.name] = (
            self.alias_counts.get(candidate.original.name, 0) + candidate.original.times_seen
        )
        self.type_counts[candidate.original.type] = (
            self.type_counts.get(candidate.original.type, 0) + candidate.original.times_seen
        )
        for section, count in candidate.original.section_distribution.items():
            self.section_distribution[section] = self.section_distribution.get(section, 0) + count
        self.source_documents.update(candidate.original.source_document_ids)
        self.normalized_names.add(candidate.normalized_name)
        self.blocklisted = self.blocklisted or candidate.blocklisted

    def is_polysemous(self, threshold: int) -> bool:
        return len([section for section, count in self.section_distribution.items() if count > 0]) >= threshold


@dataclass(slots=True)
class _PreparedCandidate:
    original: EntityCandidate
    normalized_name: str
    embedding: np.ndarray
    blocklisted: bool
    polysemous: bool


class EntityCanonicalizer:
    """Canonicalize entity candidates into stable graph nodes."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        *,
        embedding_backend: Optional[EmbeddingBackend] = None,
        embedding_dir: Optional[Path] = None,
        report_dir: Optional[Path] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._config = config or load_config()
        self._embedding_backend = embedding_backend or HashingEmbeddingBackend()
        root = Path(__file__).resolve().parents[2]
        self._embedding_dir = embedding_dir or (root / "data" / "embeddings")
        self._report_dir = report_dir or (root / "data" / "canonicalization")
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._embedding_dir.mkdir(parents=True, exist_ok=True)
        self._report_dir.mkdir(parents=True, exist_ok=True)

    def canonicalize(self, candidates: Sequence[EntityCandidate]) -> CanonicalizationResult:
        """Merge duplicate entity candidates into canonical nodes.

        Args:
            candidates: Entity candidates aggregated from extraction outputs.

        Returns:
            CanonicalizationResult: Canonical nodes and merge metadata artifacts.
        """

        prepared = [self._prepare_candidate(candidate) for candidate in candidates]
        prepared.sort(key=lambda item: (-item.original.times_seen, item.normalized_name))
        clusters: List[_Cluster] = []
        for candidate in prepared:
            assigned = False
            for cluster in clusters:
                if self._should_merge(candidate, cluster):
                    cluster.add(candidate)
                    assigned = True
                    break
            if not assigned:
                clusters.append(_Cluster(candidate))
        nodes = [self._cluster_to_node(cluster) for cluster in clusters]
        merge_map = self._build_merge_map(clusters)
        merge_map_path = self._write_merge_map(merge_map)
        report_path = self._write_merge_report(clusters)
        return CanonicalizationResult(
            nodes=nodes,
            merge_map=merge_map,
            merge_map_path=merge_map_path,
            merge_report_path=report_path,
        )

    def _prepare_candidate(self, candidate: EntityCandidate) -> _PreparedCandidate:
        """Normalize an entity candidate and attach embedding metadata.

        Args:
            candidate: Raw entity candidate to prepare.

        Returns:
            _PreparedCandidate: Candidate enriched with normalized name and embedding.
        """

        normalized = _normalize_name(candidate.name)
        embedding = self._load_or_create_embedding(normalized)
        blocklisted = normalized in _POLYSEMY_BLOCKLIST
        polysemous = self._is_polysemous_candidate(candidate)
        return _PreparedCandidate(
            original=candidate,
            normalized_name=normalized,
            embedding=embedding,
            blocklisted=blocklisted,
            polysemous=polysemous,
        )

    def _is_polysemous_candidate(self, candidate: EntityCandidate) -> bool:
        sections = [section for section, count in candidate.section_distribution.items() if count > 0]
        return len(sections) >= self._config.canonicalization.polysemy_section_diversity

    def _should_merge(self, candidate: _PreparedCandidate, cluster: _Cluster) -> bool:
        """Determine whether a candidate should merge into an existing cluster.

        Args:
            candidate: Candidate under consideration.
            cluster: Existing cluster to compare against.

        Returns:
            bool: ``True`` if the candidate should merge into the cluster.
        """

        threshold = self._config.canonicalization.base_threshold
        candidate_poly = candidate.polysemous
        cluster_poly = cluster.is_polysemous(self._config.canonicalization.polysemy_section_diversity)
        blocklisted = candidate.blocklisted or cluster.blocklisted
        if blocklisted:
            threshold = max(threshold, _BLOCKLIST_THRESHOLD)
        elif candidate_poly or cluster_poly:
            threshold = max(threshold, self._config.canonicalization.polysemy_threshold)
        similarity = cluster.similarity(candidate.embedding)
        if similarity < threshold:
            return False
        if blocklisted:
            if not _distributions_match(candidate.original.section_distribution, cluster.section_distribution):
                return False
        elif candidate_poly or cluster_poly:
            overlap = _section_overlap_ratio(candidate.original.section_distribution, cluster.section_distribution)
            if overlap < _TOP_SECTION_OVERLAP_THRESHOLD:
                return False
        return True

    def _cluster_to_node(self, cluster: _Cluster) -> Node:
        """Convert an internal cluster into a canonical node contract.

        Args:
            cluster: Cluster to convert.

        Returns:
            Node: Canonical node representing the cluster.
        """

        canonical_name = self._select_canonical_name(cluster.alias_counts)
        aliases = sorted({alias for alias in cluster.alias_counts if alias != canonical_name})
        node_type = self._select_node_type(cluster.type_counts)
        node_id = self._compute_node_id(cluster.normalized_names, cluster.section_distribution)
        return Node(
            node_id=node_id,
            name=canonical_name,
            type=node_type,
            aliases=aliases,
            section_distribution=dict(cluster.section_distribution),
            times_seen=cluster.total_times_seen,
            source_document_ids=sorted(cluster.source_documents),
        )

    def _select_canonical_name(self, alias_counts: Mapping[str, int]) -> str:
        return max(alias_counts.items(), key=lambda item: (item[1], item[0]))[0]

    def _select_node_type(self, type_counts: Mapping[str, int]) -> str:
        return max(type_counts.items(), key=lambda item: (item[1], item[0]))[0]

    def _compute_node_id(
        self,
        normalized_names: Iterable[str],
        section_distribution: Mapping[str, int],
    ) -> str:
        name_token = "|".join(sorted(normalized_names))
        normalized_distribution = _normalize_distribution(section_distribution)
        if normalized_distribution:
            section_parts = [
                f"{section}:{normalized_distribution[section]:.4f}"
                for section in sorted(normalized_distribution)
            ]
            token = "::".join([name_token, "|".join(section_parts)])
        else:
            token = name_token
        return str(uuid5(_NAMESPACE_UUID, token))

    def _build_merge_map(self, clusters: Sequence[_Cluster]) -> Dict[str, str]:
        """Construct the merge map linking aliases to canonical IDs.

        Args:
            clusters: Canonical clusters to summarise.

        Returns:
            Dict[str, str]: Mapping from alias keys to canonical node IDs.
        """

        alias_occurrences: Dict[str, int] = {}
        for cluster in clusters:
            for alias in cluster.alias_counts:
                alias_occurrences[alias] = alias_occurrences.get(alias, 0) + 1
        merge_map: Dict[str, str] = {}
        for cluster in clusters:
            node_id = self._compute_node_id(cluster.normalized_names, cluster.section_distribution)
            top_sections = _top_sections(cluster.section_distribution)
            section_suffix = "|".join(top_sections) if top_sections else "unknown"
            for alias in cluster.alias_counts:
                if alias_occurrences.get(alias, 0) <= 1:
                    key = alias
                else:
                    key = f"{alias}::{section_suffix}"
                merge_map[key] = node_id
        return dict(sorted(merge_map.items(), key=lambda item: item[0].lower()))

    def _write_merge_map(self, merge_map: Mapping[str, str]) -> Path:
        """Persist the merge map artifact to disk.

        Args:
            merge_map: Alias to canonical ID mapping.

        Returns:
            Path: Location of the written merge map file.
        """

        path = self._report_dir / "merge_map.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(merge_map, handle, indent=2, sort_keys=True)
        return path

    def _write_merge_report(self, clusters: Sequence[_Cluster]) -> Path:
        """Write the canonicalization audit report to disk.

        Args:
            clusters: Canonical clusters produced by the run.

        Returns:
            Path: File path of the written report.
        """

        timestamp = self._clock().strftime("%Y%m%dT%H%M%SZ")
        path = self._report_dir / f"merge_report_{timestamp}.json"
        report_entries = []
        sorted_clusters = sorted(
            clusters,
            key=lambda cluster: (
                -len(cluster.alias_counts),
                -cluster.total_times_seen,
                min(cluster.alias_counts),
            ),
        )
        for cluster in sorted_clusters[:20]:
            canonical_name = self._select_canonical_name(cluster.alias_counts)
            report_entries.append(
                {
                    "canonical_id": self._compute_node_id(
                        cluster.normalized_names, cluster.section_distribution
                    ),
                    "canonical_name": canonical_name,
                    "aliases": sorted(cluster.alias_counts.keys()),
                    "times_seen": cluster.total_times_seen,
                    "section_distribution": dict(cluster.section_distribution),
                }
            )
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report_entries, handle, indent=2, sort_keys=False)
        return path

    def _load_or_create_embedding(self, normalized_name: str) -> np.ndarray:
        """Load a cached embedding or create a new one if missing.

        Args:
            normalized_name: Normalized entity name used as cache key.

        Returns:
            np.ndarray: Unit-normalized embedding vector.
        """

        filename = f"{sha256(normalized_name.encode('utf-8')).hexdigest()}.npy"
        path = self._embedding_dir / filename
        if path.exists():
            try:
                vector = np.load(path)
                if isinstance(vector, np.ndarray):
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        return vector.astype(np.float32) / norm
            except Exception:  # pragma: no cover - defensive
                LOGGER.warning("Failed to load cached embedding for %s", normalized_name)
        embedding = self._embedding_backend.embed(normalized_name)
        np.save(path, embedding.astype(np.float32))
        return embedding

