"""End-to-end orchestration of the SciNets extraction pipeline."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from backend.app.canonicalization import CanonicalizationPipeline
from backend.app.config import AppConfig
from backend.app.contracts import Evidence, PaperMetadata, ParsedElement, TextSpan, Triplet
from backend.app.extraction import EntityInventoryBuilder, ExtractionResult, TwoPassTripletExtractor
from backend.app.graph import GraphWriter
from backend.app.parsing import ParsingPipeline

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchestrationResult:
    """Summary payload returned after running the orchestration pipeline."""

    doc_id: str
    metadata: PaperMetadata
    processed_chunks: int
    skipped_chunks: int
    nodes_written: int
    edges_written: int
    co_mention_edges: int
    errors: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class _EdgeRecord:
    """Internal representation of an edge pending persistence."""

    triplet: Triplet
    relation_verbatim: str
    attributes: Optional[Dict[str, str]]
    times_seen: int = 1


@dataclass(frozen=True)
class _Sentence:
    """Representation of a sentence span inside a parsed element."""

    start: int
    end: int
    text: str


@dataclass(frozen=True)
class _Occurrence:
    """Occurrence of an entity mention inside a sentence."""

    name: str
    start: int
    end: int


@dataclass(frozen=True)
class _CoMentionOccurrence:
    """Concrete co-mention evidence linking two entities."""

    doc_id: str
    element_id: str
    section: str
    sentence_start: int
    sentence_end: int
    sentence_text: str


@dataclass(frozen=True)
class _CoMentionProduct:
    """Aggregate describing co-mention derived extraction outputs."""

    extraction: ExtractionResult
    edge: _EdgeRecord


class ProcessedChunkStore:
    """Persistent registry tracking processed content hashes for idempotence."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._records = self._load()

    def should_process(self, doc_id: str, content_hash: str, pipeline_version: str) -> bool:
        """Return whether the chunk requires processing for the current version."""

        stored_version = self._records.get(doc_id, {}).get(content_hash)
        return stored_version != pipeline_version

    def mark_processed(self, doc_id: str, content_hash: str, pipeline_version: str) -> None:
        """Record that a chunk has been processed for the supplied version."""

        doc_records = self._records.setdefault(doc_id, {})
        doc_records[content_hash] = pipeline_version

    def flush(self) -> None:
        """Persist the registry to disk."""

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as handle:
            json.dump(self._records, handle, indent=2, sort_keys=True)

    def _load(self) -> Dict[str, Dict[str, str]]:
        if not self._path.exists():
            return {}
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Failed to load processed chunk registry from %s", self._path)
            return {}
        records: Dict[str, Dict[str, str]] = {}
        for doc_id, mapping in payload.items():
            if not isinstance(mapping, dict):
                continue
            doc_key = str(doc_id)
            doc_records: Dict[str, str] = {}
            for content_hash, version in mapping.items():
                if not isinstance(content_hash, str) or not isinstance(version, str):
                    continue
                doc_records[content_hash] = version
            if doc_records:
                records[doc_key] = doc_records
        return records


class CoMentionAccumulator:
    """Aggregate co-mention edges for chunks where extraction failed."""

    _SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]?", re.MULTILINE)

    def __init__(self, config: AppConfig, inventory_builder: EntityInventoryBuilder) -> None:
        self._config = config
        self._inventory_builder = inventory_builder
        self._occurrences: Dict[Tuple[str, str], List[_CoMentionOccurrence]] = {}

    def record(
        self,
        element: ParsedElement,
        candidate_entities: Optional[Sequence[str]] = None,
    ) -> None:
        """Register a chunk for co-mention analysis."""

        entities = self._prepare_candidates(candidate_entities, element)
        if not entities:
            return
        chunk_pairs: Dict[Tuple[str, str], List[_CoMentionOccurrence]] = {}
        for sentence in self._iter_sentences(element.content):
            occurrences = self._find_occurrences(sentence, entities)
            if len(occurrences) < 2:
                continue
            for idx, first in enumerate(occurrences):
                for second in occurrences[idx + 1 :]:
                    if first.name == second.name:
                        continue
                    pair = tuple(sorted((first.name, second.name), key=str.lower))
                    if len(chunk_pairs) >= 10 and pair not in chunk_pairs:
                        continue
                    distance = abs(first.start - second.start)
                    if distance > self._config.co_mention.max_distance_chars:
                        continue
                    occurrence = _CoMentionOccurrence(
                        doc_id=element.doc_id,
                        element_id=element.element_id,
                        section=element.section,
                        sentence_start=sentence.start,
                        sentence_end=sentence.end,
                        sentence_text=sentence.text,
                    )
                    chunk_pairs.setdefault(pair, []).append(occurrence)
        for pair, occurrences in chunk_pairs.items():
            self._occurrences.setdefault(pair, []).extend(occurrences)

    def finalize(self) -> List[_CoMentionProduct]:
        """Return co-mention edges that satisfy frequency requirements."""

        min_occurrences = self._config.co_mention.min_occurrences
        confidence = self._config.co_mention.confidence
        pipeline_version = self._config.pipeline.version
        results: List[_CoMentionProduct] = []
        for pair, occurrences in sorted(self._occurrences.items()):
            if len(occurrences) < min_occurrences:
                continue
            first = occurrences[0]
            evidence = Evidence(
                element_id=first.element_id,
                doc_id=first.doc_id,
                text_span=TextSpan(start=first.sentence_start, end=first.sentence_end),
                full_sentence=first.sentence_text,
            )
            triplet = Triplet(
                subject=pair[0],
                predicate="correlates-with",
                object=pair[1],
                confidence=confidence,
                evidence=evidence,
                pipeline_version=pipeline_version,
            )
            section_distribution: Dict[str, Dict[str, int]] = {pair[0]: {}, pair[1]: {}}
            for occurrence in occurrences:
                section_distribution[pair[0]][occurrence.section] = (
                    section_distribution[pair[0]].get(occurrence.section, 0) + 1
                )
                section_distribution[pair[1]][occurrence.section] = (
                    section_distribution[pair[1]].get(occurrence.section, 0) + 1
                )
            extraction = ExtractionResult(
                triplets=[triplet],
                section_distribution=section_distribution,
                relation_verbatims=["co-mentioned"],
            )
            sections = sorted({occurrence.section for occurrence in occurrences})
            edge = _EdgeRecord(
                triplet=triplet,
                relation_verbatim="co-mentioned",
                attributes={
                    "method": "co-mention",
                    "hidden": "true",
                    "sections": ",".join(sections),
                },
                times_seen=len(occurrences),
            )
            results.append(_CoMentionProduct(extraction=extraction, edge=edge))
        return results

    def _prepare_candidates(
        self, candidate_entities: Optional[Sequence[str]], element: ParsedElement
    ) -> List[str]:
        if candidate_entities is not None:
            raw = list(candidate_entities)
        else:
            raw = self._inventory_builder.build_inventory(element)
        seen: set[str] = set()
        candidates: List[str] = []
        for value in raw:
            cleaned = value.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            candidates.append(cleaned)
        return candidates

    def _iter_sentences(self, text: str) -> Iterable[_Sentence]:
        matches = list(self._SENTENCE_PATTERN.finditer(text))
        if not matches:
            stripped = text.strip()
            if stripped:
                start = text.find(stripped)
                end = start + len(stripped)
                yield _Sentence(start=start, end=end, text=stripped)
            return
        for match in matches:
            raw = match.group()
            if not raw:
                continue
            start_offset = 0
            end_offset = len(raw)
            while start_offset < end_offset and raw[start_offset].isspace():
                start_offset += 1
            while end_offset > start_offset and raw[end_offset - 1].isspace():
                end_offset -= 1
            if start_offset >= end_offset:
                continue
            start = match.start() + start_offset
            end = match.start() + end_offset
            sentence_text = text[start:end]
            if sentence_text:
                yield _Sentence(start=start, end=end, text=sentence_text)

    def _find_occurrences(self, sentence: _Sentence, candidates: Sequence[str]) -> List[_Occurrence]:
        lowered_sentence = sentence.text.lower()
        occurrences: List[_Occurrence] = []
        for candidate in candidates:
            lowered_candidate = candidate.lower()
            start = 0
            while start < len(lowered_sentence):
                index = lowered_sentence.find(lowered_candidate, start)
                if index == -1:
                    break
                absolute_start = sentence.start + index
                absolute_end = absolute_start + len(candidate)
                occurrences.append(
                    _Occurrence(name=candidate, start=absolute_start, end=absolute_end)
                )
                start = index + len(candidate)
        occurrences.sort(key=lambda item: item.start)
        return occurrences


class ExtractionOrchestrator:
    """Coordinate the multi-phase extraction pipeline."""

    def __init__(
        self,
        *,
        config: AppConfig,
        parsing_pipeline: ParsingPipeline,
        inventory_builder: EntityInventoryBuilder,
        triplet_extractor: TwoPassTripletExtractor,
        canonicalization: CanonicalizationPipeline,
        graph_writer: GraphWriter,
        chunk_store: ProcessedChunkStore,
    ) -> None:
        self._config = config
        self._parsing = parsing_pipeline
        self._inventory_builder = inventory_builder
        self._triplet_extractor = triplet_extractor
        self._canonicalization = canonicalization
        self._graph_writer = graph_writer
        self._chunk_store = chunk_store

    def run(self, *, paper_id: str, pdf_path: Path) -> OrchestrationResult:
        """Execute the full extraction pipeline for a given paper."""

        parse_result = self._parsing.parse_document(doc_id=paper_id, pdf_path=pdf_path)
        if parse_result.errors:
            LOGGER.error("Parsing failed for %s: %s", paper_id, parse_result.errors)
            return OrchestrationResult(
                doc_id=paper_id,
                metadata=parse_result.metadata,
                processed_chunks=0,
                skipped_chunks=0,
                nodes_written=0,
                edges_written=0,
                co_mention_edges=0,
                errors=list(parse_result.errors),
            )

        extraction_results: List[ExtractionResult] = []
        edge_records: List[_EdgeRecord] = []
        errors: List[str] = []
        skipped_chunks = 0
        processed_chunks = 0
        co_mention_edges = 0
        pipeline_version = self._config.pipeline.version
        co_mention_accumulator: Optional[CoMentionAccumulator] = None
        if self._config.co_mention.enabled:
            co_mention_accumulator = CoMentionAccumulator(self._config, self._inventory_builder)

        for element in parse_result.elements:
            if not self._chunk_store.should_process(
                element.doc_id, element.content_hash, pipeline_version
            ):
                skipped_chunks += 1
                continue
            candidate_entities: Optional[List[str]] = None
            if self._config.extraction.use_entity_inventory:
                candidate_entities = self._inventory_builder.build_inventory(element)
            try:
                extraction = self._triplet_extractor.extract_with_metadata(
                    element, candidate_entities
                )
            except Exception as exc:  # noqa: BLE001 - third-party errors bubble up
                LOGGER.exception("Triplet extraction failed for %s", element.element_id)
                errors.append(str(exc))
                if co_mention_accumulator is not None:
                    fallback_candidates = candidate_entities
                    if fallback_candidates is None:
                        fallback_candidates = self._inventory_builder.build_inventory(element)
                    co_mention_accumulator.record(element, fallback_candidates)
                    processed_chunks += 1
                    self._chunk_store.mark_processed(
                        element.doc_id, element.content_hash, pipeline_version
                    )
                continue

            extraction_results.append(extraction)
            processed_chunks += 1
            self._chunk_store.mark_processed(element.doc_id, element.content_hash, pipeline_version)
            for idx, triplet in enumerate(extraction.triplets):
                relation_verbatim = triplet.predicate
                if idx < len(extraction.relation_verbatims):
                    candidate = extraction.relation_verbatims[idx]
                    if candidate:
                        relation_verbatim = candidate
                edge_records.append(
                    _EdgeRecord(
                        triplet=triplet,
                        relation_verbatim=relation_verbatim,
                        attributes={"method": "llm", "section": element.section},
                        times_seen=1,
                    )
                )

        if co_mention_accumulator is not None:
            products = co_mention_accumulator.finalize()
            co_mention_edges = len(products)
            for product in products:
                extraction_results.append(product.extraction)
                edge_records.append(product.edge)

        nodes_written = 0
        edges_written = 0
        try:
            canonical_result = self._canonicalization.run(extraction_results)
            alias_lookup = self._build_alias_lookup(
                canonical_result.nodes, canonical_result.merge_map
            )
            for node in canonical_result.nodes:
                self._graph_writer.upsert_entity(node)
                nodes_written += 1
            for record in edge_records:
                src_id = self._resolve_node(alias_lookup, record.triplet.subject)
                dst_id = self._resolve_node(alias_lookup, record.triplet.object)
                if not src_id or not dst_id:
                    LOGGER.warning(
                        "Skipping edge; missing canonical nodes for %s -> %s",
                        record.triplet.subject,
                        record.triplet.object,
                    )
                    continue
                relation_verbatim = record.relation_verbatim or record.triplet.predicate
                self._graph_writer.upsert_edge(
                    src_id=src_id,
                    dst_id=dst_id,
                    relation_norm=record.triplet.predicate,
                    relation_verbatim=relation_verbatim,
                    evidence=record.triplet.evidence,
                    confidence=record.triplet.confidence,
                    attributes=record.attributes,
                    times_seen=record.times_seen,
                )
                edges_written += 1
            self._graph_writer.flush()
        except Exception as exc:  # noqa: BLE001 - protective barrier around persistence
            LOGGER.exception("Failed to persist extraction results for %s", paper_id)
            errors.append(str(exc))

        try:
            self._chunk_store.flush()
        except OSError as exc:
            LOGGER.exception("Failed to persist processed chunk registry")
            errors.append(str(exc))

        return OrchestrationResult(
            doc_id=paper_id,
            metadata=parse_result.metadata,
            processed_chunks=processed_chunks,
            skipped_chunks=skipped_chunks,
            nodes_written=nodes_written,
            edges_written=edges_written,
            co_mention_edges=co_mention_edges,
            errors=errors,
        )

    @staticmethod
    def _build_alias_lookup(
        nodes: Sequence[object], merge_map: Mapping[str, str]
    ) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for node in nodes:
            name = getattr(node, "name", "")
            node_id = getattr(node, "node_id", "")
            if name and node_id:
                lookup.setdefault(name.lower(), node_id)
            for alias in getattr(node, "aliases", []) or []:
                lookup.setdefault(str(alias).lower(), node_id)
        for alias_key, node_id in merge_map.items():
            base_alias = alias_key.split("::", 1)[0].strip().lower()
            if base_alias and node_id:
                lookup.setdefault(base_alias, node_id)
        return lookup

    @staticmethod
    def _resolve_node(lookup: Mapping[str, str], name: str) -> Optional[str]:
        return lookup.get(name.lower())

