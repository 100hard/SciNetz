"""Domain detection utilities for triplet extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from backend.app.config import (
    DomainMatchConfig,
    ExtractionConfig,
    ExtractionDomainConfig,
)
from backend.app.contracts import PaperMetadata, ParsedElement


@dataclass(frozen=True)
class ExtractionDomain:
    """Resolved domain context with extraction overrides."""

    name: str
    prompt_version: str
    entity_types: Sequence[str]
    fuzzy_match_threshold: float
    inventory_model: Optional[str]
    vocabulary: Sequence[str]


class DomainRouter:
    """Select the most appropriate extraction domain for a paper chunk."""

    def __init__(self, extraction_config: ExtractionConfig) -> None:
        self._config = extraction_config
        self._default_name = extraction_config.default_domain
        self._contexts: Dict[str, ExtractionDomain] = {}
        self._matchers: Dict[str, DomainMatchConfig] = {}
        for domain in extraction_config.domains:
            self._contexts[domain.name] = self._build_context(domain)
            self._matchers[domain.name] = domain.match
        if self._default_name not in self._contexts:
            fallback = ExtractionDomain(
                name=self._default_name,
                prompt_version=extraction_config.openai_prompt_version,
                entity_types=tuple(extraction_config.entity_types),
                fuzzy_match_threshold=extraction_config.fuzzy_match_threshold,
                inventory_model=None,
                vocabulary=tuple(),
            )
            self._contexts[self._default_name] = fallback
            self._matchers[self._default_name] = DomainMatchConfig()

    def resolve(
        self,
        *,
        metadata: Optional[PaperMetadata],
        element: Optional[ParsedElement],
    ) -> ExtractionDomain:
        """Resolve the best-matching extraction domain for the provided context."""

        best_name = self._default_name
        best_score = float("-inf")
        best_priority = self._matchers[best_name].priority
        for name, matcher in self._matchers.items():
            score = self._score(matcher, metadata, element)
            if score > best_score or (score == best_score and matcher.priority < best_priority):
                best_score = score
                best_priority = matcher.priority
                best_name = name
        return self._contexts[best_name]

    def context_for(self, name: str) -> ExtractionDomain:
        """Return the domain context for a specific domain name."""

        return self._contexts[name]

    def _build_context(self, domain: ExtractionDomainConfig) -> ExtractionDomain:
        threshold = (
            domain.fuzzy_match_threshold
            if domain.fuzzy_match_threshold is not None
            else self._config.fuzzy_match_threshold
        )
        prompt_version = domain.prompt_version or self._config.openai_prompt_version
        entity_types = domain.normalized_entity_types or tuple(self._config.entity_types)
        vocabulary = domain.normalized_vocabulary
        return ExtractionDomain(
            name=domain.name,
            prompt_version=prompt_version,
            entity_types=entity_types,
            fuzzy_match_threshold=threshold,
            inventory_model=domain.inventory_model,
            vocabulary=vocabulary,
        )

    @staticmethod
    def _score(
        matcher: DomainMatchConfig,
        metadata: Optional[PaperMetadata],
        element: Optional[ParsedElement],
    ) -> int:
        score = 0
        if metadata is not None:
            title = (metadata.title or "").lower()
            venue = (metadata.venue or "").lower()
            score += 3 * _count_matches(title, matcher.title_keywords)
            score += 4 * _count_matches(venue, matcher.venue_keywords)
        if element is not None:
            content = (element.content or "").lower()
            score += 2 * _count_matches(content, matcher.content_keywords)
        return score


def _count_matches(text: str, keywords: Sequence[str]) -> int:
    if not text:
        return 0
    return sum(1 for keyword in keywords if keyword in text)


__all__ = ["ExtractionDomain", "DomainRouter"]
