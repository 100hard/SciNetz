"""Intent detection utilities for graph-first QA queries."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence, Tuple

from backend.app.config import IntentRuleConfig, QAIntentConfig


class QAIntent(str, Enum):
    """Supported high-level intents for QA routing."""

    FACTOID = "factoid"
    ENTITY_SUMMARY = "entity_summary"
    CLUSTER_SUMMARY = "cluster_summary"
    PAPER_SUMMARY = "paper_summary"


@dataclass(frozen=True)
class IntentClassification:
    """Classification result returned by the intent detector."""

    intent: QAIntent
    document_ids: Tuple[str, ...] = ()


class QueryIntentClassifier:
    """Rule-based classifier that detects summary-style QA prompts."""

    _DOC_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-\.]{2,}")

    def __init__(self, config: QAIntentConfig) -> None:
        self._config = config
        self._enabled = bool(config.enabled)

    def classify(self, question: str) -> IntentClassification:
        """Classify the supplied question string.

        Args:
            question: Raw question text from the user.

        Returns:
            IntentClassification: Detected intent with optional metadata.
        """

        if not self._enabled:
            return IntentClassification(intent=QAIntent.FACTOID)
        cleaned = question.strip()
        if not cleaned:
            return IntentClassification(intent=QAIntent.FACTOID)

        lowered = cleaned.lower()

        paper_ids = self._detect_paper_summary(cleaned, lowered)
        if paper_ids:
            return IntentClassification(intent=QAIntent.PAPER_SUMMARY, document_ids=paper_ids)

        if self._matches_rule(lowered, self._config.cluster_summary):
            return IntentClassification(intent=QAIntent.CLUSTER_SUMMARY)

        if self._matches_rule(lowered, self._config.entity_summary):
            return IntentClassification(intent=QAIntent.ENTITY_SUMMARY)

        return IntentClassification(intent=QAIntent.FACTOID)

    def _detect_paper_summary(self, original: str, lowered: str) -> Tuple[str, ...]:
        rule = self._config.paper_summary
        if not self._matches_rule(lowered, rule):
            return ()
        document_ids = self._extract_document_ids(original, rule)
        if document_ids:
            return document_ids
        return ()

    def _matches_rule(self, lowered: str, rule: IntentRuleConfig) -> bool:
        if not rule.keywords:
            return False
        matches = sum(1 for keyword in rule.keywords if keyword in lowered)
        if matches == 0:
            return False
        denominator = max(len(rule.keywords), 1)
        score = matches / denominator
        return score >= rule.min_score

    def _extract_document_ids(self, text: str, rule: IntentRuleConfig) -> Tuple[str, ...]:
        if not rule.document_prefixes:
            return ()
        identifiers: List[str] = []
        for prefix in rule.document_prefixes:
            separated = re.compile(
                rf"{re.escape(prefix)}(?:[\s:=]+['\"]?)([A-Za-z0-9][A-Za-z0-9_\-\.]*)",
                re.IGNORECASE,
            )
            joined = re.compile(
                rf"({re.escape(prefix)}[-_][A-Za-z0-9][A-Za-z0-9_\-\.]*)",
                re.IGNORECASE,
            )
            for pattern in (separated, joined):
                for match in pattern.finditer(text):
                    candidate = match.group(1).strip(" '\".,;")
                    if candidate and candidate not in identifiers:
                        identifiers.append(candidate)
        if identifiers:
            return tuple(identifiers)

        tokens = self._DOC_TOKEN_PATTERN.findall(text)
        for token in tokens:
            normalized = token.strip()
            if not normalized:
                continue
            if any(normalized.lower().startswith(prefix) for prefix in rule.document_prefixes):
                cleaned = normalized.strip(" '\".,;")
                if cleaned and cleaned not in identifiers:
                    identifiers.append(cleaned)
        return tuple(identifiers)


IntentClassifierResult = IntentClassification
"""Backward-compatible alias for downstream imports."""
