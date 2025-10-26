"""Unit tests for the query intent classifier heuristics."""
from __future__ import annotations

from backend.app.config import IntentRuleConfig, QAIntentConfig
from backend.app.qa.intent import QAIntent, QueryIntentClassifier


def _make_config() -> QAIntentConfig:
    return QAIntentConfig(
        enabled=True,
        max_summary_edges=8,
        entity_summary=IntentRuleConfig(
            keywords=["what is", "describe", "define", "explain"],
            min_score=0.25,
        ),
        cluster_summary=IntentRuleConfig(
            keywords=["summarize", "cluster"],
            min_score=0.5,
        ),
        paper_summary=IntentRuleConfig(
            keywords=["summarize", "paper", "overview"],
            min_score=0.4,
            document_prefixes=["paper", "doc"],
        ),
    )


def test_query_intent_classifier_extracts_document_ids() -> None:
    config = _make_config()
    classifier = QueryIntentClassifier(config)

    classification = classifier.classify("Please summarize the paper doc-omega and doc-alpha.")

    assert classification.intent == QAIntent.PAPER_SUMMARY
    assert classification.document_ids == ("doc-omega", "doc-alpha")


def test_query_intent_classifier_detects_entity_summary() -> None:
    config = _make_config()
    classifier = QueryIntentClassifier(config)

    classification = classifier.classify("What is CRISPR Cas9 and why is it important?")

    assert classification.intent == QAIntent.ENTITY_SUMMARY
    assert classification.document_ids == ()
