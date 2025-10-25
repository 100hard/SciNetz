from __future__ import annotations

from backend.app.config import load_config
from backend.app.contracts import PaperMetadata, ParsedElement
from backend.app.extraction.domain import DomainRouter


def _element(content: str) -> ParsedElement:
    return ParsedElement(
        doc_id="doc-x",
        element_id="doc-x:0",
        section="Results",
        content=content,
        content_hash="a" * 64,
        start_char=0,
        end_char=len(content),
    )


def test_router_prefers_biology_domain_from_metadata() -> None:
    config = load_config()
    router = DomainRouter(config.extraction)

    metadata = PaperMetadata(
        doc_id="bio-42",
        title="Programmable RNA editing via CRISPR-Cas9",
        venue="Cell",
    )
    element = _element("Cas9 nuclease edits the BRCA1 locus in epithelial cells.")

    domain = router.resolve(metadata=metadata, element=element)

    assert domain.name == "biology"
    assert "Protein" in domain.entity_types
    assert domain.inventory_model == "en_core_sci_md"
    assert domain.fuzzy_match_threshold < config.extraction.fuzzy_match_threshold


def test_router_matches_physics_from_content_when_metadata_ambiguous() -> None:
    config = load_config()
    router = DomainRouter(config.extraction)

    metadata = PaperMetadata(doc_id="phys-1", title="Unknown study")
    content = "The interferometer observed gravitational waves during the experiment."
    element = _element(content)

    domain = router.resolve(metadata=metadata, element=element)

    assert domain.name == "physics"
    assert "Phenomenon" in domain.entity_types
    assert domain.inventory_model == "en_core_web_sm"


def test_router_falls_back_to_default_domain() -> None:
    config = load_config()
    router = DomainRouter(config.extraction)

    metadata = PaperMetadata(doc_id="ml-1", title="An unlabeled dataset study")
    element = _element("This chunk discusses evaluation metrics for models.")

    domain = router.resolve(metadata=metadata, element=element)

    assert domain.name == config.extraction.default_domain
    assert "Method" in domain.entity_types
