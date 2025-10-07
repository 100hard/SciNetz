# Agent Instructions for SciNets Knowledge Graph

## Context
Read PLAN.md for full architecture. We're building a research paper knowledge graph extraction system with strict evidence provenance.

## Critical Constraints
- NEVER fabricate character offsets - use two-pass span linking (see Phase 3)
- ALWAYS include evidence field on edges (validation required)
- NO singularization of entity names (breaks technical terms)
- ALL config values must come from config.yaml (no hardcoded thresholds)
- Pipeline version must be on every edge/triplet

## Code Style
- Use pydantic models from contracts.py (frozen, never modify without versioning)
- Type hints required on all functions
- Docstrings: Google style with Args/Returns/Raises
- Error handling: fail-soft with logging, never silent failures
- Tests: unit tests for every validator, integration for pipelines

## Testing Philosophy
- Golden files for deterministic outputs (Phase 1, 3)
- Idempotence tests for all write operations (Phase 5, 6)
- Never mock Neo4j in integration tests (use testcontainers)

## When Implementing Phases
1. Read relevant section in PLAN.md first
2. Check for risk mitigations in Appendix A
3. Implement tests BEFORE implementation (TDD)
4. Run phase-specific tests from PLAN.md
5. Update metrics in observability dashboard

## Phase-Specific Notes

### Phase 3 (Extraction)
- Implement LLM adapter pattern (OpenAIExtractor base class for now)
- Two-pass MANDATORY: LLM returns text, linker finds offsets
- Fuzzy matching threshold: 0.90 (configurable in config.yaml)
- Log rejected triples with reasons (for debugging)

### Phase 4 (Canonicalization)
- Check polysemy_section_diversity BEFORE merging
- Never rewrite node IDs in Neo4j (merge_map only)
- Export merge report after every run (data/canonicalization/merge_report_{timestamp}.json)

### Phase 7 (QA)
- Entity resolution: try exact → alias → embedding (0.83 threshold)
- ALWAYS expand to 1-hop if zero paths found
- Return "Insufficient evidence" instead of empty response

## File Organization
- Contracts in backend/app/contracts.py (FROZEN - version if changing)
- Config in config.yaml (all thresholds here)
- Tests in tests/{phase_name}/ (mirror app structure)
- Golden files in tests/fixtures/golden/

## Common Pitfalls to Avoid
- Don't use LLM for exact offset extraction (will fail 30-40% of time)
- Don't merge entities without checking section_distribution
- Don't write edges without evidence field (will break UI)
- Don't skip content_hash check (wastes compute on unchanged chunks)
- Don't forget pipeline_version field (breaks reprocessing)

## When Stuck
1. Check PLAN.md Appendix C (Design Decisions) for rationale
2. Check PLAN.md Appendix A (Risk Mitigations) for known issues
3. Prefer simpler solution if ambiguous (ship > perfect)

## Current Phase
[Update this as you progress]
Phase: 4 - Canonicalization
Status: Ready to start
Blockers: None
