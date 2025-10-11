# Plan: Eliminate Non-Primitive Neo4j Property Writes

## Context
Recent ingestion runs still fail with `Neo.ClientError.Statement.TypeError` because
our relationship payload includes nested maps (e.g. `{method: "llm", section: "Abstract"}`),
which Neo4j rejects as property values. Earlier changes flattened evidence spans into
JSON strings, but relationship attributes remain structured maps and continue to
cause write retries and data loss. We also receive `UnknownPropertyKeyWarning` for
`section_distribution` due to legacy reads referencing a property that no longer
exists on stored nodes.

## Goals
1. Guarantee every property written to Neo4j is a primitive or list of primitives.
2. Preserve rich attribute metadata by serialising/deserialising without breaking
   existing API responses.
3. Remove the noisy `section_distribution` warning while maintaining backwards
   compatibility for historical data.
4. Extend regression coverage so future schema tweaks cannot reintroduce invalid
   property types.

## Milestones

### 1. Audit and Data Contract Updates
- [ ] Trace all `RELATION` writes (`GraphWriter._edge_to_parameters`) and confirm
      the exact shape of `attributes` reaching Neo4j.
- [ ] Define a serialisation contract for edge attributes (likely JSON string with
      predictable schema version + optional compression if needed).
- [ ] Update evidence/attribute pydantic contracts if additional metadata fields
      have emerged since the last schema revision.

### 2. Writer Serialisation Changes
- [ ] Implement attribute serialisation alongside the existing evidence JSON helper.
- [ ] Ensure `attributes_provided` still drives merge semantics after switching to
      the serialised form.
- [ ] Introduce utility to emit deterministic JSON (sorted keys) for diff-friendly
      comparisons and cache hits.

### 3. Reader Deserialisation & API Layer Safeguards
- [ ] Update QA/UI repositories to decode the new attribute payload, logging
      fallbacks when deserialisation fails.
- [ ] Backfill decoding within any batch export or analytics consumers.
- [ ] Provide safe defaults when attributes are absent or malformed so responses
      remain stable.

### 4. Section Distribution Query Cleanup
- [ ] Replace direct `node.section_distribution` access with reconstruction from
      the key/value arrays to remove the Neo4j warning.
- [ ] Add migration note or cleanup job for any legacy nodes still carrying the
      old property so the read path is fully deterministic.

### 5. Testing & Verification
- [ ] Expand Phase 5 writer tests to assert that persisted edge payloads are JSON
      strings and decode to the expected structures.
- [ ] Add end-to-end QA/UI tests covering responses that include attribute payloads.
- [ ] Run targeted ingestion against a Neo4j test container to verify no warnings
      or type errors are logged.

## Open Questions
- Do any downstream analytics scripts rely on the raw map stored in `rel.attributes`?
  Audit `export/` scripts and dashboards before shipping.
- Should we version attribute payloads (e.g. `{"v": 1, "data": {...}}`) to ease
  future schema changes without breaking consumers?

## Definition of Done
- Ingestion completes without dropping edge batches due to type errors.
- Neo4j logs are free of `UnknownPropertyKeyWarning` for `section_distribution`.
- QA/UI responses faithfully return attribute metadata reconstructed from the
  serialised form, verified via tests and manual spot checks.
