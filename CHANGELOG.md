# Changelog

## Unreleased
- add a CLI utility and helper function to check the API /health endpoint connectivity
- implement Phase 6 orchestrator with co-mention fallback and `/api/extract/{paper_id}` endpoint
- add processed chunk registry and default dependency wiring for end-to-end extraction
- introduce Phase 7 graph-first QA pipeline with entity resolution, multi-hop path search, fallback evidence summaries, and `/api/qa/ask` endpoint
- generalize relation normalization and polysemy controls via config to support domain-agnostic graphs
- deliver Phase 8 UI with upload workflow, table-driven graph explorer, evidence drilldowns, QA panel surfaced on the landing page, and Playwright coverage
- enable Phase 9 shareable export links with configurable storage, revocation endpoints, and UI share controls supporting copy + revoke flows
- speed up Docker builds by caching pip downloads, installing via requirements files, and guarding dependency parity with regression tests
- switch UI upload flow to JSON/base64 payloads to remove the python-multipart dependency and ensure Docker builds succeed behind restricted proxies
- fail fast during Docker builds when the configured package index is unreachable, providing actionable guidance for proxy-restricted environments
