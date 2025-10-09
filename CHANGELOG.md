# Changelog

## Unreleased
- add a CLI utility and helper function to check the API /health endpoint connectivity
- implement Phase 6 orchestrator with co-mention fallback and `/api/extract/{paper_id}` endpoint
- add processed chunk registry and default dependency wiring for end-to-end extraction
- introduce Phase 7 graph-first QA pipeline with entity resolution, multi-hop path search, and `/api/qa/ask` endpoint
- generalize relation normalization and polysemy controls via config to support domain-agnostic graphs
- deliver Phase 8 UI with upload workflow, Cytoscape-powered graph view, evidence drawer, QA panel, and Playwright coverage
- speed up Docker builds by caching pip downloads, installing via requirements files, and guarding dependency parity with regression tests
