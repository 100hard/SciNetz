# Decision Log

## 2024-01-15: Why two-pass span linking?
Context: LLM offsets unreliable
Decision: Separate LLM (relation) from linker (offsets)
Consequences: More complex but 95%+ validation pass rate

## 2025-02-18: Replace Cytoscape graph UI with backend-aligned explorer
Context: Imported frontend diverged from available APIs and Cytoscape assets were missing
Decision: Rebuild the graph explorer around tabular summaries, shared Axios client, and `/api/ui/graph` responses so the UI mirrors backend contracts
Consequences: Users can inspect graph metadata without Cytoscape, reuse error handling, and leverage the same settings defaults as the ingestion dashboard

