# Frontend Overview

The Next.js frontend layers thin UI flows on top of the backend API surface that powers ingestion, graph exploration, and QA. The current integration assumes the backend exposes the UI helper endpoints shipped with Phase 8 of the product plan.

## HTTP client

Shared Axios configuration lives in [`lib/http.ts`](./lib/http.ts). The helper exports a preconfigured instance that automatically respects `NEXT_PUBLIC_API_URL`, along with utilities for composing base URLs and extracting human-readable error messages from FastAPI responses.

## Ingestion workflow (`/ingestion`)

* Streams uploaded PDFs as base64 payloads to `POST /api/ui/upload` with optional overrides for the paper ID.
* Surfaces backend-provided status, error messages, and edge/node write metrics in the activity table.
* Retries transient upload failures with exponential backoff to better handle long-running uploads.

## Papers dashboard (`/papers`)

* Lists all registered papers from `GET /api/ui/papers`, including metadata, graph counts, and pipeline status.
* Polls active jobs until the backend transitions them out of `running`/`pending` states.
* Mirrors ingestion metadata formatting so authors, venues, and DOIs are consistently displayed across screens.

## Graph explorer (`/graph`)

* Fetches graph snapshots through `GET /api/ui/graph`, respecting the default filters returned by `GET /api/ui/settings`.
* Presents nodes and edges as sortable tables so the UI no longer depends on Cytoscape assets.
* Highlights edge evidence, conflicting flags, and timestamps to align with the backend contracts.

## QA panel

* Available on the landing page and anywhere the `QA Panel` component is embedded.
* Sends natural-language questions to `POST /api/qa/ask`, displaying step-by-step reasoning, answers, and cited paths.
* Offers mode toggles for retrieval-only vs. retrieval-plus-generation behaviour based on backend responses.

## Development tips

Run `npm run dev` in `frontend/` with `NEXT_PUBLIC_API_URL` pointing at a local FastAPI instance. When adding new API calls, import the shared `apiClient` to inherit consistent error handling and base URL logic.
