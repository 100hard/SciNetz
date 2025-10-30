# Observability dashboard quickstart

The Phase 10 dashboard is exposed directly from the FastAPI backend so you can inspect pipeline health without deploying the full frontend.

## Prerequisites
- A configured `config.yaml` so the observability service knows where to persist JSONL artifacts.
- At least one pipeline run (or QA/export event) so that the manifest, KPI, QA, and export files exist. If you have not run a pipeline yet, the dashboard renders an empty state.

## Option 1: Run with Docker Compose
1. Copy `.env.example` to `.env` and fill in any required secrets (OpenAI key, Google client IDs, etc.).
2. Start the services:
   ```bash
   docker compose up api
   ```
   The backend container exposes port `8000` by default.
3. Open [http://localhost:8000/observability/dashboard](http://localhost:8000/observability/dashboard) in your browser.

## Option 2: Run locally with Uvicorn
1. Install the backend dependencies (either with Poetry or pip):
   ```bash
   pip install -r requirements/base.txt
   ```
2. Export any required environment variables (see `config.yaml` and `.env.example`).
3. Launch the FastAPI factory:
   ```bash
   uvicorn --factory backend.app.main:create_app --host 0.0.0.0 --port 8000
   ```
4. Visit [http://localhost:8000/observability/dashboard](http://localhost:8000/observability/dashboard).

## Regenerating data
The dashboard reflects whatever metrics JSONL files are present in the configured observability directory. Trigger new pipeline runs, QA questions, or export share actions to update those files; the dashboard reads them on every page load, so no restart is required.

## What the dashboard shows
- **Knowledge graph quality panel:** Highlights the most recent run's acceptance rate, rejection reasons, relation coverage, and edges-per-node ratios, alongside manual audit statistics and any semantic drift events detected from prior runs.
- **Run manifests:** Per-paper counts (attempted vs. accepted triples, nodes, edges) correlated by `run_id`.
- **KPI status:** Rolling history of the Phase 10 ship/no-ship guardrails with their latest status indicators.
- **Export lifecycle telemetry:** Share-link creation and download activity, first-download latency, and guardrail warnings.
- **QA performance & soft failures:** Latency percentiles, fallback usage, and zero-path counts for QA traffic.
- **Alerts & errors:** The latest quality alerts with context plus recent pipeline errors to speed up triage.
