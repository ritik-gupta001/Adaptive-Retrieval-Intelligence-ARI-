# Changelog

All notable changes to the ARI platform will be documented in this file.

## [Unreleased] - 2026-08-14

- Regenerated `requirements.lock` from clean Python 3.12 environment, removing self-referencing entries and unrelated global dependencies.
- Added GitHub Actions workflow (`.github/workflows/verify-lockfile.yml`) to enforce lockfile reproducibility on every push.
- Removed hardcoded document reference keywords and refactored adaptive router intent detection logic.
- Fixed unawaited `asyncio.sleep` in LLM retry loop.
- Added `.env.example` with complete configuration schema and documentation.
- Fixed FastAPI CORS wildcard configuration for browser credentials spec compliance.
- Rebuilt frontend dashboard with Vite + React SPA architecture.
- Added Neo4j Graph RAG ingestion script (`ingestion/build_graph_index.py`).
- Deduplicated context deduplication and query string sanitization helper utilities.
