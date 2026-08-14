# ARI — Adaptive Retrieval Intelligence

A production-grade Agentic RAG platform that dynamically selects the best retrieval strategy per question, with self-reflection, validation, confidence estimation, and persistent memory. Built with LangGraph, FastAPI, and Vite + React.

## DEMO
<img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/bace8707-b4b7-436d-aa35-51ac67cb3e0a" />

<img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/df46a2c0-c10d-4b2c-a489-d5d2102f9bb6" />

```text
question ──► memory_load ──► query_understanding ──► adaptive_router
                                                             │
        ┌────────────────────────────────────────────────────┘
        ▼
   retrieve ──► document_merge ──► rerank ──► context_validate ──► generate
        ▲                                           │                  │
        │                                           ▼                  ▼
        └─────────── query_rewrite ◄───────── [retry loop] ◄──── reflect
                                                    │
                                                    ▼
                                             confidence_score ──► finalize ──► memory_save ──► answer
```

---

## Workspace Structure

The project layout is structured as follows:

```text
ARI/
├── app/                      # Main application package
│   ├── api/                  # FastAPI routes (REST, SSE Server-Sent Events)
│   ├── config/               # Settings management and environment validation
│   ├── core/                 # Shared utilities (logging, exceptions, LLM connectors)
│   ├── graph/                # StateGraph orchestration & Compiled LangGraph pipeline
│   ├── nodes/                # Work Nodes executing state transitions
│   ├── retrievers/           # Query retrievers (Vector, BM25 Hybrid, Multi-Query, Graph RAG, Web)
│   ├── utils/                # Query intent helpers and deduplication logic
│   └── schemas/              # Pydantic schemas for structured LLM validations
├── data/                     # Ingested indices and local storage (Git ignored)
│   ├── chroma/               # Chroma Vector DB collections
│   └── raw/                  # Data documents for local indexing
├── docker/                   # Docker deployment configurations
├── evals/                    # Test dataset and Ragas evaluation runners
├── frontend/                 # Vite + React application (Dashboard UI)
│   ├── src/                  # React components & dashboard layout
│   └── dist/                 # Production compiled build output
├── ingestion/                # Document extraction, BM25, and Graph index builders
├── tests/                    # Pytest suite (Unit & Integration tests)
├── .env.example              # Environment variables template
├── .gitignore                # Folder/secret exclusions
├── CHANGELOG.md              # Project version and change history
├── netlify.toml              # Netlify SPA build & redirect configurations
├── pyproject.toml            # Package configuration and dependencies (ranges)
├── pytest.ini                # Pytest runners configuration
├── render.yaml               # Render blueprint descriptor for backend hosting

├── requirements.lock         # Exact pinned dependency lockfile for deployment
└── requirements.txt          # Unpinned package dependency list
```

---

## Core Capabilities

- **Adaptive Routing**: Evaluates query complexity and freshness constraints. Configurable via `DOCUMENT_REFERENCE_KEYWORDS` and intent detection. Defaults to local searches, then falls back to Web Search (Tavily) on retry if local retrieval is irrelevant or exhausted.
- **Context Validation**: Inspects retrieved chunks for coverage and relevance score. Prevents generation if they are off-topic or empty.
- **Self-Correction & Reflection**: Examines the generated response against context to detect hallucination risks. Retries with a rewritten query if check fails.
- **Cross-turn Memory Persistence**: Employs LangGraph checkpointers to preserve session memory across all turns.
- **Real-time SSE Dashboard**: Live progress tracking, confidence metrics, SVG execution visualization, and inline citations.

---

## Quick Start

### 1. Installation

```bash
# Set up environment variables (.env file)
# Fill in: LLM_PROVIDER, ANTHROPIC_API_KEY, OPENAI_API_KEY, and TAVILY_API_KEY
cp .env.example .env

# Install package in development mode
pip install -e ".[dev]"

# Or install from locked production dependencies
pip install -r requirements.lock
```

### 2. Document & Graph Ingestion

Place PDF, text, or markdown documents in `data/raw/` and index them:

```bash
python -m ingestion.chunk_and_embed ./data/raw
python -m ingestion.build_bm25_index ./data/raw

# Optional: Populate Neo4j Graph RAG index (if GRAPH_RAG_ENABLED=true)
python -m ingestion.build_graph_index
```

### 3. Running Locally

```bash
# Start backend API (Port 8000)
uvicorn app.main:app --reload

# Start local frontend dashboard (Port 3000)
cd frontend
npm install
npm run dev
```

---

## Production Readiness & Security Posture

- **CORS Configuration**: If `ALLOW_ORIGINS` environment variable is unset or `"*"`, `allow_credentials` is set to `False` and a warning is logged at startup to ensure browser CORS spec compliance. When explicit origin URLs are supplied (e.g. `https://your-frontend.com`), `allow_credentials=True` is enabled.
- **Local Embedding Default**: ChromaDB uses a local embedder by default (`ONNXMiniLM_L6_V2`) configured with `hnsw:space=cosine` to prevent unexpected external API embedding charges.
- **PDF Ingestion Scope**: PDF ingestion in `ingestion/chunk_and_embed.py` processes `.txt` and `.md` files in the target directory, plus PDF files matching the configured `CORE_BOOKS` set (`book-1.pdf` through `book-5.pdf`). Modify `CORE_BOOKS` in `ingestion/chunk_and_embed.py` to index additional custom PDF files.
- **LangSmith Tracing**: Controlled by `LANGSMITH_TRACING_ENABLED` (defaults to `false`). On-demand tracing can be executed via `scripts/run_langsmith_traces.py`.
- **Dependency Pinning Strategy**: Abstract compatibility ranges (`>=`) are maintained in `pyproject.toml` for library use. Exact resolved versions are locked in `requirements.lock` for reproducible production builds in `docker/Dockerfile` and `render.yaml`, verified by CI on every push.
- **Graph RAG Strategy**: Gated by `GRAPH_RAG_ENABLED` in `.env`. Requires Neo4j instance pre-populated via `ingestion/build_graph_index.py`.

---

## Known Limitations

- **Local vs Web Routing Sensitivity**: Adaptive routing relies on keyword matching and context validation relevance scores. If local document embeddings return marginal relevance scores (<0.35), the validator falls back to Web Search. Core technical keywords (`LOCAL_TECH_KEYWORDS`) are protected from false fallback triggers.
- **Cross-turn Memory Integration**: Conversation history is persisted across turns via checkpointers and injected into `generate_node` and `query_understanding_node` prompts to resolve context.

---

## Deployment

- **Backend (Render / Docker / Railway)**: Deploy the FastAPI backend service using `requirements.lock` via `render.yaml` or `docker/Dockerfile`.
- **Frontend (Netlify / Vercel)**: Build with `cd frontend && npm install && npm run build` and publish the generated `frontend/dist` directory. The included `netlify.toml` automatically handles SPA routes and build output.
