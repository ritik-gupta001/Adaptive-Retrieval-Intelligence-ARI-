# ARI Frontend — Vite + React Dashboard

Production UI for the Adaptive Retrieval Intelligence (ARI) platform.

## Features
- Real-time Server-Sent Events (SSE) stream processing
- Interactive LangGraph pipeline execution visualizer (SVG DAG)
- Adaptive strategy badges and decision explanations
- Real-time confidence score gauges, hallucination risk, and citation quality meters
- Document upload (.pdf, .txt) with automatic chunking and vector indexing
- Inline citations and session memory inspector

## Local Development

```bash
# Install dependencies
npm install

# Start local dev server (port 3000)
npm run dev
```

## Production Build

```bash
# Build optimized static bundle to frontend/dist/
npm run build

# Preview production build locally
npm run preview
```
