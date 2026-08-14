"""
POST /query
Full (non-streaming) graph invocation. Returns a complete QueryResponse
once the graph reaches END.

"""
import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.api.schemas import ConfidenceDetail, QueryRequest, QueryResponse
from app.config.settings import settings
from app.core.exceptions import ARIError
from app.core.logging import current_run_id, get_logger
from app.graph.build_graph import get_compiled_graph, get_recursion_limit
from app.observability.metrics import extract_run_metrics
from app.security.auth import check_rate_limit, verify_api_key
from app.security.guardrails import validate_input_security

logger = get_logger(__name__)
router = APIRouter()


def _build_initial_state(req: QueryRequest, conversation_id: str) -> dict:
    return {
        "question": req.question,
        "original_question": req.question,
        "conversation_id": conversation_id,
        "run_id": str(uuid.uuid4()),
        "retry_count": 0,
        "issues_log": [],
    }


def _extract_response(final_state: dict, conversation_id: str, thread_id: str) -> QueryResponse:
    confidence_raw = final_state.get("confidence") or {}

    confidence = ConfidenceDetail(
        confidence_score=confidence_raw.get("confidence_score", 0.0),
        hallucination_risk=confidence_raw.get("hallucination_risk", 0.0),
        retrieval_quality=confidence_raw.get("retrieval_quality", 0.0),
        reflection_score=confidence_raw.get("reflection_score", 0.0),
        citation_quality=confidence_raw.get("citation_quality", 0.0),
        num_sources=confidence_raw.get("num_sources", 0),
        confidence_level=confidence_raw.get("confidence_level", "low"),
        reason=confidence_raw.get("reason", ""),
    )

    return QueryResponse(
        answer=final_state.get("answer", ""),
        citations=final_state.get("citations", []),
        confidence=confidence,
        strategies_used=final_state.get("strategies", []),
        retry_count=final_state.get("retry_count", 0),
        conversation_id=conversation_id,
        thread_id=thread_id,
        clarification_needed=final_state.get("clarification_needed", False),
        clarification_question=final_state.get("clarification_question"),
        memory_context=final_state.get("long_term_context"),
        metadata=extract_run_metrics(final_state).to_dict(),
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def query(req: QueryRequest, request: Request) -> QueryResponse:
    # --- Security: validate input BEFORE touching the graph ---
    is_safe, block_reason = validate_input_security(req.question)
    if not is_safe:
        logger.warning(
            "query_blocked_by_guardrail",
            extra={"question": req.question[:100], "reason": block_reason},
        )
        raise HTTPException(status_code=400, detail=block_reason)

    conversation_id = req.resolved_conversation_id()
    thread_id = req.resolved_thread_id(conversation_id)
    run_id = str(uuid.uuid4())

    current_run_id.set(run_id)

    logger.info(
        "query_received",
        extra={
            "question": req.question[:100],
            "conversation_id": conversation_id,
            "thread_id": thread_id,
        },
    )

    graph = get_compiled_graph()
    initial_state = _build_initial_state(req, conversation_id)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": get_recursion_limit(),
    }

    try:
        if hasattr(graph, "ainvoke"):
            res = graph.ainvoke(initial_state, config)
            if asyncio.iscoroutine(res) or hasattr(res, "__await__"):
                final_state = await res
            elif isinstance(res, dict):
                final_state = res
            else:
                final_state = await asyncio.to_thread(graph.invoke, initial_state, config)
        else:
            final_state = await asyncio.to_thread(graph.invoke, initial_state, config)
    except ARIError as exc:


        logger.error("query_ari_error", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("query_unexpected_error", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail="Internal server error")

    response = _extract_response(final_state, conversation_id, thread_id)

    logger.info(
        "query_completed",
        extra={
            "conversation_id": conversation_id,
            "confidence_level": response.confidence.confidence_level,
            "retry_count": response.retry_count,
            "num_citations": len(response.citations),
        },
    )

    return response


@router.post("/ingestion/upload")
async def upload_file(file: UploadFile = File(...)):
    import uuid
    import json
    import pypdf
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from app.retrievers.registry import clear_retriever_caches
    from app.retrievers.vector_search import VectorSearchRetriever
    from app.retrievers.hybrid_search import BM25_INDEX_PATH

    filename = file.filename
    content_type = file.content_type or ""
    text = ""

    try:
        if filename.endswith(".pdf"):
            reader = pypdf.PdfReader(file.file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        else:
            # Assume text/plain
            content = await file.read()
            text = content.decode("utf-8", errors="ignore")

        if not text.strip():
            raise HTTPException(status_code=400, detail="Uploaded file is empty or could not be parsed.")

        # Chunk text using LangChain character splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.upload_chunk_size,
            chunk_overlap=settings.upload_chunk_overlap,
        )
        chunks = splitter.split_text(text)

        if not chunks:
            raise HTTPException(status_code=400, detail="No text chunks generated.")

        # Add to Chroma Vector Database
        vector_retriever = VectorSearchRetriever()
        collection = vector_retriever._get_collection()

        ids = [f"uploaded_{uuid.uuid4()}" for _ in chunks]
        metadatas = [{"source": filename} for _ in chunks]
        collection.add(ids=ids, documents=chunks, metadatas=metadatas)

        # Append to BM25 JSON Corpus Index
        corpus = []
        if BM25_INDEX_PATH.exists():
            with open(BM25_INDEX_PATH, "r") as f:
                try:
                    corpus = json.load(f)
                except Exception:
                    corpus = []

        for chunk_text in chunks:
            corpus.append({
                "content": chunk_text,
                "source": filename,
                "metadata": {"source": filename}
            })

        BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BM25_INDEX_PATH, "w") as f:
            json.dump(corpus, f, indent=2)

        # Clear retriever instance caches to force lazy load of new BM25 index on next query
        clear_retriever_caches()

        logger.info(
            "file_ingested_successfully",
            extra={"ingested_filename": filename, "chunks": len(chunks), "size_bytes": len(text)}
        )

        return {"status": "ok", "filename": filename, "chunks_ingested": len(chunks)}

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("file_ingestion_failed", extra={"ingested_filename": filename, "error": str(exc)})
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(exc)}")
