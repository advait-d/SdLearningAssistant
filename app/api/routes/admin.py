"""
Admin / diagnostic routes.

These endpoints are intended for internal tooling and are NOT part of the
public API contract.  Protect them behind an API key or network policy
before exposing to the internet.

Routes
------
GET  /admin/session/{session_id}   — inspect session memory
DELETE /admin/session/{session_id} — clear session memory
POST /admin/retrieve               — run a raw RAG retrieval query
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.models import RetrievalResultModel, RetrievedChunkModel
from app.services.memory_service import get_history, clear_history
from app.services.retriever_service import retriever_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])


# ---------------------------------------------------------------------------
# Session inspection
# ---------------------------------------------------------------------------


class SessionHistoryResponse(BaseModel):
    session_id: str
    turn_count: int
    messages: List[Dict[str, Any]]


@router.get(
    "/session/{session_id}",
    response_model=SessionHistoryResponse,
    summary="Inspect session memory",
)
async def get_session(session_id: str) -> SessionHistoryResponse:
    """Return the raw conversation history stored for a session."""
    history = get_history(session_id)
    return SessionHistoryResponse(
        session_id=session_id,
        turn_count=len(history) // 2,
        messages=history,
    )


@router.delete(
    "/session/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Clear session memory",
    response_class=Response,
)
async def delete_session(session_id: str) -> Response:
    """Wipe the conversation history for a given session."""
    clear_history(session_id)
    logger.info("Cleared session memory for session_id=%s", session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# RAG retrieval probe
# ---------------------------------------------------------------------------


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(5, ge=1, le=20)


@router.post(
    "/retrieve",
    response_model=RetrievalResultModel,
    summary="Run a raw RAG retrieval query",
)
async def retrieve(payload: RetrieveRequest) -> RetrievalResultModel:
    """
    Directly query the FAISS retriever with a raw string and inspect what
    context would be injected into the LLM prompt for a given query.
    """
    try:
        context_text = await retriever_service.retrieve_context(payload.query)
    except Exception as exc:
        logger.exception("Retrieval probe failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}",
        )

    # Also fetch the structured chunks for the detailed response
    chunks: List[RetrievedChunkModel] = []
    provider = getattr(payload, "provider", "openai")
    pipeline = retriever_service._pipelines.get(provider)
    if retriever_service._index_loaded.get(provider) and pipeline:
        try:
            raw_chunks = pipeline.retrieve_top_k(
                payload.query,
                k=payload.top_k,
                score_threshold=retriever_service.SCORE_THRESHOLD,
                provider=provider,
            )
            chunks = [
                RetrievedChunkModel(
                    text=c.text,
                    source=c.source,
                    chunk_index=c.chunk_index,
                    score=c.score,
                    metadata=c.metadata,
                )
                for c in raw_chunks
            ]
        except Exception:
            pass  # context_text already retrieved; chunks are bonus detail

    return RetrievalResultModel(
        query=payload.query,
        chunks=chunks,
        context_text=context_text,
    )
