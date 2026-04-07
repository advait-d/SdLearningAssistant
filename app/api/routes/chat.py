"""
POST /chat — main conversational endpoint.

Delegates to OrchestratorService which runs the full pipeline:
  intent classification → RAG retrieval → LLM generation → evaluation.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.models import ChatRequest, ChatResponse, ErrorResponse
from app.services.orchestrator_service import orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message to the SD Learning Assistant",
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def chat(payload: ChatRequest) -> ChatResponse:
    """
    **POST /chat**

    Accept a user message and session ID, run the full orchestration pipeline,
    and return a structured response.

    ### Pipeline
    1. **Intent classification** — decide which handler to use
    2. **OUT_OF_SCOPE guard** — return early without hitting the LLM
    3. **RAG retrieval** — fetch relevant knowledge-base context
    4. **LLM generation** — produce an expert system design answer
    5. **Evaluation** — score confidence, trigger fallback if needed
    6. **Memory update** — persist turn to in-memory session store
    """
    logger.info(
        "POST /chat | session=%s | query=%r",
        payload.session_id,
        payload.message[:80],
    )

    try:
        result = await orchestrator.handle_request(
            session_id=payload.session_id,
            user_query=payload.message,
            provider=payload.provider,
        )
    except Exception as exc:
        logger.exception("Unhandled error in orchestrator: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again later.",
        )

    return ChatResponse(
        response=result["response"],
        intent=result["intent"],
        confidence_score=result.get("score"),
        is_fallback=result.get("is_fallback", False),
        session_id=payload.session_id,
    )
