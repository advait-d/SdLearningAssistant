"""
Request and response models for the /chat endpoint.
"""

from __future__ import annotations

from typing import Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore[assignment]
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Payload sent by the client to POST /chat."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's natural-language query or design question.",
        examples=["Design a URL shortener for 1 billion users."],
    )
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description=(
            "Opaque client-supplied session identifier used to maintain "
            "conversation history across turns."
        ),
        examples=["user-abc123"],
    )
    provider: Literal["openai", "gemini"] = Field(
        default="gemini",
        description="The LLM provider to use for this request.",
    )


class ChatResponse(BaseModel):
    """Structured payload returned by POST /chat."""

    response: str = Field(
        ...,
        description="The assistant's answer or clarification request.",
    )
    intent: str = Field(
        ...,
        description=(
            "Classified intent label: CONCEPT_EXPLANATION, "
            "SYSTEM_DESIGN_QUESTION, DESIGN_REVIEW, or OUT_OF_SCOPE."
        ),
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Evaluator confidence score [0.0 – 1.0]. "
            "Null when the request was rejected as OUT_OF_SCOPE."
        ),
    )
    is_fallback: bool = Field(
        False,
        description=(
            "True when the evaluator determined the initial response was "
            "low-confidence and replaced it with a clarification request."
        ),
    )
    session_id: str = Field(
        ...,
        description="Echo of the session_id from the request.",
    )


class ErrorResponse(BaseModel):
    """Standard error envelope returned on 4xx / 5xx."""

    detail: str = Field(..., description="Human-readable error description.")
    error_code: Optional[str] = Field(
        None, description="Machine-readable error code for client handling."
    )
