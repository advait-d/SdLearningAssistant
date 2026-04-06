"""
Pydantic models that mirror the IntentService dataclasses.
Used for serialising intent metadata in API responses / internal passing.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class IntentResultModel(BaseModel):
    """
    Serialisable representation of an IntentService classification result.

    Mirrors ``app.services.intent_service.IntentResult`` so that intent data
    can be included in HTTP responses or logged without importing service
    internals into route handlers.
    """

    label: str = Field(
        ...,
        description=(
            "One of: CONCEPT_EXPLANATION, SYSTEM_DESIGN_QUESTION, "
            "DESIGN_REVIEW, OUT_OF_SCOPE."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model self-reported classification confidence [0.0 – 1.0].",
    )
    raw: str = Field(
        ...,
        description="Raw label string returned by the model before parsing.",
    )
    query: str = Field(
        ...,
        description="The original query that was classified.",
    )
