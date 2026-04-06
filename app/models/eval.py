"""
Pydantic models for evaluator / confidence scoring results.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalResultModel(BaseModel):
    """
    Serialisable result from EvaluatorService.evaluate_response().

    Captures the three values returned by ``evaluate_response()``:
    is_fallback, final_response, and confidence_score.
    """

    is_fallback: bool = Field(
        ...,
        description=(
            "True when the evaluator replaced the LLM's initial response "
            "with a clarification fallback due to low confidence."
        ),
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Evaluator confidence score assigned to the response [0.0 – 1.0].",
    )
    final_response: str = Field(
        ...,
        description=(
            "The response surfaced to the user — either the original LLM "
            "answer (if confidence was acceptable) or a generated fallback."
        ),
    )
