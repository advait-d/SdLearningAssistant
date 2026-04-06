"""
Pydantic models for RAG retrieval results.
"""

from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class RetrievedChunkModel(BaseModel):
    """
    Serialisable representation of a single retrieved knowledge-base chunk.

    Mirrors ``app.services.retriever_service.RetrievedChunk`` so retrieval
    results can be included in diagnostic / admin API responses without
    importing service internals into route handlers.
    """

    text: str = Field(..., description="The raw text content of the chunk.")
    source: str = Field(
        ..., description="Source document identifier (filename, URL, etc.)."
    )
    chunk_index: int = Field(
        ..., description="Zero-based index of this chunk within its source document."
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score [0.0 – 1.0]; higher means more relevant.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata inherited from the source document.",
    )


class RetrievalResultModel(BaseModel):
    """
    Aggregated retrieval result for a single query.
    Returned by the /admin/retrieve diagnostic endpoint.
    """

    query: str = Field(..., description="The query that was executed.")
    chunks: List[RetrievedChunkModel] = Field(
        default_factory=list,
        description="Retrieved chunks sorted by descending relevance score.",
    )
    context_text: str = Field(
        ...,
        description=(
            "Pre-formatted context string injected into the LLM prompt — "
            "the concatenation of all chunk texts with headers."
        ),
    )
