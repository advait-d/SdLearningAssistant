"""
app.models — Pydantic request / response schemas.

Import from here to keep route handlers clean:
    from app.models import ChatRequest, ChatResponse
"""

from app.models.chat import ChatRequest, ChatResponse, ErrorResponse
from app.models.intent import IntentResultModel
from app.models.eval import EvalResultModel
from app.models.retrieval import RetrievedChunkModel, RetrievalResultModel

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ErrorResponse",
    "IntentResultModel",
    "EvalResultModel",
    "RetrievedChunkModel",
    "RetrievalResultModel",
]
