"""
Whiteboard Grading API routes — app/api/routes/whiteboard.py
=============================================================
POST /api/v1/whiteboard/grade  — Grade an architecture diagram during a mock interview

Accepts the Excalidraw elements JSON from the frontend, along with interview context
(prompt, topic, difficulty, recent conversation messages) and returns structured
AI feedback that the interview page injects back into the conversation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.whiteboard_service import whiteboard_grading_service
from app.services.llm_service import LLMOverloadedError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/whiteboard", tags=["whiteboard"])


# ─── Request / Response Models ─────────────────────────────────────────────────

class ConversationMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class WhiteboardGradeRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=3,
        description="The interview design problem (e.g. 'Design Twitter's feed system')",
    )
    topic: str = Field(
        default="system_design",
        description="Interview topic: system_design, low_level_design, behavioral, ml_design, dsa",
    )
    difficulty: str = Field(
        default="mid_senior",
        description="Difficulty level: junior, mid_senior, staff, principal",
    )
    elements: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Excalidraw elements array from the frontend canvas",
    )
    conversation_messages: Optional[List[ConversationMessage]] = Field(
        default=None,
        description="Recent interview conversation messages for context-aware grading",
    )


class AnnotationsResponse(BaseModel):
    correct: List[str]
    missing: List[str]
    suggestions: List[str]


class WhiteboardGradeResponse(BaseModel):
    feedback: str
    follow_up: str
    annotations: AnnotationsResponse
    diagram_score: int
    diagram_verdict: str


# ─── Routes ────────────────────────────────────────────────────────────────────

@router.post("/grade", response_model=WhiteboardGradeResponse)
async def grade_whiteboard(req: WhiteboardGradeRequest):
    """
    Grade an architecture diagram in the context of an ongoing interview session.

    The AI interviewer analyzes the diagram elements, cross-references them
    against the interview prompt and conversation history, and returns:
    - Natural conversational feedback (injected into chat)
    - Structured annotations (missing/correct/suggestions)
    - A sharp follow-up question to continue the interview
    """
    if not req.elements:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No diagram elements provided. Please draw something on the whiteboard first.",
        )

    try:
        # Convert Pydantic conversation messages to plain dicts
        conv_messages = None
        if req.conversation_messages:
            conv_messages = [
                {"role": m.role, "content": m.content}
                for m in req.conversation_messages
            ]

        result = await whiteboard_grading_service.grade_diagram(
            prompt=req.prompt,
            topic=req.topic,
            difficulty=req.difficulty,
            elements=req.elements,
            conversation_messages=conv_messages,
        )

        return WhiteboardGradeResponse(
            feedback=result["feedback"],
            follow_up=result["follow_up"],
            annotations=AnnotationsResponse(**result["annotations"]),
            diagram_score=result["diagram_score"],
            diagram_verdict=result["diagram_verdict"],
        )

    except LLMOverloadedError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI diagram evaluator is temporarily unavailable. Please try again shortly.",
            headers={"Retry-After": "15"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to grade whiteboard diagram")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to grade diagram: {str(e)}",
        )
