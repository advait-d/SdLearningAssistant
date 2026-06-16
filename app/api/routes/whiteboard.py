"""
Whiteboard Grading API routes — app/api/routes/whiteboard.py
=============================================================
POST /api/v1/whiteboard/grade  — Grade an architecture diagram during a mock interview
POST /api/v1/whiteboard/followup — Process user answers to follow-up questions
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.whiteboard_service import whiteboard_grading_service
from app.services.llm_service import LLMOverloadedError, llm_service

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
        description="Interview topic",
    )
    difficulty: str = Field(
        default="mid_senior",
        description="Difficulty level",
    )
    elements: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Excalidraw elements array from the frontend canvas",
    )
    conversation_messages: Optional[List[ConversationMessage]] = Field(
        default=None,
        description="Recent interview conversation messages for context-aware grading",
    )


class Issue(BaseModel):
    text: str
    node_label: Optional[str]


class IssuesResponse(BaseModel):
    critical: List[Issue]
    important: List[Issue]
    nice_to_have: List[Issue]


class DimensionsResponse(BaseModel):
    coverage: int
    scalability: int
    reliability: int
    cost: int
    security: int
    observability: int
    tradeoff_quality: int


class WhiteboardGradeResponse(BaseModel):
    feedback: str
    hiring_recommendation: str
    architecture_score: int
    dimensions: DimensionsResponse
    issues: IssuesResponse
    follow_up_questions: List[str]


class FollowUpRequest(BaseModel):
    question: str
    answer: str
    prompt: str
    current_score: int


class FollowUpResponse(BaseModel):
    feedback: str
    score_delta: int
    new_hiring_recommendation: str


# ─── Routes ────────────────────────────────────────────────────────────────────

@router.post("/grade", response_model=WhiteboardGradeResponse)
async def grade_whiteboard(req: WhiteboardGradeRequest):
    """
    Grade an architecture diagram in the context of an ongoing interview session.
    """
    if not req.elements:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No diagram elements provided. Please draw something on the whiteboard first.",
        )

    try:
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
            hiring_recommendation=result["hiring_recommendation"],
            architecture_score=result["architecture_score"],
            dimensions=DimensionsResponse(**result["dimensions"]),
            issues=IssuesResponse(**result["issues"]),
            follow_up_questions=result["follow_up_questions"],
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

@router.post("/followup", response_model=FollowUpResponse)
async def process_followup(req: FollowUpRequest):
    """
    Process a candidate's answer to a follow-up question and dynamically adjust their score.
    """
    try:
        eval_prompt = f"""You are an expert system design interviewer. 
The candidate is designing: {req.prompt}

You previously asked this follow-up question regarding their architecture:
"{req.question}"

The candidate responded:
"{req.answer}"

Evaluate their answer. Respond ONLY with valid JSON in this exact shape:
{{
  "feedback": "<2-3 sentence reaction to their answer. Either accept their tradeoff or correct them.>",
  "score_delta": <integer between -5 and +10 based on answer quality>,
  "new_hiring_recommendation": "<Strong Hire|Hire|Leaning Hire|No Hire>"
}}
"""
        raw = await llm_service.generate_structured_response(
            system_prompt="You are an expert Staff-level technical interviewer.",
            user_input=eval_prompt,
            provider="gemini",
            model="gemini-2.5-flash",
            temperature=0.3,
        )

        return FollowUpResponse(
            feedback=raw.get("feedback", "Thanks for clarifying that tradeoff."),
            score_delta=raw.get("score_delta", 0),
            new_hiring_recommendation=raw.get("new_hiring_recommendation", "Leaning Hire")
        )

    except Exception as e:
        logger.exception("Failed to process followup")
        return FollowUpResponse(
            feedback="Thanks for the explanation.",
            score_delta=0,
            new_hiring_recommendation="Leaning Hire"
        )
