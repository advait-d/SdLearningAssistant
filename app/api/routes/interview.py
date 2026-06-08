"""
Interview API routes — app/api/routes/interview.py
===================================================
POST /api/v1/interview/start  — Start a new mock interview session
POST /api/v1/interview/turn   — Process one candidate turn
POST /api/v1/interview/end    — End interview and generate scorecard
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.interview_service import (
    interview_service,
    InterviewTopic,
    InterviewDifficulty,
)
from app.services.llm_service import LLMOverloadedError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/interview", tags=["interview"])


# ─── Request / Response Models ─────────────────────────────────────────────────

class StartRequest(BaseModel):
    topic: str = Field(
        ...,
        description="Interview topic: system_design, low_level_design, behavioral, ml_design, dsa"
    )
    prompt: str = Field(
        ...,
        min_length=3,
        description="The design problem or question to interview on"
    )
    difficulty: str = Field(
        default="mid_senior",
        description="Difficulty level: junior, mid_senior, staff, principal"
    )
    company: str = Field(
        default="general",
        description="Target company: general, google, meta, amazon, stripe, netflix"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID. Auto-generated if not provided."
    )


class TurnRequest(BaseModel):
    session_id: str = Field(..., description="Session ID from the start response")
    message: str = Field(..., min_length=1, description="Candidate's response")


class EndRequest(BaseModel):
    session_id: str = Field(..., description="Session ID to end")


# ─── Routes ────────────────────────────────────────────────────────────────────

@router.post("/start")
async def start_interview(req: StartRequest):
    """Start a new AI mock interview session."""

    # Validate topic
    try:
        topic = InterviewTopic(req.topic)
    except ValueError:
        valid = [t.value for t in InterviewTopic]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid topic '{req.topic}'. Valid options: {valid}",
        )

    # Validate difficulty
    try:
        difficulty = InterviewDifficulty(req.difficulty)
    except ValueError:
        valid = [d.value for d in InterviewDifficulty]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid difficulty '{req.difficulty}'. Valid options: {valid}",
        )

    session_id = req.session_id or str(uuid.uuid4())

    try:
        result = await interview_service.start_session(
            session_id=session_id,
            topic=topic,
            prompt=req.prompt,
            difficulty=difficulty,
            company=req.company,
        )
        return result

    except LLMOverloadedError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI interviewer is temporarily unavailable due to high demand. Please try again shortly.",
            headers={"Retry-After": "30"},
        )
    except Exception as e:
        logger.exception("Failed to start interview session")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start interview: {str(e)}",
        )


@router.post("/turn")
async def interview_turn(req: TurnRequest):
    """Process one candidate turn and get the interviewer's response."""

    try:
        result = await interview_service.process_turn(
            session_id=req.session_id,
            user_message=req.message,
        )

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"],
            )

        return result

    except HTTPException:
        raise
    except LLMOverloadedError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI interviewer is temporarily unavailable. Your session is preserved — try again shortly.",
            headers={"Retry-After": "15"},
        )
    except Exception as e:
        logger.exception("Failed to process interview turn")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process turn: {str(e)}",
        )


@router.post("/end")
async def end_interview(req: EndRequest):
    """End the interview and generate a structured scorecard."""

    try:
        result = await interview_service.end_session(session_id=req.session_id)

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"],
            )

        return result

    except HTTPException:
        raise
    except LLMOverloadedError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI is temporarily unavailable for scorecard generation. Please try again shortly.",
            headers={"Retry-After": "30"},
        )
    except Exception as e:
        logger.exception("Failed to end interview and generate scorecard")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate scorecard: {str(e)}",
        )
