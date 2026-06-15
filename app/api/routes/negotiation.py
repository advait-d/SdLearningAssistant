from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from app.services.negotiation_service import negotiation_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["negotiation"])


class Message(BaseModel):
    role: str
    content: str


class NegotiationRequest(BaseModel):
    messages: List[Message]
    scenario: str
    target_company: str
    target_level: str
    base_target: str
    candidate_skills: Optional[List[str]] = None


@router.post("/chat")
async def chat_negotiation(request: NegotiationRequest):
    """
    Continues a negotiation roleplay conversation.
    Optionally accepts candidate_skills to personalize the recruiter persona.
    """
    try:
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        reply = await negotiation_service.chat(
            messages=messages_dict,
            scenario=request.scenario,
            target_company=request.target_company,
            target_level=request.target_level,
            base_target=request.base_target,
            candidate_skills=request.candidate_skills,
        )

        return {"status": "success", "reply": reply}
    except Exception as e:
        logger.error(f"Negotiation chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during negotiation.")
