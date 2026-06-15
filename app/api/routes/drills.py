import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.drill_service import drill_service

router = APIRouter(prefix="/drills", tags=["drills"])
logger = logging.getLogger(__name__)

class GenerateRequest(BaseModel):
    topic: str
    difficulty: str = "Medium"

class EvaluateRequest(BaseModel):
    topic: str
    prompt: str
    transcript: str

@router.post("/generate")
async def generate_drill(req: GenerateRequest):
    try:
        data = await drill_service.generate_prompt(req.topic, req.difficulty)
        return data
    except Exception as e:
        logger.error(f"Error generating drill: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate drill prompt")

@router.post("/evaluate")
async def evaluate_drill(req: EvaluateRequest):
    try:
        data = await drill_service.evaluate_answer(req.topic, req.prompt, req.transcript)
        return data
    except Exception as e:
        logger.error(f"Error evaluating drill: {e}")
        raise HTTPException(status_code=500, detail="Failed to evaluate drill answer")
