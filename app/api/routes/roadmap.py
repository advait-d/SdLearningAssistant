from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import logging
from app.services.roadmap_service import roadmap_service

from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(tags=["roadmap"])

@router.post("/generate")
async def generate_roadmap(
    file: Optional[UploadFile] = File(None),
    resume_text: Optional[str] = Form(None),
    target_role: str = Form(...),
    timeline_days: int = Form(30)
):
    """
    Generates a personalized study roadmap from a resume.
    """
    try:
        content = await file.read() if file else None
        plan = await roadmap_service.generate_roadmap_from_pdf(
            target_role=target_role,
            timeline_days=timeline_days,
            pdf_bytes=content,
            resume_text=resume_text
        )
        return {"status": "success", "data": plan}
    except Exception as e:
        logger.error(f"Failed to generate roadmap: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during roadmap generation.")
