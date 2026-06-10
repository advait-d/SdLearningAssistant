from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import logging
from app.services.roadmap_service import roadmap_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["roadmap"])

@router.post("/generate")
async def generate_roadmap(
    file: UploadFile = File(...),
    target_role: str = Form(...),
    timeline_days: int = Form(30)
):
    """
    Generates a personalized study roadmap from a resume.
    """
    try:
        content = await file.read()
        plan = await roadmap_service.generate_roadmap_from_pdf(
            pdf_bytes=content,
            target_role=target_role,
            timeline_days=timeline_days
        )
        return {"status": "success", "data": plan}
    except Exception as e:
        logger.error(f"Failed to generate roadmap: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during roadmap generation.")
