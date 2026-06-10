from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.services.resume_service import resume_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["resume"])

@router.post("/review")
async def review_resume(
    file: UploadFile = File(...),
    target_role: str = Form("Software Engineer")
):
    """
    Accepts a PDF resume upload, extracts text, and uses an LLM to provide structured feedback.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        feedback = await resume_service.review_resume(file, target_role)
        return {"status": "success", "data": feedback}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Failed to review resume: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while processing resume.")
