import io
import logging
from pypdf import PdfReader
from fastapi import UploadFile
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

class ResumeService:
    async def review_resume(self, file: UploadFile, target_role: str = "Software Engineer") -> dict:
        try:
            content = await file.read()
            
            # Parse PDF
            reader = PdfReader(io.BytesIO(content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
                
            if not text.strip():
                raise ValueError("Could not extract any text from the PDF. Is it an image-based PDF?")
                
            system_prompt = f"""You are an elite Staff Software Engineer and hiring manager at a top-tier tech company.
Your goal is to review a candidate's resume for a '{target_role}' role.
Provide structural feedback on their resume. Be extremely critical, focusing on system design, impact, metrics, and clarity.

Output JSON strictly in this format:
{{
  "score": <0-100 integer>,
  "estimated_level": "...", 
  "target_level": "{target_role}",
  "weak_bullets_count": <integer>,
  "missing_metrics_count": <integer>,
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "action_items": ["...", "..."],
  "overall_summary": "...",
  "extracted_skills": ["...", "..."]
}}
"""
            
            user_input = f"Here is the resume text:\n\n{text}"
            
            response = await llm_service.generate_structured_response(
                system_prompt=system_prompt,
                user_input=user_input,
                provider="gemini",
                model="gemini-2.5-flash",
                temperature=0.3
            )
            
            
            return {
                "feedback": response,
                "resume_text": text[:4000]
            }
        except Exception as e:
            logger.error(f"Error parsing or reviewing resume: {str(e)}")
            raise

resume_service = ResumeService()
