from typing import List, Dict, Any
import logging
import io
from pypdf import PdfReader
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

class RoadmapService:
    async def generate_roadmap_from_pdf(
        self, 
        pdf_bytes: bytes,
        target_role: str,
        timeline_days: int
    ) -> Dict[str, Any]:
        
        # Extract text from PDF
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            logger.error(f"Error parsing PDF for roadmap: {str(e)}")
            raise ValueError("Could not parse the uploaded PDF file.")

        system_prompt = f"""You are an elite Staff Software Engineer coach. Your goal is to analyze the candidate's resume, identify their architectural gaps and weaknesses, and generate a highly focused, day-by-day study roadmap for them.

Target Role: {target_role}
Time until interview: {timeline_days} days

Candidate Resume:
{text[:4000]}  # limit to first ~4000 chars

Output JSON strictly in this format:
{{
  "overview": "Brief aggressive summary of their gaps and the sprint ahead...",
  "projected_hours": <integer>,
  "weeks": [
    {{
      "week_number": 1,
      "focus": "...",
      "days": [
        {{"day": 1, "topic": "...", "action": "..."}}
      ]
    }}
  ]
}}
Ensure the plan targets their resume weaknesses and matches the short timeline. Only return valid JSON.
"""
        
        user_input = "Analyze my resume and generate the brutalist study roadmap."
        
        try:
            response = await llm_service.generate_structured_response(
                system_prompt=system_prompt,
                user_input=user_input,
                provider="openai",
                model="gpt-4-turbo",
                temperature=0.3
            )
            return response
        except Exception as e:
            logger.error(f"Error generating roadmap: {str(e)}")
            raise

roadmap_service = RoadmapService()
