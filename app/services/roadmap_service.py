from typing import List, Dict, Any, Optional
import logging
import io
from pypdf import PdfReader
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

ROADMAP_SYSTEM_PROMPT = """You are a Staff Software Engineer coach with 15 years of experience helping engineers land roles at top-tier tech companies (Google, Meta, Amazon, Apple, Microsoft, Stripe, Airbnb, etc.).

You have been given a candidate's resume. Your job is to:
1. Identify their exact technical gaps relative to the target role
2. Assess their current level vs. the target level
3. Build a REALISTIC, day-by-day study sprint given their timeline

## Assessment Rules
- Be **brutally honest** about gaps. If they lack distributed systems experience, say so.
- Prioritize by **interview probability** — what will DEFINITELY come up in the onsite?
- Do NOT pad the schedule with generic topics. Every day must target a specific identified weakness.
- Use PRECISE topics (e.g. "Consistent hashing & virtual nodes", NOT just "distributed systems").
- For each day, provide a concrete action (what to read/build/practice).

## Topic Priority for Software Engineering Roles
High priority (must cover): System design fundamentals, data modeling, API design, concurrency, caching strategies, database internals, distributed consensus
Medium priority: Kubernetes/container orchestration, CI/CD, observability/SLOs, cost optimization
Lower priority (only if time): Language-specific internals, niche algorithms

## Output Format
Return ONLY valid JSON — no prose, no markdown fences, no comments:
{
  "level_gap": "<e.g. 'Currently L4, targeting L5 — missing cross-team scope and system design depth'>",
  "critical_gaps": [
    "<specific gap with evidence from resume>",
    "<specific gap with evidence from resume>"
  ],
  "overview": "<2-3 sentence aggressive coach briefing — what needs to happen and why>",
  "projected_hours": <integer total study hours>,
  "weeks": [
    {
      "week_number": 1,
      "focus": "<single sentence theme for this week>",
      "days": [
        {
          "day": 1,
          "topic": "<specific topic>",
          "action": "<specific resource or exercise — e.g. 'Read: Designing Data-Intensive Applications Ch.5. Then design a Kafka-backed event sourcing system on paper.'>",
          "hours": <integer 1-4>
        }
      ]
    }
  ]
}"""


class RoadmapService:
    async def generate_roadmap_from_pdf(
        self,
        target_role: str,
        timeline_days: int,
        pdf_bytes: Optional[bytes] = None,
        resume_text: Optional[str] = None,
    ) -> Dict[str, Any]:

        # Resolve text from either source
        text = resume_text or ""
        if not text and pdf_bytes:
            try:
                reader = PdfReader(io.BytesIO(pdf_bytes))
                pages = []
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        pages.append(extracted.strip())
                text = "\n\n".join(pages)
            except Exception as e:
                logger.error(f"Error parsing PDF for roadmap: {str(e)}")
                raise ValueError("Could not parse the uploaded PDF file.")

        if not text.strip():
            raise ValueError("No resume text provided or extracted.")

        user_input = f"""Target Role: {target_role}
Timeline: {timeline_days} days until the interview

Candidate Resume:
{text[:5500]}

Generate a realistic, gap-targeted study roadmap for exactly {timeline_days} days ({timeline_days // 7} full weeks + {timeline_days % 7} extra days). Each week should have exactly 7 days (or fewer for the final partial week)."""

        try:
            response = await llm_service.generate_structured_response(
                system_prompt=ROADMAP_SYSTEM_PROMPT,
                user_input=user_input,
                provider="gemini",
                model="gemini-2.5-flash",
                temperature=0.25,
            )
            return response
        except Exception as e:
            logger.error(f"Error generating roadmap: {str(e)}")
            raise


roadmap_service = RoadmapService()
