import json
import logging
from typing import Dict, Any

from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

class DrillService:
    def __init__(self):
        self.llm = llm_service

    async def generate_prompt(self, topic: str, difficulty: str = "Medium") -> Dict[str, Any]:
        prompt = f"""
You are an expert System Design interviewer giving a "Quick Drill" test.
The candidate selected the topic: {topic}. Difficulty: {difficulty}.

Provide a specific, single-question prompt that the candidate must answer verbally in under 90 seconds.
Make it punchy, practical, and focused on trade-offs or a specific mechanism.
Do NOT give away the answer. Just ask the question.
Output your response as JSON in this format:
{{
  "prompt": "Explain consistent hashing and how it handles node failure, you have 90 seconds. Go."
}}
        """
        result = await self.llm.generate(prompt, json_mode=True)
        try:
            return json.loads(result)
        except Exception:
            # fallback
            return {"prompt": f"Explain {topic} in 90 seconds. Go."}

    async def evaluate_answer(self, topic: str, prompt: str, transcript: str) -> Dict[str, Any]:
        eval_prompt = f"""
You are an expert System Design interviewer evaluating a "Quick Drill".
The topic was: {topic}
The question asked was: {prompt}
The candidate's transcribed verbal answer was:
"{transcript}"

Evaluate their answer based on accuracy, conciseness (they only had 90 seconds), and depth.
Output your evaluation as JSON in this exact format:
{{
  "score": 8,
  "feedback": "A short 2-3 sentence paragraph explaining what was good and what was missing.",
  "ideal_points": ["Point 1 they should have mentioned", "Point 2 they should have mentioned"],
  "passed": true
}}
        """
        result = await self.llm.generate(eval_prompt, json_mode=True)
        try:
            return json.loads(result)
        except Exception:
            return {
                "score": 0,
                "feedback": "Failed to parse evaluation.",
                "ideal_points": [],
                "passed": False
            }

drill_service = DrillService()
