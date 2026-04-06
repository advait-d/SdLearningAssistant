import os
import logging
from typing import Tuple, Dict, Any
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

class EvaluatorService:
    """
    Evaluates AI responses to ensure quality, extract confidence scores, 
    detect hallucinations, and trigger fallback mechanisms when necessary.
    """

    def __init__(self):
        # Configuration for evaluation rules
        self.confidence_threshold = 0.6
        
        # Phrases that often indicate the model is hallucinating, guessing, or lacking context
        self.hallucination_signals = [
            "i don't know",
            "i am not sure",
            "i lack information",
            "i do not have enough information",
            "as an ai",
            "there is no context provided",
            "it is not mentioned"
        ]
        
        # Resolve prompt directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.prompts_dir = os.path.join(base_dir, "prompts")

    def _read_prompt(self, filename: str) -> str:
        """Helper to read a prompt template file."""
        filepath = os.path.join(self.prompts_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    async def evaluate_response(
        self, 
        user_request: str, 
        context: str, 
        ai_response: str
    ) -> Tuple[bool, str, float]:
        """
        Evaluates the generated response.
        
        Returns:
            Tuple[bool, str, float]: 
                - is_fallback: True if the evaluator triggered a fallback.
                - final_response: Either the original valid response or the generated fallback.
                - confidence_score: The calculated score (0.0 to 1.0).
        """
        # 1. Check for heuristic hallucination signals in the text
        lower_response = ai_response.lower()
        has_vague_signals = any(signal in lower_response for signal in self.hallucination_signals)
        
        if has_vague_signals:
            logger.info("Heuristic hallucination or uncertainty signals detected in response.")

        # 2. Extract Confidence Score via LLM
        score = 1.0
        reason = "Passed heuristic checks"
        
        try:
            eval_template = self._read_prompt("confidence_scoring.txt")
            
            # Format the prompt with inputs
            system_prompt = eval_template.format(
                user_request=user_request,
                context=context,
                ai_response=ai_response
            )
            
            # Ask the LLM to score the response (JSON mode, temperature=0.0)
            eval_result = await llm_service.generate_structured_response(
                system_prompt=system_prompt,
                user_input="Please evaluate the response and return the confidence score JSON.",
                temperature=0.0
            )
            
            score = float(eval_result.get("score", 1.0))
            reason = eval_result.get("reason", "No justification provided.")
            logger.info(f"Evaluator Score: {score} | Reason: {reason}")
            
        except Exception as e:
            logger.error(f"Failed to extract confidence score: {str(e)}")
            # If scoring fails, we fall back to heuristics
            if has_vague_signals:
                score = 0.4 

        # 3. Decision Rules: Should we trigger a fallback?
        # Rule A: Score is explicitly below the threshold
        # Rule B: Strong heuristic signals were found, and score is borderline (< 0.8)
        needs_fallback = (score < self.confidence_threshold) or (has_vague_signals and score < 0.8)

        if needs_fallback:
            logger.warning(f"Response rejected (Score: {score}). Triggering fallback.")
            fallback_msg = await self._trigger_fallback(user_request, context)
            return True, fallback_msg, score

        # Response passed evaluation
        return False, ai_response, score

    async def _trigger_fallback(self, user_request: str, context: str) -> str:
        """
        Generates a fallback response asking for clarification or suggesting to narrow the query.
        """
        try:
            fallback_template = self._read_prompt("fallback_clarification.txt")
            
            system_prompt = fallback_template.format(
                context=context,
                user_request=user_request
            )
            
            # Generate the clarification questions
            fallback_response = await llm_service.generate_response(
                system_prompt=system_prompt,
                user_input="Generate 2-3 clarification questions.",
                temperature=0.4 # slight temperature for conversational naturalness
            )
            return fallback_response
            
        except Exception as e:
            logger.error(f"Failed to generate dynamic fallback: {str(e)}")
            # Hardcoded failsafe fallback
            return (
                "I do not have enough confidence or context to provide an accurate system design for this request. "
                "Could you please narrow down your query or provide more specific architectural constraints?"
            )

# Singleton instance for use across routes/services
evaluator_service = EvaluatorService()
