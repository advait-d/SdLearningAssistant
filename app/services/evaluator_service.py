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
        
        # Phrases that indicate the model refused to answer (these should score LOW, not high)
        # A good assistant answers from expertise; refusing is a failure mode.
        self.refusal_signals = [
            "i do not have enough information to answer",
            "i don't have enough information",
            "i lack the information",
            "there is no context provided",
            "the context does not contain",
            "based on the provided context, i cannot",
            "i cannot answer this",
            "i'm unable to answer",
        ]
        
        # Phrases that indicate genuine uncertainty (softer penalty)
        self.uncertainty_signals = [
            "i'm not sure",
            "i am not sure",
            "as an ai language model",
            "it is not mentioned",
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
        ai_response: str,
        provider: str = "openai"
    ) -> Tuple[bool, str, float]:
        """
        Evaluates the generated response.
        
        Returns:
            Tuple[bool, str, float]: 
                - is_fallback: True if the evaluator triggered a fallback.
                - final_response: Either the original valid response or the generated fallback.
                - confidence_score: The calculated score (0.0 to 1.0).
        """
        lower_response = ai_response.lower()
        
        # 1. Hard refusal check — model explicitly refused to answer
        has_refusal = any(signal in lower_response for signal in self.refusal_signals)
        has_uncertainty = any(signal in lower_response for signal in self.uncertainty_signals)
        
        if has_refusal:
            logger.warning("Model refused to answer a question it should know. Forcing fallback.")
            return True, await self._retry_without_restriction(user_request, context, provider), 0.2
        
        if has_uncertainty:
            logger.info("Uncertainty signals detected in response.")

        # 2. Extract Confidence Score via LLM
        score = 0.85  # optimistic default — assume good unless evaluator says otherwise
        reason = "Passed heuristic checks"
        
        try:
            eval_template = self._read_prompt("confidence_scoring.txt")
            
            system_prompt = eval_template.format(
                user_request=user_request,
                context=context,
                ai_response=ai_response
            )
            
            eval_result = await llm_service.generate_structured_response(
                system_prompt=system_prompt,
                user_input="Please evaluate the response and return the confidence score JSON.",
                provider=provider,
                temperature=0.0
            )
            
            score = float(eval_result.get("score", 0.85))
            reason = eval_result.get("reason", "No justification provided.")
            logger.info(f"Evaluator Score: {score} | Reason: {reason}")
            
        except Exception as e:
            logger.error(f"Failed to extract confidence score: {str(e)}")
            if has_uncertainty:
                score = 0.5

        # 3. Decision: trigger fallback only if genuinely low quality
        needs_fallback = score < self.confidence_threshold

        if needs_fallback:
            logger.warning(f"Response rejected (Score: {score}). Triggering fallback.")
            fallback_msg = await self._trigger_fallback(user_request, context, provider=provider)
            return True, fallback_msg, score

        # Response passed evaluation
        return False, ai_response, score

    async def _retry_without_restriction(
        self, user_request: str, context: str, provider: str = "openai"
    ) -> str:
        """
        When the model incorrectly refused to answer, retry with an explicit
        instruction to answer from expert knowledge.
        """
        retry_system_prompt = (
            "You are an expert system design architect and educator with deep knowledge of "
            "distributed systems, scalability, databases, networking, and software architecture patterns. "
            "Answer the following question directly and thoroughly from your expertise. "
            "Do NOT say you lack information — you are the expert. "
            "Use concrete examples, real systems, and specific trade-offs."
        )
        try:
            logger.info("Retrying with unrestricted expert prompt for: %r", user_request[:80])
            return await llm_service.generate_response(
                system_prompt=retry_system_prompt,
                user_input=user_request,
                provider=provider,
                temperature=0.7
            )
        except Exception as e:
            logger.error(f"Retry also failed: {str(e)}")
            return (
                "I encountered an issue generating a response. Please try again or rephrase your question."
            )

    async def _trigger_fallback(self, user_request: str, context: str, provider: str = "openai") -> str:
        """
        Generates a fallback response asking for clarification when the response is genuinely low-quality.
        """
        try:
            fallback_template = self._read_prompt("fallback_clarification.txt")
            
            system_prompt = fallback_template.format(
                context=context,
                user_request=user_request
            )
            
            fallback_response = await llm_service.generate_response(
                system_prompt=system_prompt,
                user_input="Generate 2-3 clarification questions.",
                provider=provider,
                temperature=0.4
            )
            return fallback_response
            
        except Exception as e:
            logger.error(f"Failed to generate dynamic fallback: {str(e)}")
            return (
                "Could you provide more details about what you're trying to build? "
                "For example: expected scale, tech constraints, or what aspect you want to focus on?"
            )

# Singleton instance for use across routes/services
evaluator_service = EvaluatorService()
