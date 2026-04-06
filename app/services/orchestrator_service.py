import os
import logging
from typing import Dict, Any

from app.services.intent_service import intent_service
from app.services.retriever_service import retriever_service
from app.services.memory_service import get_history, update_history
from app.services.llm_service import llm_service
from app.services.evaluator_service import evaluator_service

logger = logging.getLogger(__name__)

class OrchestratorService:
    """
    Main orchestration pipeline that coordinates user intent, context retrieval,
    LLM generation, and evaluation/fallbacks.
    """
    
    def __init__(self):
        # Resolve prompt directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.prompts_dir = os.path.join(base_dir, "prompts")

    def _read_prompt(self, filename: str) -> str:
        filepath = os.path.join(self.prompts_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def _format_history(self, history: list) -> str:
        """Helper to convert the message dictionary list into a readable string."""
        if not history:
            return "No previous conversation history."
            
        formatted_history = ""
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {msg['content']}\n\n"
        return formatted_history

    async def handle_request(self, session_id: str, user_query: str) -> Dict[str, Any]:
        """
        Main pipeline to handle a user's incoming chat request.
        
        Steps:
        1. Classify Intent
        2. Reject if out of scope
        3. Retrieve relevant RAG context
        4. Load conversation history
        5. Build system prompt based on intent
        6. Call the LLM
        7. Evaluate the response for confidence/hallucinations
        8. Update memory with final answer
        """
        logger.info(f"Processing request for session '{session_id}' | Query: '{user_query}'")

        # 1. Classify Intent
        intent = await intent_service.classify_intent(user_query)
        logger.info(f"Classified Intent: {intent}")

        # 2. Reject OUT_OF_SCOPE early
        if intent == "OUT_OF_SCOPE":
            msg = "I'm sorry, but I can only assist with software architecture and system design topics. How can I help you with those today?"
            return {"response": msg, "intent": intent, "score": None}

        # 3. Retrieve Context (RAG)
        context = await retriever_service.retrieve_context(user_query)

        # 4. Get Conversation History
        raw_history = get_history(session_id)
        formatted_history = self._format_history(raw_history)

        # 5. Select and Format the Prompt
        if intent == "DESIGN_REVIEW":
            template_content = self._read_prompt("design_review.txt")
            # For review, we map user_query to proposed_design
            system_prompt = template_content.format(
                context=context,
                proposed_design=user_query
            )
        else: # Standard SYSTEM_DESIGN advice
            template_content = self._read_prompt("system_prompt.txt")
            system_prompt = template_content.format(
                context=context,
                history=formatted_history,
                user_request=user_query
            )

        # 6. Call LLM for Initial Generation
        try:
            initial_response = await llm_service.generate_response(
                system_prompt=system_prompt,
                user_input=user_query,
                temperature=0.7
            )
        except Exception as e:
            logger.error(f"LLM Generation failed: {str(e)}")
            return {
                "response": "An error occurred while generating a system design response. Please try again later.",
                "intent": intent,
                "score": 0.0
            }

        # 7. Evaluate and potentially trigger Fallback
        is_fallback, final_response, confidence_score = await evaluator_service.evaluate_response(
            user_request=user_query,
            context=context,
            ai_response=initial_response
        )

        if is_fallback:
            logger.info("Pipeline returned a fallback clarification due to low confidence.")

        # 8. Update Session Memory
        update_history(
            session_id=session_id, 
            user_msg=user_query, 
            assistant_msg=final_response
        )

        # 9. Return final payload
        return {
            "response": final_response,
            "intent": intent,
            "score": confidence_score,
            "is_fallback": is_fallback
        }

# Singleton instance
orchestrator = OrchestratorService()