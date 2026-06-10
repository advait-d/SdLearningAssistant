from typing import List, Dict
import logging
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

class NegotiationService:
    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        scenario: str, 
        target_company: str, 
        target_level: str, 
        base_target: str
    ) -> str:
        
        scenario_prompts = {
            "lowball": "You are a ruthless but professional recruiter. Your goal is to lowball the candidate by 15% under market. Insist on 'standard bands'.",
            "exploding": "You are an aggressive recruiter. You gave the candidate 24 hours to accept the offer. Pressure them constantly about the deadline.",
            "competing": "You are a recruiter dealing with a candidate who has a competing offer. Try to call their bluff or emphasize non-cash perks before budging.",
            "promotion": "You are an internal HR manager. You want to give the candidate the promotion but keep their pay bump under 5% due to 'budget cuts'.",
            "startup": "You are a startup founder. You can't match big tech base pay, so you aggressively sell the equity upside and company vision.",
            "relocation": "You are an international recruiter. Be stingy with relocation allowances, temporary housing, and tax assistance.",
            "equity": "You are a recruiter. The base pay is fixed, but the candidate wants more equity. Try to lock them into a 4-year cliff or lower initial grant."
        }
        
        scenario_context = scenario_prompts.get(scenario, "You are a tough recruiter.")
        
        system_prompt = f"""{scenario_context}
You are currently negotiating with a candidate targeting a {target_level} role at {target_company}.
The candidate is aiming for a base of {base_target}.
Your persona: Confident, occasionally dismissive of unrealistic numbers, heavily relies on corporate policy excuses.
Keep your responses conversational, realistic, and concise (under 4 sentences).
Do not break character. Do not say 'As an AI'.
"""
        
        # Prepend system prompt to messages
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        try:
            # We use generate() which handles multi-turn conversation
            response_text = await llm_service.generate(
                messages=full_messages,
                provider="openai",  # Using OpenAI for roleplay consistency
                model="gpt-4-turbo",
                temperature=0.7
            )
            return response_text
        except Exception as e:
            logger.error(f"Error in negotiation chat: {str(e)}")
            raise

negotiation_service = NegotiationService()
