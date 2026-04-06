import os
import json
import logging
from typing import Dict, Any, Optional
from openai import AsyncOpenAI, OpenAIError

# Configure logger for this service
logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for interacting with OpenAI Chat Completions API.
    Handles standard chat responses, structured data parsing, and error handling.
    """
    def __init__(self, api_key: Optional[str] = None):
        # Initialize the async OpenAI client using the provided key or environment variable
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    async def generate_response(
        self, 
        system_prompt: str, 
        user_input: str, 
        model: str = "gpt-4-turbo", 
        temperature: float = 0.7,
        json_mode: bool = False
    ) -> str:
        """
        Calls the OpenAI API with a system prompt and user input.
        
        Args:
            system_prompt: The system instructions setting the context and rules.
            user_input: The actual query or task from the user.
            model: The OpenAI model to use (default: gpt-4-turbo).
            temperature: Controls randomness (0.0 = deterministic, 1.0 = highly creative).
            json_mode: If True, forces the model to return valid JSON.
            
        Returns:
            The raw string response from the model.
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
            
            # Use json_object response format if requested
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format=response_format
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Received empty response from the model.")
                
            return content
            
        except OpenAIError as e:
            logger.error(f"OpenAI API Error: {str(e)}")
            # Raise a more generic exception to be handled by the FastAPI exception handlers
            raise RuntimeError(f"Failed to communicate with the LLM provider: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in LLM Service: {str(e)}")
            raise RuntimeError(f"An unexpected error occurred: {str(e)}")

    async def generate_structured_response(
        self, 
        system_prompt: str, 
        user_input: str, 
        model: str = "gpt-4-turbo", 
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Generates a structured JSON response and parses it into a Python dictionary.
        Highly useful for Confidence Scoring or formatted Design Reviews.
        
        Note: The system prompt MUST explicitly tell the model to output JSON 
        for this to work reliably.
        
        Args:
            system_prompt: System instructions (must mention JSON output).
            user_input: The query or task.
            model: The OpenAI model.
            temperature: Defaults to 0.0 for more predictable structured outputs.
            
        Returns:
            A dictionary containing the parsed JSON structure.
        """
        try:
            result_text = await self.generate_response(
                system_prompt=system_prompt,
                user_input=user_input,
                model=model,
                temperature=temperature,
                json_mode=True
            )
            
            # Parse the text into a dictionary
            parsed_json = json.loads(result_text)
            return parsed_json
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured output. Raw response: {result_text}")
            raise ValueError(f"Invalid JSON response from LLM: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating structured response: {str(e)}")
            raise

# Create a singleton instance for easy import across the app
llm_service = LLMService()
