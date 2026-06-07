import os
import json
import logging
from typing import Dict, Any, Optional
from openai import AsyncOpenAI, OpenAIError, RateLimitError

try:
    from google import genai
    from google.genai import types
    has_genai = True
except ImportError:
    has_genai = False

# Configure logger for this service
logger = logging.getLogger(__name__)


class LLMOverloadedError(RuntimeError):
    """Raised when the LLM provider is temporarily overloaded (HTTP 503 / 429)."""
    pass

class LLMService:
    """
    Service for interacting with OpenAI and Google Gemini Chat APIs.
    Handles standard chat responses, structured data parsing, and error handling.
    """
    def __init__(self, api_key: Optional[str] = None):
        # Initialize the async OpenAI client using the provided key or environment variable
        self.openai_client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        # Initialize Gemini Client if available
        self.gemini_client = None
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if has_genai and gemini_api_key:
            self.gemini_client = genai.Client(api_key=gemini_api_key)

    async def generate_response(
        self, 
        system_prompt: str, 
        user_input: str, 
        provider: str = "openai",
        model: Optional[str] = None, 
        temperature: float = 0.7,
        json_mode: bool = False
    ) -> str:
        """
        Calls the selected LLM API with a system prompt and user input.
        """
        if provider == "gemini":
            return await self._generate_gemini(
                system_prompt, user_input, model or "gemini-2.5-flash", temperature, json_mode
            )
        else:
            return await self._generate_openai(
                system_prompt, user_input, model or "gpt-4-turbo", temperature, json_mode
            )

    async def _generate_openai(
        self, system_prompt: str, user_input: str, model: str, temperature: float, json_mode: bool
    ) -> str:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}
            
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format=response_format
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Received empty response from the model.")
            return content
            
        except RateLimitError as e:
            logger.error(f"OpenAI Rate Limit / Overloaded: {str(e)}")
            raise LLMOverloadedError(f"OpenAI is currently overloaded. Please try again shortly.")
        except OpenAIError as e:
            logger.error(f"OpenAI API Error: {str(e)}")
            raise RuntimeError(f"Failed to communicate with OpenAI: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in LLM Service: {str(e)}")
            raise RuntimeError(f"An unexpected error occurred: {str(e)}")

    async def _generate_gemini(
        self, system_prompt: str, user_input: str, model: str, temperature: float, json_mode: bool
    ) -> str:
        if not self.gemini_client:
            raise RuntimeError("Gemini API is not configured (missing GEMINI_API_KEY or google-genai).")
            
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                response_mime_type="application/json" if json_mode else "text/plain"
            )
            
            response = await self.gemini_client.aio.models.generate_content(
                model=model,
                contents=user_input,
                config=config
            )
            
            content = response.text
            if not content:
                raise ValueError("Received empty response from the Gemini model.")
            return content
            
        except Exception as e:
            err_str = str(e)
            logger.error(f"Gemini API Error: {err_str}")
            # Detect 503 UNAVAILABLE — model is temporarily overloaded
            if "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str.lower():
                raise LLMOverloadedError(
                    "The AI model is currently experiencing high demand. Please try again in a moment."
                )
            raise RuntimeError(f"Failed to communicate with Gemini: {err_str}")

    async def generate_structured_response(
        self, 
        system_prompt: str, 
        user_input: str, 
        provider: str = "openai",
        model: Optional[str] = None, 
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Generates a structured JSON response and parses it into a Python dictionary.
        """
        try:
            result_text = await self.generate_response(
                system_prompt=system_prompt,
                user_input=user_input,
                provider=provider,
                model=model,
                temperature=temperature,
                json_mode=True
            )
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
