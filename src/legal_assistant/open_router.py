import os
import json
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

from src.legal_assistant.centralized_logger import CentralizedLogger

load_dotenv()  # Load environment variables from .env file

logger = CentralizedLogger().get_logger()


class OpenRouterLegalAssistant:
    """
    OpenRouter API client specialized for legal AI assistant operations

    Features:
    - Secure credential management
    - API-compliant request formatting
    - Error handling and validation
    - Configurable model parameters
    """

    def __init__(self):
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.api_key = os.getenv("OPENROUTER_KEY")
        self.default_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",  # Required for local dev
            "X-Title": "Legal AI Assistant v1.0",  # Max 32 chars
        }
        self.default_model = "google/gemini-2.5-pro-exp-03-25:free" #"deepseek/deepseek-r1"

    def _validate_messages(self, messages: List[Dict]) -> bool:
        """
        Validate messages structure for OpenRouter API

        Checks:
        - All messages are dictionaries
        - All messages have required 'role' and 'content' keys
        - Message roles are valid ('system', 'user', 'assistant')
        - Content values are non-empty strings
        """
        if not messages or not isinstance(messages, list):
            logger.info("Messages must be a non-empty list")
            return False

        required_keys = {"role", "content"}
        valid_roles = {"system", "user", "assistant"}

        for msg in messages:
            # Check message is a dictionary with required keys
            if not isinstance(msg, dict) or not required_keys.issubset(msg.keys()):
                logger.info(f"Invalid message format: {msg}")
                return False

            # Check role is valid
            if msg["role"] not in valid_roles:
                logger.info(
                    f"Invalid role '{msg['role']}'. Must be one of {valid_roles}"
                )
                return False

            # Check content is a non-empty string
            if not isinstance(msg["content"], str) or not msg["content"].strip():
                logger.info(f"Content must be a non-empty string")
                return False

        # Check there's at least one user message
        if not any(msg["role"] == "user" for msg in messages):
            logger.info("At least one user message is required")
            return False

        return True

    def generate_response(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
    ) -> Optional[Dict]:
        """
        Generate AI response through OpenRouter API

        Args:
            messages: List of message dictionaries in OpenAI format
            model: OpenRouter model identifier
            temperature: Creativity control (0.0-2.0)
            max_tokens: Maximum response length

        Returns:
            API response dictionary or None on failure
        """
        if not self._validate_messages(messages):
            raise ValueError("Invalid messages format")

        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                url=self.base_url,
                headers=self.default_headers,
                data=json.dumps(payload),
                timeout=600,
            )
            logger.info(f"API Response: {response.status_code} - {response.text}")
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.info(f"API Request failed: {str(e)}")
            if hasattr(e, "response") and e.response:
                logger.info(f"Error details: {e.response.text}")
            return None


# Usage Example
if __name__ == "__main__":
    assistant = OpenRouterLegalAssistant()

    test_messages = [{"role": "user", "content": "What is the meaning of life?"}]

    response = assistant.generate_response(test_messages)

    if response:
        logger.info("Legal Assistant Response:")
        logger.info(response["choices"][0]["message"]["content"])
    else:
        logger.info("Failed to get response")
