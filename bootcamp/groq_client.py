from typing import Dict, List, Optional, Union

import os
import groq
from dotenv import load_dotenv
from loguru import logger

from bootcamp.config import Config

# Load environment variables from .env file
load_dotenv()

# Groq model IDs
GENERAL_MODEL = Config.Model.LLAMA_3_3_70B
COMPLEX_MODEL = Config.Model.DEEPSEEK_R1_DISTILL
TEMPERATURE = 0

# Initialize Groq client with API key from environment variable
client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))


def chat(
    model: str = GENERAL_MODEL,
    messages: List[Dict[str, str]] = None,
    max_tokens: int = 4096,
    temperature: float = TEMPERATURE,
    keep_alive: Optional[Union[str, int]] = None,  # Not used with Groq API
    tools: Optional[List[Dict]] = None,
    format: Optional[Dict] = None,
    options: Optional[Dict] = None,  # Not used with Groq API
):
    """
    Call the Groq API with the given parameters.
    This function provides a similar interface to the previous Ollama function for easy migration.
    """
    if messages is None:
        messages = []

    # Handle format param (used for structured output in Ollama)
    response_format = None
    if format:
        response_format = {"type": "json_object"}

    # Process tools param for function calling
    function_call = None
    functions = None
    if tools:
        functions = tools
        function_call = "auto"

    try:
        # Determine which model to use based on input complexity
        selected_model = model

        # Make the API call
        response = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            tools=functions,
            tool_choice=function_call if function_call else None,
        )

        # Convert Groq response to match expected format from Ollama
        adapted_response = AdaptedGroqResponse(
            response=response,
            model=model,
        )

        return adapted_response

    except Exception as e:
        logger.error(f"Error calling Groq API: {e}")
        raise


class AdaptedGroqResponse:
    """
    Adapter class to make Groq responses compatible with the existing Ollama-based code.
    """

    def __init__(self, response, model):
        self.response = response
        self.model = model
        self.message = AdaptedMessage(self.response.choices[0].message)


class AdaptedMessage:
    """
    Adapter for Groq message to match Ollama message format.
    """

    def __init__(self, message):
        self.content = message.content
        self.role = message.role

        # Handle tool calls
        self.tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            self.tool_calls = [
                AdaptedToolCall(tool_call) for tool_call in message.tool_calls
            ]


class AdaptedToolCall:
    """
    Adapter for Groq tool calls to match Ollama tool calls format.
    """

    def __init__(self, tool_call):
        self.id = tool_call.id
        self.type = tool_call.type
        self.function = AdaptedFunction(tool_call.function)


class AdaptedFunction:
    """
    Adapter for Groq function to match Ollama function format.
    """

    def __init__(self, function):
        self.name = function.name
        self.arguments = function.arguments
