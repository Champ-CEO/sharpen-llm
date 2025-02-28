from typing import Any, Dict, List, Literal

import requests
from ollama import chat
from pydantic import BaseModel, Field
from rich import print as pprint
from rich.console import Console
from rich.markdown import Markdown

from bootcamp.build_ai_products.ai_toolkit.question import QuizQuestion

MODEL = "qwen2.5"
TEMPERATURE = 0

TRIVIA_API_URL = "https://opentdb.com/api.php"
TRIVIA_CATEGORY_IDS = {
    "computers": 18,
    "vehicles": 28,
}


class FetchTriviaParameters(BaseModel):
    count: int = Field(description="The number of quiz questions to fetch.")
    category: Literal["computers", "vehicles"] = Field(
        description="The category of quiz questions. Allowed values are 'computers' and 'vehicles'."
    )


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "fetch_trivia_questions",
        "description": "Retrieve a list of quiz questions from the Open Trivia Database API based on the provided count and category.",
        "parameters": FetchTriviaParameters.model_json_schema(),
    },
}


def fetch_trivia_questions(
    count: int, category: str = Literal["computers", "vehicles"]
) -> List[QuizQuestion]:
    params = {
        "amount": count,
        "type": "multiple",
        "category": TRIVIA_CATEGORY_IDS[category],
    }

    response = requests.get(TRIVIA_API_URL, params=params)
    response.raise_for_status()

    json_data = response.json()
    questions = [QuizQuestion(**item) for item in json_data["results"]]
    return questions


def execute_tool_call(prompt: str) -> List[Dict[str, Any]]:
    available_tools = {"fetch_trivia_questions": fetch_trivia_questions}

    messages = [{"role": "user", "content": prompt}]

    response = chat(
        model=MODEL,
        messages=messages,
        keep_alive=-1,
        options={"temperature": TEMPERATURE},
        tools=[TOOL_SPEC],
    )

    message = response.message
    pprint(message)

    tool = message.tool_calls[0]
    tool_to_call = available_tools[tool.function.name]
    tool_result = tool_to_call(**tool.function.arguments)
    pprint(tool_result)

    messages.append(message)
    messages.append(
        {"role": "tool", "content": str(tool_result), "name": tool.function.name}
    )

    return messages


def display_final_response(messages: List[Dict[str, Any]]):
    final_response = chat(
        model=MODEL,
        messages=messages,
        keep_alive=-1,
        options={"temperature": TEMPERATURE},
    )

    console = Console()
    md = Markdown(final_response.message.content)
    console.print(md)


prompt = "Get me 3 quiz questions about computers. Format the output as markdown."
messages = execute_tool_call(prompt)
display_final_response(messages)
