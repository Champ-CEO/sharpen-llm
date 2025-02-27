import json
from typing import Optional

from ollama import chat
from rich import print as pprint

from bootcamp.build_ai_products.ai_toolkit.question import QuizQuestion

MODEL = "qwen2.5"
TEMPERATURE = 0

QUESTIONS_DATABASE = """
[
  {
    "question":"What was the first Android version specifically optimized for tablets?",
    "correct_answer":"Honeycomb",
    "incorrect_answers":[
      "Eclair",
      "Froyo",
      "Marshmellow"
    ]
  },
  {
    "question":"On which computer hardware device is the BIOS chip located?",
    "correct_answer":"Motherboard",
    "incorrect_answers":[
      "Hard Disk Drive",
      "Central Processing Unit",
      "Graphics Processing Unit"
    ]
  },
  {
    "question":"Which of the following is a personal computer made by the Japanese company Fujitsu?",
    "correct_answer":"FM-7",
    "incorrect_answers":[
      "PC-9801",
      "Xmillennium ",
      "MSX"
    ]
  },
  {
    "question":"What is the domain name for the country Tuvalu?",
    "correct_answer":".tv",
    "incorrect_answers":[
      ".tu",
      ".tt",
      ".tl"
    ]
  },
  {
    "question":"Which of these people was NOT a founder of Apple Inc?",
    "correct_answer":"Jonathan Ive",
    "incorrect_answers":[
      "Steve Jobs",
      "Ronald Wayne",
      "Steve Wozniak"
    ]
  }
]
"""

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "search_questions",
        "description": "Find a question in the database based on the provided question text",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question text to search for",
                },
            },
            "required": ["question"],
        },
    },
}


def search_questions(question: str) -> Optional[QuizQuestion]:
    questions = json.loads(QUESTIONS_DATABASE)
    for item in questions:
        if question.lower() in item["question"].lower():
            return QuizQuestion(**item)
    return None


def retrieve_answer_with_tools(prompt: str) -> str:
    available_tools = {"search_questions": search_questions}

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    tool_selection_response = chat(
        model=MODEL,
        messages=messages,
        keep_alive=-1,
        options={"temperature": TEMPERATURE},
        tools=[TOOL_SPEC],
    )

    message = tool_selection_response.message
    messages.append(message)

    assert len(message.tool_calls) == 1

    tool = message.tool_calls[0]
    tool_to_call = available_tools[tool.function.name]
    tool_result = tool_to_call(**tool.function.arguments)

    messages.append(
        {"role": "tool", "content": str(tool_result), "name": tool.function.name}
    )

    final_response = chat(
        model=MODEL,
        messages=messages,
        keep_alive=-1,
        options={"temperature": TEMPERATURE},
    )

    return final_response.message.content


result = search_questions("What is the domain name for the country Tuvalu?")
pprint("Direct search result:")
pprint(result)


query = (
    "What is the domain name for the country Tuvalu? Find the answer from the database."
)
answer = retrieve_answer_with_tools(query)
pprint("LLM retrieval result:")
pprint(answer)
