from bootcamp.groq_client import chat
from rich import print as pprint
from bootcamp.config import Config

from bootcamp.build_ai_products.ai_toolkit.question import QuizQuestion

MODEL = Config.Model.LLAMA_3_3_70B
TEMPERATURE = 0


QUESTION = """
What did the name of the Tor Anonymity Network orignially stand for?

Correct answer: The Onion Router
Incorrect answers: Onion routing, The Omni Router, Garlic routing
""".strip()

PROMPT = """
Extract the question and answers from the text and format them as a JSON object.

<text>
{text}
</text>
"""

messages = [
    {
        "role": "user",
        "content": PROMPT.format(text=QUESTION),
    }
]

response = chat(
    model=MODEL,
    messages=messages,
    temperature=TEMPERATURE,
    format=QuizQuestion.model_json_schema(),
)

question = QuizQuestion.model_validate_json(response.message.content)
pprint(question)
