import os

import opik
from loguru import logger
from openai import OpenAI
from opik.integrations.openai import track_openai

from bootcamp.config import configure_logging, seed_everything

opik.configure(use_local=True)

seed_everything()
configure_logging()


os.environ["OPIK_PROJECT_NAME"] = "tracing"

client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama",
)

client = track_openai(client)


MODEL = "llama3.2"
TEMPERATURE = 0

PROMPT = """
You are a UX writer specializing in clear, actionable error messages.

Write a payment failure error message in 2 parts:

- What happened (max 10 words)
- What to do (max 15 words)

The result should be a single error message that is 25 words or less. Format it as a JSON object.
""".strip()

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": PROMPT,
        }
    ],
    temperature=TEMPERATURE,
    response_format={"type": "json_object"},
    model=MODEL,
)

logger.debug(chat_completion.choices[0].message.content)
