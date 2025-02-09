import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel

from bootcamp.config import configure_logging, seed_everything

seed_everything()
configure_logging()

load_dotenv()

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_COMPLETION_TOKENS = 128

client = OpenAI()

PROMPT = """
You are a UX writer specializing in clear, actionable error messages.

Write a payment failure error message in 2 parts:

- What happened (max 10 words)
- What to do (max 15 words)

The result should be a single error message that is 25 words or less. Format it as a JSON object.
""".strip()


response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": PROMPT,
        }
    ],
    max_completion_tokens=MAX_COMPLETION_TOKENS,
    temperature=TEMPERATURE,
    model=MODEL,
)

logger.debug(f"Response message\n{response.choices[0].message.content}")

usage = response.usage


df = pd.DataFrame(
    {
        "prompt_tokens": [usage.prompt_tokens],
        "completion_tokens": [usage.completion_tokens],
        "total_tokens": [usage.total_tokens],
    }
)
logger.debug(f"Usage:\n{df.to_markdown(index=None)}")


class ErrorMessage(BaseModel):
    type: str
    message: str


response = client.beta.chat.completions.parse(
    messages=[
        {
            "role": "user",
            "content": PROMPT,
        }
    ],
    max_completion_tokens=MAX_COMPLETION_TOKENS,
    temperature=TEMPERATURE,
    model=MODEL,
    response_format=ErrorMessage,
)

response_object = response.choices[0].message.parsed
logger.debug(f"Parsed response object:\n{response_object}")
