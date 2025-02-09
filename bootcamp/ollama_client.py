from loguru import logger
from ollama import chat

from bootcamp.config import configure_logging, seed_everything

seed_everything()
configure_logging()

MODEL = "llama3.2"
TEMPERATURE = 0

PROMPT = """
You are a UX writer specializing in clear, actionable error messages.

Write a payment failure error message in 2 parts:

- What happened (max 10 words)
- What to do (max 15 words)

The result should be a single error message that is 25 words or less. Format it as a JSON object.
""".strip()


response = chat(
    model=MODEL,
    messages=[{"role": "user", "content": PROMPT}],
    options={"temperature": TEMPERATURE},
    format="json",
)


logger.debug(f"Response\n{response.message.content}")
