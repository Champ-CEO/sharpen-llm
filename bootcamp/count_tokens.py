import pandas as pd
import tiktoken
from loguru import logger
from tiktoken import Encoding

from bootcamp.config import configure_logging, seed_everything

seed_everything()
configure_logging()

MODEL = "gpt-4o-mini"


encoder = tiktoken.encoding_for_model(MODEL)


def show_encoding_comparison(text: str, encoder: Encoding):
    encoding = encoder.encode(text)

    word_count = len(text.split())
    token_count = len(encoding)

    logger.debug(f"Text:\n{text}")
    logger.debug(f"Encoding:\n\n{encoding[:12]}\n")

    df = pd.DataFrame(
        {
            "word_count": [word_count],
            "token_count": [token_count],
            "increase": [round(token_count / word_count, 2)],
        }
    )
    logger.debug(f"Statistics:\n\n{df.to_markdown(index=None)}\n")


prompt = """
You are a UX writer specializing in clear, actionable error messages.

Write a payment failure error message in 2 parts:

- What happened (max 10 words)
- What to do (max 15 words)
""".strip()


show_encoding_comparison(prompt, encoder)


response = """
{
  "type": "payment_failure",
  "message": "Your payment method has expired. Please update your card details or use a different payment method to complete the transaction."
}
""".strip()


show_encoding_comparison(response, encoder)
