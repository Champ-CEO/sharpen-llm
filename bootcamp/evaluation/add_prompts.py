import opik
from opik import Opik

from bootcamp.config import Config, configure_logging, seed_everything

opik.configure(use_local=True)

seed_everything()
configure_logging()


client = Opik()

CLASSIFY_ARTICLE_PROMPT = """
Classify the following news article:

- sentiment (positive, neutral, negative)
- subjectivity (float)
- polarity (float)

{{input}}

The response must be in JSON. Here's an example:
```
{
    "sentiment": "positive",
    "subjectivity": 0.5,
    "polarity": -0.1
}
```
Reply only with the JSON object.
""".strip()

client.create_prompt(Config.Prompt.CLASSIFY_ARTICLE, CLASSIFY_ARTICLE_PROMPT)

CLASSIFY_ARTICLE_FOCUS_SENTIMENT_PROMPT = """
Classify the following news article:

- sentiment (positive, neutral, negative)
- subjectivity (float)
- polarity (float)

{{input}}

The response must be in JSON. Here's an example:
```
{
    "sentiment": "positive",
    "subjectivity": 0.5,
    "polarity": -0.1
}
```
Focus on correctly finding the sentiment. This is the most important part.
Reply only with the JSON object.
""".strip()

client.create_prompt(
    Config.Prompt.CLASSIFY_ARTICLE_FOCUS_SENTIMENT,
    CLASSIFY_ARTICLE_FOCUS_SENTIMENT_PROMPT,
)


SUMMARIZE_EMAIL_PROMPT = """
Summarize the following email:

{{input}}

Focus only on the most important points and keep it short.
Reply only with the text of the summary.
"""


client.create_prompt(
    Config.Prompt.SUMMARIZE_EMAIL,
    SUMMARIZE_EMAIL_PROMPT,
)
