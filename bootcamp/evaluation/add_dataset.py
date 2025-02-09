import ast

import opik
import pandas as pd
from opik import Opik

from bootcamp.config import Config, configure_logging, seed_everything

opik.configure(use_local=True)

seed_everything()
configure_logging()

N_CRYPTO_NEWS = 100
N_EMAIL_SUMMARIES = 100

client = Opik()

ARTICLE_FORMAT = """
<article>
    <title>{title}</title>
    <text>{text}</text>
</article>
""".strip()

EMAIL_FORMAT = """
<email>
{text}
</email>
""".strip()


def create_crypto_data() -> pd.DataFrame:
    df = pd.read_parquet(Config.Path.DATA_DIR / "crypto-news.parquet")
    sentiment_df = pd.DataFrame(list(df["sentiment"].apply(ast.literal_eval).values))
    return pd.concat([df.drop("sentiment", axis=1), sentiment_df], axis=1)


def add_crypto_news_dataset(client: Opik):
    client.delete_dataset(Config.Dataset.CRYPTO_NEWS)
    dataset = client.get_or_create_dataset(Config.Dataset.CRYPTO_NEWS)

    df = create_crypto_data().sample(N_CRYPTO_NEWS)

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "input": ARTICLE_FORMAT.format(title=row.title, text=row.text),
                "expected_output": {
                    "sentiment": row["class"],
                    "polarity": row["polarity"],
                    "subjectivity": row["subjectivity"],
                },
            }
        )
    dataset.insert(rows)


def create_email_summaries_data() -> pd.DataFrame:
    return pd.read_parquet(Config.Path.DATA_DIR / "email-summaries.parquet")


def add_email_summaries_dataset(client: Opik):
    client.delete_dataset(Config.Dataset.EMAIL_SUMMARIES)
    dataset = client.get_or_create_dataset(Config.Dataset.EMAIL_SUMMARIES)
    df = create_email_summaries_data().sample(N_EMAIL_SUMMARIES)
    rows = [{"input": EMAIL_FORMAT.format(text=row.email)} for _, row in df.iterrows()]
    dataset.insert(rows)


add_crypto_news_dataset(client)
add_email_summaries_dataset(client)
