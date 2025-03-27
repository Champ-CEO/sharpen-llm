import os
import random
import sys
from pathlib import Path

import numpy as np
from loguru import logger


class Config:
    SEED = 42

    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATA_DIR = APP_HOME / "data"

    class Model:
        JUDGE_LLM = "llama-3.3-70b-versatile"  # Updated to Groq model
        QWEN = "llama-3.3-70b-versatile"  # Updated to Groq model
        DEEPSEEK_R1 = "deepseek-r1-distill-llama-70b"  # Updated to Groq model

        # New Groq model IDs
        LLAMA_3_3_70B = "llama-3.3-70b-versatile"
        DEEPSEEK_R1_DISTILL = "deepseek-r1-distill-llama-70b"

    class Dataset:
        CRYPTO_NEWS = "crypto-news"
        EMAIL_SUMMARIES = "email-summaries"

    class Prompt:
        CLASSIFY_ARTICLE = "classify-article"
        CLASSIFY_ARTICLE_FOCUS_SENTIMENT = "classify-article-focus-sentiment"
        SUMMARIZE_EMAIL = "summarize-email"


def seed_everything(seed: int = Config.SEED):
    random.seed(seed)
    np.random.seed(seed)


def configure_logging():
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "colorize": True,
                "format": "<green>{time:YYYY-MM-DD - HH:mm:ss}</green> | <level>{level}</level> | {message}",
            },
        ]
    }
    logger.configure(**config)
