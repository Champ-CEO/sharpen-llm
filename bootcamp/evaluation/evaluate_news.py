import opik
from opik import Opik
from opik.evaluation import evaluate_prompt, models

from bootcamp.config import Config, configure_logging, seed_everything
from bootcamp.evaluation.metrics import AccuracyMetric

opik.configure(use_local=True)

seed_everything()
configure_logging()

client = Opik()
dataset = client.get_or_create_dataset(Config.Dataset.CRYPTO_NEWS)


evaluate_prompt(
    project_name="crypto-news",
    experiment_name="sentiment-accuracy",
    dataset=dataset,
    messages=[
        {
            "role": "user",
            "content": client.get_prompt(Config.Prompt.CLASSIFY_ARTICLE).prompt,
        },
    ],
    scoring_metrics=[AccuracyMetric(name="sentiment_accuracy", field="sentiment")],
    model=models.LiteLLMChatModel(
        model_name=f"ollama/{Config.Model.DEEPSEEK_R1}",
        temperature=0,
    ),
)
