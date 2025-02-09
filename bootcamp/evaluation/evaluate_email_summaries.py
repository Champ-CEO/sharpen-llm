import opik
from opik import Opik
from opik.evaluation import evaluate_prompt, models
from opik.evaluation.metrics import Hallucination

from bootcamp.config import Config, configure_logging, seed_everything

opik.configure(use_local=True)

seed_everything()
configure_logging()

client = Opik()
dataset = client.get_or_create_dataset(Config.Dataset.EMAIL_SUMMARIES)


evaluate_prompt(
    project_name="email-summaries",
    experiment_name="summarize-email",
    dataset=dataset,
    messages=[
        {
            "role": "user",
            "content": client.get_prompt(Config.Prompt.SUMMARIZE_EMAIL).prompt,
        },
    ],
    scoring_metrics=[
        Hallucination(
            name="summary_hallucination",
            model=models.LiteLLMChatModel(
                model_name=f"ollama/{Config.Model.JUDGE_LLM}", temperature=0
            ),
        ),
    ],
    model=models.LiteLLMChatModel(
        model_name=f"ollama/{Config.Model.QWEN}",
        temperature=0,
    ),
)
