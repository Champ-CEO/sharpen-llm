# MLExpert.io Bootcamp

Materials for the "Get Things Done with AI" bootcamp. Currently for the v2 edition.

## Install

Make sure you have [`uv` installed](https://docs.astral.sh/uv/getting-started/installation/).

Clone the repository (feel free to rename the folder):

```bash
git clone git@github.com:mlexpertio/bootcamp.git .
cd bootcamp
```

Install Python:

```bash
uv python install 3.12.8
```

Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Install dependencies:

```bash
uv sync
```

Install package in editable mode:

```bash
uv pip install -e .
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

### Run Ollama

Most of the tutorials use Ollama for LLM inference (to run models locally). Watch this video to see how to install Ollama: https://www.youtube.com/watch?v=lmFCVCqOlz8

To get a model for the tutorials, run (example for Qwen 2.5):

```bash
ollama pull qwen2.5
```

Feel free to experiment with other models.

## Datasets

- [Cryptocurrency news `/data/crypto-news.parquet`](https://huggingface.co/datasets/NickyNicky/crypto-news-small)
- [Email summaries `data/email-summaries.parquet`](https://huggingface.co/datasets/argilla/FinePersonas-Conversations-Email-Summaries)

## Opik

```sh
cd opik/deployment/docker-compose
```

Start

```sh
docker compose up --detach
```

Stop

```sh
docker compose down
```