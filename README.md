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

### Setup Groq API

The tutorials use Groq API for LLM inference. Follow these steps to set up your environment:

1. Sign up for a Groq account at [groq.com](https://console.groq.com/signup) and get your API key
2. Create a `.env` file in the project root (or copy from `.env.template`):
   ```bash
   cp .env.template .env
   ```
3. Add your Groq API key to the `.env` file:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

The bootcamp uses two main models:
- `llama-3.3-70b-versatile` - For general tasks
- `deepseek-r1-distill-llama-70b` - For complex reasoning tasks

You can validate your API setup by running:
```bash
python test_groq_api.py
```

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