from bootcamp.groq_client import chat
from bootcamp.config import Config

MODEL = Config.Model.LLAMA_3_3_70B
TEMPERATURE = 0


messages = [
    {
        "role": "user",
        "content": "What did the name of the Tor Anonymity Network orignially stand for?",
    },
    {
        "role": "assistant",
        "content": "Onion routing",
    },
    {
        "role": "user",
        "content": "Think deeply and try again. Reply only with the full name.",
    },
]

response = chat(
    model=MODEL,
    messages=messages,
    temperature=TEMPERATURE,
)


print(response.message.content)
