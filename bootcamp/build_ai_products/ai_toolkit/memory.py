from ollama import chat

MODEL = "qwen2.5"
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
    keep_alive=-1,
    options={"temperature": TEMPERATURE},
)


print(response.message.content)
