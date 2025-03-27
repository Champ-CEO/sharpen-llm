import groq
from bootcamp.config import Config

# Model configuration
MODEL = Config.Model.LLAMA_3_3_70B  # "llama-3.3-70b-versatile"
COMPLEX_MODEL = Config.Model.DEEPSEEK_R1_DISTILL  # "deepseek-r1-distill-llama-70b"
TEMPERATURE = 0

client = groq.Client()  # Make sure to set your GROQ_API_KEY as an environment variable


def call_model(prompt: str, complex_task: bool = False) -> str:
    """
    Call the Groq API with the given prompt.
    Set complex_task=True to use the more powerful deepseek model for complex reasoning tasks.
    """
    selected_model = COMPLEX_MODEL if complex_task else MODEL

    response = client.chat.completions.create(
        model=selected_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )

    return response.choices[0].message.content


# Example usage
if __name__ == "__main__":
    prompt = "Write an error message for when a user's payment fails."
    response = call_model(prompt)
    print(response)

    # Example with the more powerful model for complex tasks
    complex_prompt = """
    Explain the mathematical intuition behind why gradient descent works for finding 
    local minima in non-convex optimization problems.
    """
    complex_response = call_model(complex_prompt, complex_task=True)
    print(complex_response)
