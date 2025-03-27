"""
Test script to validate Groq API connection
"""

import os
from dotenv import load_dotenv
import groq
from rich import print as pprint

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ ERROR: GROQ_API_KEY not found in environment variables")
    print("Please check your .env file")
    exit(1)

print(f"✓ API key found: {api_key[:5]}...{api_key[-4:]}")

# Initialize client
client = groq.Client(api_key=api_key)

# Test models
try:
    # List available models
    models = client.models.list()
    print("\n✓ Successfully connected to Groq API")
    print("\nAvailable models:")
    for model in models.data:
        print(f"- {model.id}")

    # Test a simple completion
    print("\nTesting simple completion...")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Say hello and confirm you're connected via Groq API",
            },
        ],
        max_tokens=100,
    )

    print("\n✓ Response received:")
    pprint(response.choices[0].message.content)

    print("\n✅ Groq API is working correctly!")

except Exception as e:
    print(f"\n❌ Error connecting to Groq API: {e}")
    print("Please check your API key and internet connection")
