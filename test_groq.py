import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Retrieve API key and model
api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL")

# Check whether API key exists
if not api_key:
    raise ValueError("GROQ_API_KEY was not found in .env")

# Initialize Groq client using OpenAI SDK
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# Send a test request
response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful NLP research assistant."
        },
        {
            "role": "user",
            "content": "Paraphrase this sentence while preserving its sentiment: The movie was absolutely fantastic."
        }
    ],
    temperature=0.3,
    max_completion_tokens=300
)

# Print generated response
generated_text = response.choices[0].message.content

print("Groq API connection successful!")
print("Model:", model)
print("Generated response:", generated_text)