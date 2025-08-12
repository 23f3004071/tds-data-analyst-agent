import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

if not api_key or not base_url:
    raise RuntimeError("Environment variables OPENAI_API_KEY or OPENAI_BASE_URL not set.")

client = OpenAI(api_key=api_key, base_url=base_url)

try:
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input="Say 'Hello from AI Pipe!'"
    )
    print("✅ Connection successful!")
    print("Response:", resp.output_text)
except Exception as e:
    print("❌ Connection failed:", e)
