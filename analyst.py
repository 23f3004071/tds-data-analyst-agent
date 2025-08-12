import os
import asyncio
import httpx
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List
from openai import OpenAI

# Initialize FastAPI
app = FastAPI()

# Get API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are a data analyst agent. You will receive questions and optionally files. "
    "You will write Python code in your head, run it mentally, and give the results "
    "in JSON exactly as requested."
)

async def build_prompt(questions: str, files: List[UploadFile]) -> str:
    """
    Build the user prompt from the question file and optional file contents.
    Compress file contents if large.
    """
    prompt_parts = [questions.strip()]
    if files:
        for file in files:
            try:
                content = (await file.read()).decode("utf-8", errors="ignore")
                if len(content) > 2000:
                    content = content[:2000] + "...[truncated]"
                prompt_parts.append(f"\n\n--- File: {file.filename} ---\n{content}")
            except Exception as e:
                prompt_parts.append(f"\n\n--- File: {file.filename} ---\n[Error reading file: {e}]")
    return "\n".join(prompt_parts)

async def call_openai_with_retry(system_prompt: str, user_prompt: str, retries: int = 4, timeout: int = 300):
    """
    Call OpenAI API with retry and timeout.
    """
    for attempt in range(1, retries + 1):
        try:
            async with asyncio.timeout(timeout):
                response = client.responses.create(
                    model="gpt-4.1",
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.output_text
        except Exception as e:
            if attempt == retries:
                return f"[]  # Failed after {retries} retries: {e}"
            await asyncio.sleep(1)  # brief wait before retry

@app.post("/analyze")
async def analyze(
    questions: UploadFile = File(...),
    files: List[UploadFile] = File(default=[])
):
    """
    Main API endpoint for analysis.
    """
    try:
        # Read questions.txt content
        questions_text = (await questions.read()).decode("utf-8", errors="ignore")

        # Build the full user prompt
        user_prompt = await build_prompt(questions_text, files)

        # Call OpenAI API with retries and timeout
        result_text = await call_openai_with_retry(SYSTEM_PROMPT, user_prompt)

        # Ensure valid JSON is returned (even if incorrect answers)
        try:
            import json
            parsed = json.loads(result_text)
            return JSONResponse(content=parsed)
        except Exception:
            # If model output is not valid JSON, wrap it
            return JSONResponse(content=[result_text])

    except Exception as e:
        return JSONResponse(content=[f"Error: {e}"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
