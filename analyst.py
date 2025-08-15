from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import List, Dict, Any
import os
import asyncio
import base64
import json
import pandas as pd
import pdfplumber
from io import BytesIO
from openai import AsyncOpenAI
import re

# FastAPI app
app = FastAPI()

# Env vars
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
if not OPENAI_API_KEY or not OPENAI_BASE_URL:
    raise RuntimeError("Missing OPENAI_API_KEY or OPENAI_BASE_URL")

client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# Config
AI_TIMEOUT = 20  # seconds for each AI call
AI_MAX_RETRIES = 2  # number of extra retries
SYSTEM_PROMPT = """
You are a versatile data analysis assistant.
You will receive:
1. Questions in plain text
2. Short summaries of provided data files

Rules:
- Return answers ONLY as a valid JSON object
- Keys should be short descriptions of each question
- Values should be the direct answer or null if data is insufficient
- No explanations, markdown, or extra text outside JSON
- If a calculation is required, compute it logically
"""

@app.get("/")
async def home():
    return PlainTextResponse("Welcome to the Data Analysis API. Use POST /api with 'questions.txt' and optional files.")

def summarize_file(file_data: bytes, filename: str) -> str:
    """Summarize file contents for AI without sending raw big data."""
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(file_data))
            return json.dumps({
                "filename": filename,
                "columns": df.columns.tolist(),
                "shape": df.shape,
                "sample_rows": df.head(3).to_dict()
            })
        elif filename.lower().endswith(".pdf"):
            with pdfplumber.open(BytesIO(file_data)) as pdf:
                text = "\n".join((page.extract_text() or "") for page in pdf.pages[:1])
            return f"{filename} (first page text): {text[:800]}"
        elif filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            b64 = base64.b64encode(file_data).decode("utf-8")[:200]  # snippet
            return f"{filename} (image base64 start): {b64}..."
        else:
            text = file_data.decode(errors="ignore")
            return f"{filename} (text excerpt): {text[:800]}"
    except Exception as e:
        return f"{filename} could not be processed: {e}"

async def call_openai(prompt: str) -> Dict[str, Any]:
    """Call AI with retries, strict timeout & safe JSON parse."""
    for attempt in range(AI_MAX_RETRIES + 1):
        try:
            async with asyncio.timeout(AI_TIMEOUT):
                resp = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    timeout=AI_TIMEOUT,
                    max_retries=0
                )
                text = resp.choices[0].message.content.strip()
                text = re.sub(r"```json\s*|\s*```", "", text)  # strip markdown
                return json.loads(text)
        except Exception as e:
            print(f"AI attempt {attempt+1} failed: {e}")
            await asyncio.sleep(1)  # short delay before retry
    return {}  # fallback if all retries fail

@app.post("/api")
async def analyze(
    questions_file: UploadFile = File(...),
    extra_files: List[UploadFile] = File(default=[])
):
    try:
        # Read questions
        questions_text = (await questions_file.read()).decode(errors="ignore")

        # Summarize extra files
        file_summaries = []
        for f in extra_files:
            content = await f.read()
            file_summaries.append(summarize_file(content, f.filename))

        # Build AI prompt
        prompt = f"Questions:\n{questions_text.strip()}\n\nData summaries:\n" + "\n".join(file_summaries)

        # Get AI answers
        ai_answers = await call_openai(prompt)

        # Ensure valid JSON object
        if not isinstance(ai_answers, dict):
            ai_answers = {}

        return JSONResponse(content=ai_answers)

    except Exception as e:
        print(f"Fatal error: {e}")
        return JSONResponse(content={}, status_code=500)
