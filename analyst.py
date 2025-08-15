from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import os
import asyncio
import base64
import json
import pandas as pd
import pdfplumber
import pytesseract
from PIL import Image
from io import BytesIO
from openai import OpenAI
import re

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
if not OPENAI_API_KEY or not OPENAI_BASE_URL:
    raise RuntimeError("Missing OPENAI_API_KEY or OPENAI_BASE_URL")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

AI_TIMEOUT = 20
SYSTEM_PROMPT = """
You are a versatile data analysis assistant. 
You will be given:
1. Questions in plain text
2. Summarized descriptions of provided data files

Rules:
- Answer every question directly in JSON format
- If data is insufficient, set the value to null
- Do not output explanations or extra text, only the JSON object
- Ensure JSON is valid and parseable
"""

def summarize_file(file_data: bytes, filename: str) -> str:
    """Summarize file content without sending raw data to AI."""
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(BytesIO(file_data))
            summary = {
                "filename": filename,
                "columns": df.columns.tolist(),
                "head": df.head(3).to_dict(),
                "shape": df.shape
            }
            return json.dumps(summary)
        elif filename.endswith(".pdf"):
            with pdfplumber.open(BytesIO(file_data)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages[:2])
            return f"{filename} (first 2 pages text): {text[:1000]}"
        elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(BytesIO(file_data))
            text = pytesseract.image_to_string(img)
            return f"{filename} (OCR text): {text[:500]}"
        else:
            text = file_data.decode(errors="ignore")
            return f"{filename} (text excerpt): {text[:1000]}"
    except Exception as e:
        return f"{filename} could not be processed: {e}"

async def call_openai(prompt: str) -> Dict[str, Any]:
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
            text = re.sub(r"```json\s*|\s*```", "", text)
            return json.loads(text)
    except Exception as e:
        print(f"AI failed: {e}")
        return {}


@app.get("/")
async def root():
    return {"message": "Welcome to the Data Analysis API. Give POST at /api for analysis."}


@app.post("/api")
async def analyze(
    questions_file: UploadFile = File(...),
    extra_files: List[UploadFile] = File(default=[])
):
    try:
        # Read and summarize files
        questions_text = (await questions_file.read()).decode(errors="ignore")
        file_summaries = []
        for f in extra_files:
            content = await f.read()
            summary = summarize_file(content, f.filename)
            file_summaries.append(summary)

        # Build compact prompt
        prompt = f"Questions:\n{questions_text.strip()}\n\nData summaries:\n" + "\n".join(file_summaries)

        # Call AI
        ai_answers = await call_openai(prompt)

        # Always return valid JSON
        if not isinstance(ai_answers, dict):
            ai_answers = {}

        return JSONResponse(content=ai_answers)

    except Exception as e:
        print(f"Fatal error: {e}")
        return JSONResponse(content={}, status_code=500)
