from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import os
import asyncio
import base64
from openai import OpenAI
import json

# Initialize FastAPI
app = FastAPI()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
if not OPENAI_API_KEY or not OPENAI_BASE_URL:
    raise RuntimeError("Please set the OPENAI_API_KEY and OPENAI_BASE_URL environment variables")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

SYSTEM_PROMPT = """
You are a data analysis assistant. Analyze the provided data and questions.
Return ONLY a JSON object containing the answers. 
Do not include any markdown formatting or code blocks.
If you cannot solve a problem, return a JSON object with an 'error' key explaining why.
"""

async def read_file_content(file: UploadFile) -> Dict[str, Any]:
    """Read and process file content based on file type"""
    content = await file.read()
    await file.seek(0)  # Reset file pointer
    
    try:
        if file.filename.endswith(('.txt', '.csv')):
            return {
                'type': 'text',
                'content': content.decode('utf-8'),
                'filename': file.filename
            }
        elif file.filename.endswith(('.png', '.jpg', '.jpeg')):
            return {
                'type': 'image',
                'content': base64.b64encode(content).decode('utf-8'),
                'filename': file.filename
            }
        else:
            return {
                'type': 'unknown',
                'content': None,
                'filename': file.filename
            }
    except Exception as e:
        return {
            'type': 'error',
            'content': str(e),
            'filename': file.filename
        }

async def build_prompt(questions: str, files: List[Dict[str, Any]]) -> str:
    """Build structured prompt from questions and files"""
    prompt_parts = [f"Questions:\n{questions.strip()}"]
    
    for file in files:
        if file['type'] == 'text':
            prompt_parts.append(f"\nContents of {file['filename']}:\n{file['content']}")
        elif file['type'] == 'image':
            prompt_parts.append(f"\nImage file included: {file['filename']} (base64 encoded)")
        else:
            prompt_parts.append(f"\nFile included: {file['filename']} (unprocessed)")
    
    return "\n".join(prompt_parts)

async def call_openai_with_retry(prompt: str, max_retries: int = 3) -> str:
    """Call OpenAI API with retry logic"""
    for attempt in range(max_retries):
        try:
            async with asyncio.timeout(180):  # 3 minute timeout
                response = await client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                raise HTTPException(status_code=504, detail="Request timeout")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=str(e))
            await asyncio.sleep(2 ** attempt)

@app.post("/api")
async def analyze(
    questions_file: UploadFile = File(..., alias="questions.txt"),
    extra_files: List[UploadFile] = File(default=[])
):
    """Process analysis request and return JSON response"""
    try:
        # Read questions
        questions = (await questions_file.read()).decode('utf-8')
        
        # Process all files
        file_contents = [
            await read_file_content(f) for f in extra_files
        ]
        
        # Build prompt
        prompt = await build_prompt(questions, file_contents)
        
        # Get OpenAI response
        response_text = await call_openai_with_retry(prompt)
        
        try:
            # Parse response as JSON
            result = json.loads(response_text)
            return result  # FastAPI will automatically convert dict to JSON
        except json.JSONDecodeError:
            # If response isn't valid JSON, return error
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Invalid JSON response from AI model",
                    "raw_response": response_text
                }
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

