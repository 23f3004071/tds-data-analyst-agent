import requests
import os

# Change this if running on a server or different port
BASE_URL = "https://tds-data-analyst-agent-murex.vercel.app/api"

# Required file
files = [
    ("questions.txt", ("questions.txt", open("questions.txt", "rb"), "text/plain")),
]

# Add data.csv only if it exists
if os.path.exists("data.csv"):
    files.append(
        ("extra_files", ("data.csv", open("data.csv", "rb"), "text/csv"))
    )

# Send POST request
response = requests.post(BASE_URL, files=files)

# Print response
print("Status Code:", response.status_code)
try:
    print("JSON:", response.json())

except Exception:
    print("Text:", response.text)
