import requests

# Change this if running on a server or different port
BASE_URL = "http://127.0.0.1:8000/api"

# Required file
files = [
    ("questions.txt", ("questions.txt", open("questions.txt", "rb"), "text/plain")),
]

# Optional extra files
optional_files = [
    ("extra_files", ("data.csv", open("data.csv", "rb"), "text/csv")),
]

# Merge required + optional
files.extend(optional_files)

# Send POST request
response = requests.post(BASE_URL, files=files)

# Print response
print("Status Code:", response.status_code)
try:
    print("JSON:", response.json())

except Exception:
    print("Text:", response.text)
