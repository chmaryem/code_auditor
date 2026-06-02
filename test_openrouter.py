import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m2.5:free")

print("MODEL =", model)
print("KEY FOUND =", bool(api_key))

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",
    "X-Title": "Code Auditor Test",
}

payload = {
    "model": model,
    "messages": [
        {"role": "user", "content": "Reply only with OK"}
    ],
    "max_tokens": 20,
    "temperature": 0,
}

response = requests.post(url, headers=headers, json=payload, timeout=(15, 120))

print("STATUS:", response.status_code)
print(response.text)