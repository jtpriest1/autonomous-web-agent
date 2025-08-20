import requests

resp = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3.1:8b", "prompt": "Say hello and give one sentence on autonomous web agents.", "stream": False},
    timeout=120
)
resp.raise_for_status()
print("\n--- MODEL OUTPUT ---\n" + resp.json().get("response", "<no response>"))
