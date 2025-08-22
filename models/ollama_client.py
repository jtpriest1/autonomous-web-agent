# models/ollama_client.py
# Simple helper to call your local Ollama model and return text.

from typing import Any, Dict
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate(prompt: str, model: str = "llama3.1:8b", **kwargs) -> str:
    """
    Send a prompt to the local model and return its text response.
    kwargs can include: options={"temperature":0.2, ...}
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if "options" in kwargs:
        payload["options"] = kwargs["options"]

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response") or ""
        return text.strip()

    except requests.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e
