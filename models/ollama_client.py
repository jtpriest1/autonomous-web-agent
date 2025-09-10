# models/ollama_client.py
# tiny client for Ollama. pick model per call.

from typing import Any, Dict, Optional
import os
import requests

# url: prefer explicit OLLAMA_URL; otherwise build from host/port
_host = os.getenv("OLLAMA_HOST", "localhost")
_port = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_URL = os.getenv("OLLAMA_URL", f"http://{_host}:{_port}/api/generate")

# defaults you can override
DEFAULT_MODEL = os.getenv("AGENT_MODEL", "llama3.2:3b")
QUALITY_MODEL = "llama3.1:8b"

BASE_OPTIONS: Dict[str, Any] = {
    "temperature": 0.2,
    "num_predict": 160,   # short answers = fast
    "num_ctx": 2048,
    "keep_alive": "5m",
}

def _post_ollama(payload: Dict[str, Any], timeout: int) -> str:
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    *,
    options: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> str:
    """plain text out; will try the bigger model once if the first call fails"""
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {**BASE_OPTIONS, **(options or {})},
    }
    try:
        return _post_ollama(payload, timeout)
    except requests.RequestException:
        if model != QUALITY_MODEL:
            payload["model"] = QUALITY_MODEL
            try:
                return _post_ollama(payload, timeout)
            except requests.RequestException as e:
                raise RuntimeError(f"Ollama request failed (fallback too): {e}") from e
        raise

# small helpers if you like these names
def generate_fast(prompt: str, **kw) -> str:
    return generate(prompt, model=DEFAULT_MODEL, **kw)

def generate_quality(prompt: str, **kw) -> str:
    return generate(prompt, model=QUALITY_MODEL, **kw)
