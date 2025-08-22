# models/ollama_client.py
# Call your local Ollama model and return text. Choose model per call.

from typing import Any, Dict, Optional
import os
import requests

# You can override these via env vars if you want.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# Fast default for your Mac; change to "llama3.1:8b" if you prefer.
DEFAULT_MODEL = os.getenv("AGENT_MODEL", "llama3.2:3b")
QUALITY_MODEL = "llama3.1:8b"

# Sensible speed/quality defaults; callers can override with `options=...`
BASE_OPTIONS: Dict[str, Any] = {
    "temperature": 0.2,
    "num_predict": 160,   # keep summaries short => faster
    "num_ctx": 2048,
    "keep_alive": "5m",   # keep model in memory between calls
}

def _post_ollama(payload: Dict[str, Any], timeout: int) -> str:
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()

def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    *,
    options: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> str:
    """Send a prompt to Ollama and return plain text. Falls back to 8B on error."""
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {**BASE_OPTIONS, **(options or {})},
    }

    try:
        return _post_ollama(payload, timeout)
    except requests.RequestException:
        # Simple safety net: if fast model fails, try the quality model once.
        if model != QUALITY_MODEL:
            payload["model"] = QUALITY_MODEL
            try:
                return _post_ollama(payload, timeout)
            except requests.RequestException as e:
                raise RuntimeError(f"Ollama request failed (fallback too): {e}") from e
        raise  # already on quality model

# Convenience wrappers if you want to call directly:
def generate_fast(prompt: str, **kw) -> str:
    return generate(prompt, model=DEFAULT_MODEL, **kw)

def generate_quality(prompt: str, **kw) -> str:
    return generate(prompt, model=QUALITY_MODEL, **kw)
