# models/hf_summarizer.py
# Lightweight Hugging Face summarizer (DistilBART) with Apple MPS support.

from functools import lru_cache
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

MODEL_ID = "sshleifer/distilbart-cnn-12-6"

@lru_cache(maxsize=1)
def _get_pipeline():
    """
    Load the tokenizer/model once, move to MPS if available, and cache the pipeline.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    if device == "mps":
        model = model.to("mps")  # run on Apple GPU

    # pipeline will use model.device automatically
    return pipeline("summarization", model=model, tokenizer=tokenizer, framework="pt")

def summarize_hf(text: str, max_words: int = 90, min_words: int = 50) -> str:
    """
    Summarize text to ~max_words on the fastest available device (MPS if present).
    Notes:
      - First call will download the model (~300MB) into your HF cache.
      - We truncate long inputs to keep it snappy.
    """
    if not text or not text.strip():
        return ""

    pipe = _get_pipeline()

    # Rough token estimates (BART ≈ 1.3–1.6 tokens/word). Keep it simple:
    max_new_tokens = max(32, int(max_words * 1.6))
    min_new_tokens = max(8, int(min_words * 1.3))

    out = pipe(
        text,
        do_sample=False,          # deterministic
        truncation=True,          # clip overly long inputs for speed
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
    )[0]["summary_text"].strip()

    return out
