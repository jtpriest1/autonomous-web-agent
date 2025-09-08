# models/hf_summarizer.py
# Lightweight Hugging Face summarizer (DistilBART) with Apple MPS support.

from functools import lru_cache
import re
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

MODEL_ID = "sshleifer/distilbart-cnn-12-6"

def _clean(text: str) -> str:
    """
    Remove lines that look like instructions (e.g., 'make a summary', 'in 2 sentences').
    """
    if not text:
        return ""
    lines = text.splitlines()
    bad = re.compile(r"(make.*summary|summarize|2-?3 sentences|write.*summary)", re.I)
    keep = [ln for ln in lines if not bad.search(ln)]
    cleaned = "\n".join(keep).strip()
    return cleaned or text  # fallback if we removed everything

@lru_cache(maxsize=1)
def _get_pipeline():
    """
    Load the tokenizer/model once, move to MPS if available, and cache the pipeline.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    if device == "mps":
        model = model.to("mps")
    return pipeline("summarization", model=model, tokenizer=tokenizer, framework="pt")

def summarize_hf(text: str, max_words: int = 90, min_words: int = 50) -> str:
    """
    Summarize text to ~max_words on the fastest available device (MPS if present).
    First call downloads the model; later calls are much faster.
    """
    text = _clean(text)
    if not text or not text.strip():
        return ""

    pipe = _get_pipeline()

    # Use lengths (BART is often happier with max_length/min_length).
    # Rough token-per-word ≈ 1.5 (very approximate).
    max_length = max(32, int(max_words * 1.5))
    min_length = max(8, int(min_words * 1.3))

    out = pipe(
        text,
        do_sample=False,              # deterministic
        truncation=True,              # clip long inputs for speed
        max_length=max_length,
        min_length=min_length,
        no_repeat_ngram_size=3,       # reduce repetition
        num_beams=4,                  # better quality
    )[0]["summary_text"].strip()

    return out
