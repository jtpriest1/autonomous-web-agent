# models/reranker.py
# Tiny neural reranker using sentence-transformers (MiniLM).

from __future__ import annotations
from typing import List, Tuple
import torch

try:
    from sentence_transformers import SentenceTransformer, util
except Exception as e:
    raise RuntimeError(
        "sentence-transformers is missing. Install it in your venv:\n"
        "pip install sentence-transformers"
    ) from e

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None
_device = "mps" if torch.backends.mps.is_available() else "cpu"


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME, device=_device)
    return _model


def embed(texts: List[str]) -> torch.Tensor:
    """Return L2-normalized embeddings [N, D]."""
    model = _get_model()
    embs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, device=_device)
    return embs


def rerank(query: str, docs: List[str], top_k: int | None = None) -> List[Tuple[int, float]]:
    """
    Return [(idx, score)] sorted by score desc.
    - query: user question
    - docs: list of doc strings (title + snippet)
    - top_k: if set, truncate to that many
    """
    if not docs:
        return []

    q = embed([query])            # [1, D]
    d = embed(docs)               # [N, D]
    sims = util.cos_sim(q, d).squeeze(0)  # [N]
    pairs = [(i, float(sims[i].item())) for i in range(len(docs))]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k] if top_k else pairs


if __name__ == "__main__":
    demo_docs = [
        "Open-source web agents list and comparisons.",
        "Neural ranking with MiniLM for better search quality.",
        "Cooking tips for summer BBQ.",
    ]
    order = rerank("how to improve a web agent with reranking", demo_docs)
    print(order)
