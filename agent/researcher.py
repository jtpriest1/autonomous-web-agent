# agent/researcher.py
# Orchestrates: search -> (rerank) -> fetch page -> summarize with the chosen model.

from typing import List, Tuple, Optional
import time

from models.reranker import rerank
from models.hf_summarizer import summarize_hf
from tools.web_search import web_search
from tools.fetch_page import fetch_page
from models.ollama_client import generate
from utils.logger import get_logger, log_kv

logger = get_logger("researcher")


def _summarize_page(
    query: str,
    title: str,
    url: str,
    text: str,
    model: str,
) -> str:
    """Small prompt for concise, factual bullets."""
    prompt = f"""You are a precise research assistant.
Summarize the page in 5 short bullet points (max 90 words total), focused on:
- facts relevant to the user query
- specific takeaways
- avoid fluff

User query: {query}
Page title: {title}
Page URL: {url}
Page text (truncated):
{text}

Output format:
- bullet 1
- bullet 2
- bullet 3
- bullet 4
- bullet 5
[Citation: {url}]
"""
    return generate(prompt, model=model, options={"temperature": 0.2})


def research(
    query: str,
    k: int = 3,
    model: Optional[str] = None,
    max_chars: int = 1200,
) -> str:
    """
    Run a small research task:
      1) web_search -> top k results
      2) rerank by neural relevance (titles)
      3) fetch each page (truncate to max_chars)
      4) summarize each with the chosen model
    """
    chosen_model = model or "llama3.2:3b"

    t_start = time.time()
    log_kv(logger, event="start", query=query, k=k, model=chosen_model, max_chars=max_chars)

    # 1) Search
    results: List[Tuple[str, str]] = web_search(query, max_results=k)
    log_kv(logger, event="search_ok", n_results=len(results))
    for title, url in results:
        log_kv(logger, event="candidate", title=title[:80], url=url)

    if not results:
        return f"# Research results for: {query}\n\n(No results found.)"

    # 2) Rerank (best-first) using titles as lightweight proxies.
    try:
        titles_only = [t for (t, _) in results]
        order = rerank(query, titles_only, top_k=k)  # returns list[(idx, score)]
        if order:
            results = [results[i] for (i, _score) in order]
            log_kv(logger, event="rerank_ok", order=[i for (i, _s) in order])
    except Exception as e:
        log_kv(logger, event="rerank_error", err=str(e)[:120])
        # fall back to original order

    # 3) Fetch + 4) Summarize
    sections = [f"# Research results for: {query}\n"]
    for i, (title, url) in enumerate(results, start=1):
        try:
            page = fetch_page(url, max_chars=max_chars)
            log_kv(
                logger,
                event="fetched",
                idx=i,
                url=url,
                title=(page["title"] or title)[:80],
                chars=len(page["text"]),
            )

            # HF path if model starts with "hf:", else Ollama.
            if str(chosen_model).startswith("hf:"):
                summary = summarize_hf(
                    f"{(page['title'] or title)}\n{url}\n\n{page['text']}",
                    max_words=90,
                    min_words=50,
                )
            else:
                summary = _summarize_page(
                    query=query,
                    title=page["title"] or title,
                    url=url,
                    text=page["text"],
                    model=chosen_model,
                )

            log_kv(logger, event="summarized", idx=i, url=url)
            sections.append(f"## {i}. {page['title'] or title}\n{summary}\n")

        except Exception as e:
            log_kv(logger, event="error", idx=i, url=url, err=str(e)[:120])
            sections.append(f"## {i}. {title}\n(Skipped due to error: {e})\nSource: {url}\n")

    elapsed = round(time.time() - t_start, 2)
    log_kv(logger, event="done", seconds=elapsed)

    return "\n".join(sections)


if __name__ == "__main__":
    # Quick manual test (try either Ollama or HF route):
    # - model="llama3.2:3b"
    # - model="hf:distilbart"
    print(research("practical uses of autonomous web agents in e-commerce", k=3, model="hf:distilbart", max_chars=1200))
