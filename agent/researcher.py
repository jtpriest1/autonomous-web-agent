# agent/researcher.py
# Orchestrates: search -> fetch page -> summarize with the local model.

from typing import List, Tuple
from tools.web_search import web_search
from tools.fetch_page import fetch_page
from models.ollama_client import generate
from utils.logger import get_logger, log_kv
import time

logger = get_logger("researcher")


def _summarize_page(query: str, title: str, url: str, text: str) -> str:
    """
    Ask the local LLM for a short, useful summary tied to the page.
    Keep it tight so it runs fast on your hardware. Never return None.
    """
    prompt = f"""You are a precise research assistant.
Summarize the page in 5 short bullet points (max 90 words total), focused on:
- facts relevant to the user query
- specific takeaways
- avoid fluff
- if the page has little info, still produce bullets using only what's given

User query: {query}
Page title: {title}
Page URL: {url}
Page text (truncated):
{text}

Output format (exactly):
- bullet 1
- bullet 2
- bullet 3
- bullet 4
- bullet 5
[Citation: {url}]
"""
    out = generate(prompt, options={"temperature": 0.2})
    out = (out or "").strip()
    if not out:
        out = (
            "- (no usable content found)\n"
            "- (page may be blocked or thin)\n"
            "- (try a different source)\n"
            "- \n"
            "- \n"
            f"[Citation: {url}]"
        )
    return out


def research(query: str, k: int = 3) -> str:
    """
    Run a small research task:
    1) web_search -> top k results
    2) fetch each page (short)
    3) summarize each with the model
    Returns one printable string with sections.
    """
    # start timing + log the run beginning
    t_start = time.time()
    log_kv(logger, event="start", query=query, k=k)

    # Search
    results: List[Tuple[str, str]] = web_search(query, max_results=k)
    # log how many results + list each candidate URL
    log_kv(logger, event="search_ok", n_results=len(results))
    for title, url in results:
        log_kv(logger, event="candidate", title=title[:80], url=url)

    if not results:
        return f"# Research results for: {query}\n\n(No results found.)"

    sections = [f"# Research results for: {query}\n"]
    for i, (title, url) in enumerate(results, start=1):
        try:
            page = fetch_page(url, max_chars=1800)
            # log after fetching the page
            log_kv(logger, event="fetched", idx=i, url=url, title=(page["title"] or title)[:80], chars=len(page["text"]))

            summary = _summarize_page(query, page["title"] or title, url, page["text"])
            # log after summarizing
            log_kv(logger, event="summarized", idx=i, url=url)

            sections.append(f"## {i}. {page['title'] or title}\n{summary}\n")
        except Exception as e:
            # log errors too
            log_kv(logger, event="error", idx=i, url=url, err=str(e)[:120])
            sections.append(f"## {i}. {title}\n(Skipped due to error: {e})\nSource: {url}\n")

    # log how long the whole run took
    elapsed = round(time.time() - t_start, 2)
    log_kv(logger, event="done", seconds=elapsed)

    return "\n".join(sections)


if __name__ == "__main__":
    print(research("practical uses of autonomous web agents in e-commerce", k=3))
