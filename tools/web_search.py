# tools/web_search.py
# Simple, reliable web search using the duckduckgo-search library.

from typing import List, Tuple
from ddgs import DDGS

def web_search(query: str, max_results: int = 5) -> list[tuple[str, str]]:
    """
    Returns top results as a list of (title, url) tuples.
    """
    results: list[tuple[str, str]] = []
    # DDGS handles the scraping/search under the hood.
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            title = (r.get("title") or "").strip()
            url = r.get("href")
            if url:
                results.append((title, url))
    return results

if __name__ == "__main__":
    print(web_search("autonomous web agents", 3))