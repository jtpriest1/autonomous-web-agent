# tools/fetch_page.py

import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (Macintosh) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"}

def fetch_page(url: str, max_chars: int = 4000) -> dict:
    """Return {title, url, text} from a web page."""
    r = requests.get(url, headers=UA, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = (soup.title.string if soup.title else "").strip()
    text = " ".join(s.strip() for s in soup.get_text(" ").split())
    return {"url": url, "title": title[:200], "text": text[:max_chars]}
