# app.py — Streamlit UI for the web agent

import re
import streamlit as st
from agent.researcher import research

st.set_page_config(
    page_title="Autonomous Web Agent — Local",
    page_icon="🕸️",
    layout="wide",
)

# cache the slow bit; 10 min is fine
@st.cache_data(show_spinner=False, ttl=600)
def run_research_cached(query: str, k: int, model: str, max_chars: int, use_reranker: bool) -> str:
    return research(query=query, k=k, model=model, max_chars=max_chars, use_reranker=use_reranker)

def split_sections(md: str):
    # take the markdown blob and carve out cards
    md = md.strip()
    if not md:
        return "Results", []
    lines = md.splitlines()
    header = lines[0].replace("# ", "").strip() if lines and lines[0].startswith("# ") else "Results"

    parts = md.split("\n## ")
    cards = []
    for part in parts[1:]:
        block = part.strip()
        if not block:
            continue
        block_lines = block.splitlines()
        title_line = block_lines[0].strip()
        body = "\n".join(block_lines[1:]).strip()

        m = re.search(r"\[Citation:\s*(.*?)\s*\]", body)
        url = m.group(1) if m else None
        cards.append({"title": title_line, "url": url, "body": body})
    return header, cards

# --- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    # label -> internal id
    options = {
        "Llama 3.2 — 3B (fast)": "llama3.2:3b",
        "Llama 3.1 — 8B (deeper)": "llama3.1:8b",
        "HF DistilBART (MPS) — fast summarizer": "hf:distilbart",
    }
    choice = st.selectbox("Model", list(options.keys()), index=0)
    model = options[choice]

    max_chars = st.number_input(
        "Max characters per page",
        min_value=500, max_value=4000, step=100, value=1200,
        help="Lower = faster; higher = more context sent to the model.",
    )

    use_reranker = st.toggle(
        "Use neural reranker (MiniLM)",
        value=True,
        help="Reorders search hits with sentence-transformers; small extra RAM + a bit of time."
    )

    st.caption("Tip: First run of **HF DistilBART** downloads ~1.2 GB once, then it’s fast on Apple M-series (MPS).")

st.markdown("[GitHub →](https://github.com/jtpriest1/autonomous-web-agent)")

# --- header ----------------------------------------------------------------
st.title("🕸️ Autonomous Web Agent — Local")
st.write(
    "Ask a question. I’ll search, fetch pages, and summarize locally "
    "(Ollama or Hugging Face on MPS). No cloud keys needed."
)

# --- main form -------------------------------------------------------------
with st.form("query_form", clear_on_submit=False):
    query = st.text_input("Your research question", placeholder="e.g., What are the best tech companies to work for in 2025?")
    k = st.slider("How many results to summarize", min_value=1, max_value=5, value=3)
    submitted = st.form_submit_button("Run research", use_container_width=True)

if submitted:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Working… searching, fetching, summarizing…"):
            md = run_research_cached(
                query=query.strip(),
                k=int(k),
                model=model,
                max_chars=int(max_chars),
                use_reranker=use_reranker,
            )

        header, cards = split_sections(md)
        st.subheader(header)

        if not cards:
            st.info("No sections returned.")
        else:
            for i, c in enumerate(cards, start=1):
                with st.container(border=True):
                    st.markdown(f"#### {i}. {c['title']}")
                    if c["url"]:
                        st.caption(f"🔗 Source: [{c['url']}]({c['url']})")
                    st.markdown(c["body"])