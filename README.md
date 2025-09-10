# Autonomous Web Agent (local, open-source)

A small but real web-research agent that runs **fully local**:
- **Models:** Llama 3.2-3B (fast) & Llama 3.1-8B (deeper) via **Ollama**, or **HF DistilBART** on Apple M-series (MPS)
- **Tools:** DuckDuckGo search (`ddgs`), HTML fetch/clean (BeautifulSoup), **neural reranker** (MiniLM)
- **UI:** Streamlit
- **Flow:** search → **rerank** → fetch → summarize → cite

## Why
- Show **systems thinking**: model wrapper, tools, orchestration, logging, eval.
- **Privacy & cost:** no external LLM API; all on your machine.
- **1-command run** + clear architecture for recruiters.

---

## Quickstart (local)

**Prereqs**
- macOS, Python 3.12+
- [Ollama](https://ollama.com) running
- First-time model pulls:
```bash
brew install ollama
brew services start ollama
ollama pull llama3.2:3b
# optional:
ollama pull llama3.1:8b
