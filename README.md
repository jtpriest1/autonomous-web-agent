# Autonomous Web Agent — Local, OSS

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black)
![HF Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![Apple M-series](https://img.shields.io/badge/Apple-MPS%20accelerated-lightgrey)

A small but real autonomous web-research agent that runs fully local.

- **Models**: Ollama (Llama 3.2 3B fast • Llama 3.1 8B deeper) + **HF DistilBART** summarizer on MPS  
- **Tools**: DuckDuckGo search (`ddgs`), HTML fetch + clean (Requests + BeautifulSoup)  
- **Brains**: Neural **reranker** (sentence-transformers MiniLM), per-step logs, simple eval harness  
- **Privacy**: No cloud keys. Everything runs on your machine.  
- **Docker**: 1-image run that talks to your host’s Ollama.

---

## Quickstart

### Local (venv)
```bash
# 0) Requirements: macOS (Apple Silicon works great), Python 3.12+, Homebrew
# 1) Ollama
brew install ollama
brew services start ollama
ollama pull llama3.2:3b
# (Optional) deeper model:
# ollama pull llama3.1:8b

# 2) Repo
git clone https://github.com/jtpriest1/autonomous-web-agent.git
cd autonomous-web-agent

# 3) Python env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4) Run
streamlit run app.py
# open http://localhost:8501 and choose a model in the sidebar
