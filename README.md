# Autonomous Web Agent (local, open-source stack)

A small but real agent that runs **fully local**:
- **Model:** Llama 3.1 (8B) via **Ollama**
- **Tools:** DuckDuckGo search (`ddgs`) + HTML fetch/clean (BeautifulSoup)
- **Orchestrator:** `researcher.research()` = search → fetch → summarize (with citations)

## Why this project
- Show **systems thinking**: model wrapper, tools, orchestration, logging.
- **Privacy & cost:** no external LLM API; all on your machine.
- **Recruiter-friendly:** 1-command run, clear architecture, roadmap.

---

## Quickstart

### 0) Requirements
- macOS, Python 3.12+, Homebrew
- Ollama installed and running
- Pulled model `llama3.1:8b`

```bash
# install/start Ollama (macOS)
brew install ollama
brew services start ollama
ollama pull llama3.1:8b
