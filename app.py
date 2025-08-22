# app.py
import streamlit as st
from agent.researcher import research

st.set_page_config(page_title="Autonomous Web Agent (Local)", layout="wide")

st.title("😄 Autonomous Web Agent — Local (Ollama + OSS)")

st.caption("Type a question. The agent will search the web, fetch pages, and summarize them with your local model.")

# Sidebar: model picker (fast vs quality)
model = st.sidebar.selectbox(
    "Model",
    options=["llama3.2:3b", "llama3.1:8b"],
    index=0,
    help="3B = fast; 8B = higher-quality (slower)."
)

query = st.text_input("Your research question", placeholder="What are the best tech companies to work for in 2025?")
k = st.slider("How many results to summarize", 1, 5, 3)

if st.button("Run research", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running search and summaries..."):
            result = research(query, k, model=model)
        st.markdown(result)
