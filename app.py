# app.py — minimal Streamlit UI for your autonomous web agent.

import streamlit as st
from agent.researcher import research

st.set_page_config(page_title="Autonomous Web Agent — Local (Ollama + OSS)", layout="wide")

st.title("😄 Autonomous Web Agent — Local (Ollama + OSS)")
st.caption("Type a question. The agent will search the web, fetch pages, and summarize them with your local model.")

# --- Sidebar controls ---
with st.sidebar:
    st.subheader("Model")
    model = st.selectbox(
        "Pick a local model",
        options=["llama3.2:3b", "llama3.1:8b"],
        index=0,  # default to 3B for speed
    )

    max_chars = st.number_input(
        "Max characters per page",
        min_value=500,
        max_value=4000,
        value=1200,
        step=100,
        help="We only send this many characters from each page to the model. Lower = faster, higher = more context.",
    )

# --- Main inputs ---
query = st.text_input(
    "Your research question",
    placeholder="e.g., What are the best tech companies to work for in 2025?",
)
k = st.slider("How many results to summarize", min_value=1, max_value=5, value=3)

if st.button("Run research"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running the agent…"):
            result = research(query=query, k=int(k), model=model, max_chars=int(max_chars))
        st.markdown(result)
