FROM python:3.12-slim

WORKDIR /app

# deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# code
COPY . .

# talk to host’s Ollama by default
ENV OLLAMA_HOST=host.docker.internal
ENV OLLAMA_PORT=11434
ENV PYTHONUNBUFFERED=1

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
