FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and static UI.
COPY app ./app
COPY static ./static

# The container talks to the Ollama server on the host. Override per environment.
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    LLM_MODEL=llama3.1:8b \
    DATABASE_URL=sqlite:////data/chat.db

# Persist the SQLite database outside the image layer.
VOLUME ["/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
