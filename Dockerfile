FROM python:3.11-slim

WORKDIR /app

# Install build tools needed for some pip packages (e.g. onnxruntime, tokenizers)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1. Copy & install dependencies FIRST (Docker layer caching optimization)
#    This layer is only rebuilt when requirements.txt changes, not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy application code
COPY . .

# Expose port (Hugging Face Spaces defaults to 7860)
EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
