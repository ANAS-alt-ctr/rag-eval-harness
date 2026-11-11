# RAG System with Failure Analysis + LLM Eval Harness
FROM python:3.12-slim

WORKDIR /app

# System deps: gcc is needed only if a wheel is missing for some package.
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the embedding model at image build time so first use is fast.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000 8501

# Default: the FastAPI backend. Run Streamlit separately (see docker-compose.yml).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
