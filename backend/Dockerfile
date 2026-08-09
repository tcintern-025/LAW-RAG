FROM python:3.11-slim

WORKDIR /app

# System deps needed by chromadb / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the vector index at image build time so the container is ready to
# serve immediately on boot. For a client with documents that change often,
# move this into an entrypoint script or a separate ingestion job instead.
RUN python -m app.ingestion.build_index

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
