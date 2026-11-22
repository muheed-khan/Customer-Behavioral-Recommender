FROM python:3.11-slim

WORKDIR /app

# System libs needed by tensorflow/scipy/gensim wheels at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

# Single worker: the recommender model is trained in-process at startup and
# is not safe to share across multiple worker processes.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "300", "app:app"]
