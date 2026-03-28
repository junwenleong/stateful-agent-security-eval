FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY experiments/ experiments/
COPY data/ data/
COPY scripts/ scripts/
COPY pytest.ini .

CMD ["python", "-m", "src.runner.runner"]
