# Pin to a specific digest to prevent silent base image updates.
# To get the current digest: docker pull python:3.11-slim && docker inspect python:3.11-slim --format '{{index .RepoDigests 0}}'
# Then replace the tag below with: python:3.11-slim@sha256:<digest>
FROM python:3.11-slim@sha256:233de06753d30d120b1a3ce359d8d3be8bda78524cd8f520c99883bfe33964cf

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
