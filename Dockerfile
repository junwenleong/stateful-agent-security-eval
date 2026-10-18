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

# What this image is for: it provides the pinned Python environment together with the source,
# configurations and scripts, and its default command re-checks the paper's numbers against the
# raw result files (scripts/verify_canonical.py). It does NOT run model inference: the experiments
# were run against Ollama and hosted-model endpoints that are not part of this image, and the
# per-trial result data are not copied in. Mount them to verify:
#   docker build -t sleeperbench .
#   docker run --rm -v "$PWD/results:/app/results:ro" sleeperbench
# The verifier reads only /app/results and makes no network or model calls.
CMD ["python", "scripts/verify_canonical.py"]
