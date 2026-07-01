"""Dump the effective Ollama runtime configuration for the live daemon.

Captures everything that could differ between Ollama versions and change a reasoning
model's token-level behaviour: version, per-model Modelfile parameters (num_ctx,
num_predict, temperature, etc.), model details (quantization, param size), and the
OLLAMA_* server env vars visible to this process.

Run this once per version arm (0.20.6 and 0.30.11), for the model under test, so that
if the ASR flips we can attribute it to a concrete default change (num_predict, KV
cache dtype, context length, flash attention) rather than "the version, somehow".

Per reproducibility.md: Ollama's num_predict default, num_ctx default, KV cache type,
and flash-attention behaviour have all changed across releases. Record them.

Usage:
    .venv/bin/python scripts/dump_ollama_runtime.py --model qwen3.5:9b \
        --out results/version_ab/runtime_0p20p6.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import requests

BASE_URL = "http://localhost:11434"

# Server-level env vars that affect inference/numeric behaviour (see reproducibility.md).
OLLAMA_ENV_KEYS = [
    "OLLAMA_HOST", "OLLAMA_CONTEXT_LENGTH", "OLLAMA_NUM_PARALLEL",
    "OLLAMA_MAX_LOADED_MODELS", "OLLAMA_KEEP_ALIVE", "OLLAMA_FLASH_ATTENTION",
    "OLLAMA_KV_CACHE_TYPE", "OLLAMA_NUM_GPU", "OLLAMA_NUM_THREAD",
    "OLLAMA_GPU_OVERHEAD", "OLLAMA_LLM_LIBRARY", "OLLAMA_SCHED_SPREAD",
]


def get_version() -> str:
    try:
        r = requests.get(f"{BASE_URL}/api/version", timeout=5)
        r.raise_for_status()
        return r.json().get("version", "unknown")
    except Exception as e:
        return f"unknown ({e})"


def show_model(model: str) -> dict:
    """POST /api/show returns parameters, template, details, model_info."""
    try:
        r = requests.post(f"{BASE_URL}/api/show", json={"name": model}, timeout=15)
        r.raise_for_status()
        d = r.json()
        # Keep the fields that matter for reproducibility; drop the giant template blob.
        out = {
            "parameters": d.get("parameters"),          # Modelfile PARAMETER lines (num_ctx, etc.)
            "details": d.get("details"),                # quantization_level, parameter_size, family
            "model_info_keys": sorted((d.get("model_info") or {}).keys()),
        }
        # Pull a few high-value model_info numeric fields if present.
        mi = d.get("model_info") or {}
        for k in list(mi.keys()):
            kl = k.lower()
            if "context_length" in kl or "rope" in kl or "attention" in kl or "head" in kl:
                out.setdefault("model_info_selected", {})[k] = mi[k]
        return out
    except Exception as e:
        return {"error": str(e)}


def live_digest(model: str) -> str:
    try:
        r = requests.get(f"{BASE_URL}/api/tags", timeout=10)
        r.raise_for_status()
        for m in r.json().get("models", []):
            if m.get("name") == model:
                return (m.get("digest", "") or "")[:12]
    except Exception:
        pass
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = {
        "ollama_version": get_version(),
        "model": args.model,
        "digest": live_digest(args.model),
        "server_env": {k: os.environ.get(k) for k in OLLAMA_ENV_KEYS},
        "show": show_model(args.model),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote runtime manifest -> {args.out}")
    print(f"  version={manifest['ollama_version']} digest={manifest['digest']}")
    params = manifest["show"].get("parameters")
    if params:
        print("  Modelfile parameters:")
        for line in str(params).splitlines():
            print(f"    {line}")
    print("  server_env:")
    for k, v in manifest["server_env"].items():
        if v is not None:
            print(f"    {k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
