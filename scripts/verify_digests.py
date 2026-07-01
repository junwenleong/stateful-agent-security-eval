"""Pre-flight digest guard: refuse to run experiments if Ollama model weights drifted.

Ollama tags are mutable -- a background `ollama pull` can swap the weights under a
fixed tag (e.g. qwen3.5:122b), silently changing model behaviour between runs. This
was the root cause of the April-vs-June qwq:32b reproducibility crisis, where the
April weights/environment could not be reconstructed because no digest was recorded.

This script compares the live Ollama digests against a pinned manifest and exits
non-zero on any mismatch or missing model. Run it before any factorial or A/B.

Usage:
    # Verify against a manifest
    .venv/bin/python scripts/verify_digests.py --manifest experiments/digest_manifest.json

    # Write the current live digests to a manifest (to pin the current stack)
    .venv/bin/python scripts/verify_digests.py --write experiments/digest_manifest.json
"""
import argparse
import json
import sys
from pathlib import Path

import requests

BASE_URL = "http://localhost:11434"


def live_digests() -> dict:
    resp = requests.get(f"{BASE_URL}/api/tags", timeout=10)
    resp.raise_for_status()
    out = {}
    for m in resp.json().get("models", []):
        name = m.get("name", "")
        digest = (m.get("digest", "") or "")[:12]
        if name:
            out[name] = digest
    return out


def ollama_version() -> str:
    try:
        r = requests.get(f"{BASE_URL}/api/version", timeout=5)
        r.raise_for_status()
        return r.json().get("version", "unknown")
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="Path to pinned manifest JSON to verify against.")
    ap.add_argument("--write", help="Write current live digests + version to this manifest path.")
    args = ap.parse_args()

    try:
        live = live_digests()
    except Exception as e:
        print(f"ERROR: could not reach Ollama at {BASE_URL}: {e}", file=sys.stderr)
        return 2

    ver = ollama_version()

    if args.write:
        manifest = {"ollama_version": ver, "digests": live}
        Path(args.write).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {len(live)} model digests + ollama_version={ver} to {args.write}")
        return 0

    if not args.manifest:
        # No manifest: just print the live state.
        print(f"ollama_version: {ver}")
        for name, digest in sorted(live.items()):
            print(f"  {name:<32} {digest}")
        print("\n(no --manifest given; nothing verified. Use --write to pin.)")
        return 0

    manifest = json.loads(Path(args.manifest).read_text())
    pinned = manifest.get("digests", manifest)  # tolerate bare {name: digest} maps
    pinned_ver = manifest.get("ollama_version")

    problems = []
    if pinned_ver and pinned_ver != ver:
        problems.append(f"ollama_version drift: manifest={pinned_ver} live={ver}")

    for name, want in pinned.items():
        have = live.get(name)
        if have is None:
            problems.append(f"MISSING: {name} (manifest {want}) not present in Ollama")
        elif want and have != want:
            problems.append(f"DIGEST DRIFT: {name} manifest={want} live={have}")

    if problems:
        print("DIGEST GUARD FAILED -- refusing to certify this stack:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print(f"Digest guard OK: {len(pinned)} models match manifest; ollama_version={ver}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
