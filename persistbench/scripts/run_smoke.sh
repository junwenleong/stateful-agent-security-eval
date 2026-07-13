#!/usr/bin/env bash
# PersistBench Smoke Test - Fully offline, <10 minutes
# Uses pre-recorded traces, no API keys needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

echo "=== PersistBench Smoke Test ==="
echo "This runs entirely offline using recorded model traces."
echo ""

python -m persistbench.cli run --profile smoke --output results/smoke

echo ""
echo "=== Smoke test complete ==="
echo "Results: results/smoke/results.jsonl"
echo "Report:  results/smoke/report.json"
