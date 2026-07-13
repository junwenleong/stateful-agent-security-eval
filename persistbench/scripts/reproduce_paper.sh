#!/usr/bin/env bash
# Reproduce all paper figures and tables from PersistBench.
# Requires: OPENAI_API_KEY, ANTHROPIC_API_KEY set for full reproduction.
# Without keys, runs smoke profile only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$ROOT_DIR/results/paper"

cd "$ROOT_DIR"

echo "=== PersistBench: Reproducing Paper Results ==="
echo "Output directory: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

# Step 1: Smoke test (always works, validates harness)
echo ""
echo "--- Step 1/4: Smoke test (offline, ~1 min) ---"
python -m persistbench.cli run --profile smoke --output "$RESULTS_DIR/smoke"

# Step 2: Core runs per model (requires API keys)
MODELS=("gpt-4o-mini" "gpt-4o" "claude-sonnet-4-20250514")

for model in "${MODELS[@]}"; do
    echo ""
    echo "--- Step 2: Core benchmark with $model ---"
    
    # Check if model is available
    if python -c "
from persistbench.cli import get_adapter
a = get_adapter('$model')
exit(0 if a.is_available() else 1)
" 2>/dev/null; then
        python -m persistbench.cli run --profile core --model "$model" \
            --output "$RESULTS_DIR/$model"
    else
        echo "SKIPPED: $model (credentials not configured)"
    fi
done

# Step 3: Generate combined report
echo ""
echo "--- Step 3: Generating combined report ---"
if ls "$RESULTS_DIR"/*/results.jsonl 1>/dev/null 2>&1; then
    cat "$RESULTS_DIR"/*/results.jsonl > "$RESULTS_DIR/combined_results.jsonl"
    python -m persistbench.cli report "$RESULTS_DIR/combined_results.jsonl"
fi

echo ""
echo "=== Reproduction complete ==="
echo "Results directory: $RESULTS_DIR"
echo ""
echo "To generate figures, run:"
echo "  python scripts/generate_figures.py --input $RESULTS_DIR/combined_results.jsonl"
