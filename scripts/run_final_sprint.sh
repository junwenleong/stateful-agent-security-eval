#!/usr/bin/env bash
# ============================================================================
# Combined: 122b arm-order control + 7B judge rerun (one command, sequential)
#
# Usage (on Mac Studio):
#   N_ARM=5 N_JUDGE=40 bash scripts/run_final_sprint.sh "$(which ollama)"
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OLLAMA_BIN="${1:?Usage: run_final_sprint.sh <ollama-binary-path>}"
N_ARM="${N_ARM:-5}"
N_JUDGE="${N_JUDGE:-40}"

echo "=============================================="
echo " FINAL SPRINT | $(date)"
echo " Phase 1: 122b arm-order control (N=$N_ARM/arm, ~30 min)"
echo " Phase 2: 7B judge rerun (N=$N_JUDGE/arm, ~5-6h)"
echo "=============================================="

# --- Phase 1: arm-order control ---
N="$N_ARM" bash scripts/run_122b_armorder.sh "$OLLAMA_BIN"

# --- Phase 2: 7B judge rerun ---
N="$N_JUDGE" bash scripts/run_freshload_judge7b.sh "$OLLAMA_BIN"

echo ""
echo "=============================================="
echo " ALL DONE | $(date)"
echo " Verify:"
echo "   .venv/bin/python scripts/freshload_deep_analysis.py"
echo "   Check arm-order: results/armorder_122b/ (expect S0 identical, ASR 100/0/0/100)"
echo "   Check judge: results/freshload_judge7b/ (expect judge_model_hash populated, no 404)"
echo "=============================================="
