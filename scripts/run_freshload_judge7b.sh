#!/usr/bin/env bash
# ============================================================================
# Fresh-load 7B Judge Suite (re-run after config fix)
#
# The original judge_7b.yaml was missing the injection query as benign_queries[0],
# so session 0 never retrieved the malicious doc and injection never occurred.
# All 230 prior runs are invalid. This re-runs with the fixed config.
#
# Usage:
#   N=40 bash scripts/run_freshload_judge7b.sh "$(which ollama)"
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OLLAMA_BIN="${1:?Usage: run_freshload_judge7b.sh <ollama-binary-path>}"
N="${N:-40}"
[ -x "$OLLAMA_BIN" ] || { echo "ERROR: '$OLLAMA_BIN' not executable."; exit 1; }

JUDGE_MODELS=(qwen2.5:14b qwen3:32b qwen3.5:122b)

start_fresh () {
  pkill -f "ollama serve" 2>/dev/null || true
  sleep 4
  OLLAMA_HOST=0.0.0.0:11434 \
  OLLAMA_CONTEXT_LENGTH=16384 \
  OLLAMA_NUM_PARALLEL=1 \
  OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_KEEP_ALIVE=5m \
  OLLAMA_FLASH_ATTENTION=1 \
  "$OLLAMA_BIN" serve &
  OLLAMA_PID=$!
  for _ in $(seq 1 20); do
    curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
}
stop_daemon () { kill "${OLLAMA_PID:-0}" 2>/dev/null || true; wait "${OLLAMA_PID:-0}" 2>/dev/null || true; sleep 2; }

safe () { echo "$1" | tr ':/.' '___'; }

echo "=============================================="
echo " FRESH-LOAD 7B JUDGE (re-run) | N=$N/arm | $(date)"
echo " ollama: $OLLAMA_BIN"
echo "=============================================="

mkdir -p results/freshload_judge7b

for m in "${JUDGE_MODELS[@]}"; do
  echo ""
  echo "###### 7B Judge fresh load: $m ######"
  start_fresh
  .venv/bin/python scripts/run_freshload_cell.py \
    --config experiments/configs/judge_7b.yaml \
    --model "$m" --n "$N" --out "results/freshload_judge7b/$(safe "$m").jsonl"
  stop_daemon
done

echo ""
echo "=============================================="
echo " 7B JUDGE COMPLETE | $(date)"
echo "=============================================="
