#!/usr/bin/env bash
# ============================================================================
# Fresh-load suite — the "proper" redo of the sprint's uninterpretable arms.
#
# The RATG sprint ran all 9 models in ONE long-running daemon; that churn
# degraded reasoning-model trigger-session behaviour (Iteration 55): no-tool-call
# stalls, team-only degraded sends, address confabulation — 0/~324 refusals,
# NOT safety. Fresh daemon loads restore coherent behaviour (122b: 60/60 = 100%).
#
# This suite fixes that by restarting Ollama FRESH before every model, so each
# model runs in isolation with no prior-model GPU/Metal churn. It produces:
#   1. Clean reasoning-model RATG data (no_defense baseline + ratg), 6 models.
#   2. The 7B-judge capability test (no_defense + rag_llm_judge_7b), 3 models.
#
# The 3 mechanical models (qwen2.5:14b/72b, qwen3:32b) already have clean RATG
# data from the sprint and are NOT re-run here (they did not degrade).
#
# Usage:
#   bash scripts/run_freshload_suite.sh "$(which ollama)"
#   N=40 bash scripts/run_freshload_suite.sh "$(which ollama)"   # rigorous
#
# Then classify:
#   .venv/bin/python scripts/classify_122b_ab.py results/freshload_ratg/*.jsonl
#   .venv/bin/python scripts/run_ratg_factorial.py  # (no) -- use analyze below
#   .venv/bin/python scripts/freshload_summary.py    # per-model no_defense vs defended
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OLLAMA_BIN="${1:?Usage: run_freshload_suite.sh <ollama-binary-path>}"
N="${N:-20}"
[ -x "$OLLAMA_BIN" ] || { echo "ERROR: '$OLLAMA_BIN' not executable."; exit 1; }

# Reasoning models whose sprint RATG arms were uninterpretable (degraded baseline).
RATG_MODELS=(qwen3.5:9b qwen3.5:122b qwq:32b gpt-oss:20b gpt-oss-safeguard:120b glm-4.7-flash:bf16)
# 7B-judge test models (from judge_7b.yaml).
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
echo " FRESH-LOAD SUITE | N=$N/arm | fresh daemon per model | $(date)"
echo " ollama: $OLLAMA_BIN"
echo "=============================================="

# ---- 1. Reasoning-model RATG (no_defense + ratg), fresh daemon per model ----
mkdir -p results/freshload_ratg
for m in "${RATG_MODELS[@]}"; do
  echo ""
  echo "###### RATG fresh load: $m ######"
  start_fresh
  .venv/bin/python scripts/verify_digests.py --write "results/freshload_ratg/manifest_$(safe "$m").json" || true
  .venv/bin/python scripts/run_freshload_cell.py \
    --config experiments/configs/ratg_factorial.yaml \
    --model "$m" --n "$N" --out "results/freshload_ratg/$(safe "$m").jsonl"
  stop_daemon
done

# ---- 2. 7B-judge capability test (no_defense + rag_llm_judge_7b), fresh per model ----
mkdir -p results/freshload_judge7b
for m in "${JUDGE_MODELS[@]}"; do
  echo ""
  echo "###### 7B-judge fresh load: $m ######"
  start_fresh
  .venv/bin/python scripts/run_freshload_cell.py \
    --config experiments/configs/judge_7b.yaml \
    --model "$m" --n "$N" --out "results/freshload_judge7b/$(safe "$m").jsonl"
  stop_daemon
done

echo ""
echo "=============================================="
echo " SUITE DONE. Summarize:"
echo "   .venv/bin/python scripts/freshload_summary.py"
echo "=============================================="
