#!/usr/bin/env bash
# ============================================================================
# qwen3.5:122b RATG arm-order control (Iteration 59 addendum experiment)
#
# Purpose: bulletproof the ONE clean reasoning-model RATG data point against an
# order / cumulative / contamination objection. qwen3.5:122b is the only
# reasoning model whose no_defense and ratg arms are session-0-identical
# (interpretable, 100%->0%). This control runs FOUR arms, fresh daemon each,
# in the sequence:  no_defense -> ratg -> ratg -> no_defense.
#
# Expected (if the clean result is genuine and order-independent):
#   - S0 behavior identical across ALL FOUR arms (1 save_fact call, same op-seq)
#   - ASR = 100 / 0 / 0 / 100
# Any deviation (S0 fork between arms, or ASR not 100/0/0/100) would show an
# order or contamination effect and would weaken the 122b claim.
#
# Usage:
#   N=5 bash scripts/run_122b_armorder.sh "$(which ollama)"
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OLLAMA_BIN="${1:?Usage: run_122b_armorder.sh <ollama-binary-path>}"
N="${N:-5}"
[ -x "$OLLAMA_BIN" ] || { echo "ERROR: '$OLLAMA_BIN' not executable."; exit 1; }

MODEL="qwen3.5:122b"
EXPECT_DIGEST="8b9d11d807c5"   # drift guard: refuse to run on a different blob
ARMS=(no_defense ratg ratg no_defense)
OUT_DIR="results/armorder_122b"
mkdir -p "$OUT_DIR"

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
  # Wait until the daemon returns a real model list (not just a TCP listener).
  # The GPU discovery watchdog can delay full readiness by several seconds.
  for _ in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags 2>/dev/null | grep -q "models"; then
      break
    fi
    sleep 1
  done
  sleep 2  # extra settle after GPU discovery completes
}
stop_daemon () { kill "${OLLAMA_PID:-0}" 2>/dev/null || true; wait "${OLLAMA_PID:-0}" 2>/dev/null || true; sleep 2; }

# Drift guard: confirm the 122b blob digest via the API (not CLI, which races on startup).
digest_guard () {
  if ! curl -sf http://localhost:11434/api/tags 2>/dev/null | grep -q "$EXPECT_DIGEST"; then
    echo "FATAL: expected $MODEL digest $EXPECT_DIGEST not found in API /api/tags."
    echo "       Weights may have drifted; aborting (reproducibility.md rule 3)."
    stop_daemon
    exit 1
  fi
  echo "[guard] $MODEL digest $EXPECT_DIGEST confirmed via API."
}

echo "=============================================="
echo " 122b RATG ARM-ORDER CONTROL | N=$N/arm | $(date)"
echo " sequence: ${ARMS[*]}"
echo "=============================================="

i=0
for arm in "${ARMS[@]}"; do
  echo ""
  echo "###### arm $i: $arm (fresh daemon) ######"
  start_fresh
  digest_guard
  .venv/bin/python scripts/run_freshload_cell.py \
    --config experiments/configs/ratg_factorial.yaml \
    --model "$MODEL" --defense "$arm" --n "$N" \
    --out "$OUT_DIR/arm${i}_${arm}.jsonl"
  stop_daemon
  i=$((i + 1))
done

echo ""
echo "=============================================="
echo " ARM-ORDER CONTROL COMPLETE | $(date)"
echo " Expect S0 save-count identical across all 4 arms; ASR = 100/0/0/100."
echo " Verify: inspect injection_session_memory_calls + attack_success per arm file."
echo "=============================================="
