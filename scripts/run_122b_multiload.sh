#!/usr/bin/env bash
# ============================================================================
# qwen3.5:122b multi-fresh-load stability probe
#
# Answers ONE question: is qwen3.5:122b's ASR stable across independent fresh
# daemon loads, or does it flip/scatter between loads?
#
# The A/B (run_122b_version_ab.sh) showed a single fresh load = 100% ASR (with
# suffix), while the sprint (122b mid-factorial) = 0/40. This probe restarts the
# Ollama daemon from scratch K times and runs the SAME cell (with-suffix, default
# num_predict) N times per load, writing each load to its own file.
#
# Reads:
#   - All loads ~100%  -> fresh-load is stable; the sprint's 0/40 was the churned
#     multi-model load context. Load-dependent but reproducible-by-context.
#   - Loads scatter (e.g. 100/0/60) -> irreducible per-load nondeterminism, like
#     qwq:32b's 18/40. BSI<1.0 confirmed on a second reasoning model.
#
# Usage:
#   bash scripts/run_122b_multiload.sh "$(which ollama)"
#   K=3 N=20 SUFFIX=with bash scripts/run_122b_multiload.sh "$(which ollama)"
#
# Then:
#   .venv/bin/python scripts/classify_122b_ab.py results/122b_multiload/*.jsonl
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OLLAMA_BIN="${1:?Usage: run_122b_multiload.sh <ollama-binary-path>}"
K="${K:-3}"                 # number of independent fresh daemon loads
N="${N:-20}"                # runs per load
SUFFIX="${SUFFIX:-with}"    # with | without
OUTDIR="results/122b_multiload"
mkdir -p "$OUTDIR"

[ -x "$OLLAMA_BIN" ] || { echo "ERROR: '$OLLAMA_BIN' is not executable."; exit 1; }

echo "=============================================="
echo " 122b multi-load probe | K=$K loads | N=$N/load | suffix=$SUFFIX"
echo " ollama: $OLLAMA_BIN | $(date)"
echo "=============================================="

for i in $(seq 1 "$K"); do
  echo ""
  echo "###### LOAD $i/$K — fresh daemon restart ######"
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

  OUT="$OUTDIR/load${i}_${SUFFIX}.jsonl"
  .venv/bin/python scripts/run_122b_ab_cell.py \
    --suffix "$SUFFIX" --num-predict default --n "$N" --out "$OUT"

  kill "$OLLAMA_PID" 2>/dev/null || true
  wait "$OLLAMA_PID" 2>/dev/null || true
  sleep 2
done

echo ""
echo "=============================================="
echo " DONE. Per-load results in $OUTDIR/"
echo "   .venv/bin/python scripts/classify_122b_ab.py $OUTDIR/*.jsonl"
echo ""
echo " Read: all loads ~equal  -> fresh-load stable (sprint 0/40 was churned context)"
echo "       loads scatter      -> irreducible per-load nondeterminism (BSI<1.0, 2nd model)"
echo "=============================================="
