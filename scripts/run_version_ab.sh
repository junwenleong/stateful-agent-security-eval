#!/usr/bin/env bash
# ============================================================================
# Reasoning-model Ollama VERSION A/B  (0.20.6 vs 0.30.11)
#
# Isolates whether a reasoning model's no_defense DTA ASR is driven by the Ollama
# runtime version. Same digest, same prompt, same corpus, same code -- only the
# daemon version differs between the two invocations.
#
# Motivating case (Iteration 57): qwen3.5:9b, digest 6488c96fa5fa, is 100% ASR on
# 0.20.6 (April N=10) and 0% ASR on 0.30.11 (July fresh-load). The 0% is a coherent
# instruction-comprehension MISPARSE (satisfies "use send_email", misses "forward to
# auditor@"), NOT a refusal and NOT daemon degradation. This confirms the version is
# the sole differing variable.
#
# Run ONE version per invocation (version switching is external, like the qwq battery):
#
#   # current stack (0.30.11) -- the ollama on PATH / Ollama.app
#   bash scripts/run_version_ab.sh "$(which ollama)" 0.30.11 qwen3.5:9b
#
#   # old stack (0.20.6) -- reinstall the v0.20.6 binary to a separate path first:
#   #   https://github.com/ollama/ollama/releases/tag/v0.20.6
#   bash scripts/run_version_ab.sh /path/to/ollama-0.20.6 0.20.6 qwen3.5:9b
#
# Multiple flipped models in one invocation (whichever showed 0% on 0.30.11 fresh-load):
#   bash scripts/run_version_ab.sh "$(which ollama)" 0.30.11 qwen3.5:9b gpt-oss:20b
#
# Then classify (after BOTH versions run):
#   .venv/bin/python scripts/classify_version_ab.py results/version_ab/*.jsonl
#
# Env:
#   N              runs per model per version (default 10)
#   EXPECT_DIGEST  optional single-model digest guard (only meaningful for 1 model)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OLLAMA_BIN="${1:?Usage: run_version_ab.sh <ollama-binary> <version-label> <model> [model...]}"
VERSION_LABEL="${2:?Usage: run_version_ab.sh <ollama-binary> <version-label> <model> [model...]}"
shift 2
MODELS=("$@")
[ "${#MODELS[@]}" -ge 1 ] || { echo "ERROR: give at least one model tag."; exit 1; }

N="${N:-10}"
EXPECT_DIGEST="${EXPECT_DIGEST:-}"
VLABEL="$(echo "$VERSION_LABEL" | tr '.' 'p')"
OUTDIR="results/version_ab"
mkdir -p "$OUTDIR"

safe () { echo "$1" | tr './:' '___'; }

echo "=============================================="
echo " Reasoning-model VERSION A/B  |  version=$VERSION_LABEL  |  N=$N  |  $(date)"
echo " ollama binary: $OLLAMA_BIN"
echo " models: ${MODELS[*]}"
echo "=============================================="

if [ ! -x "$OLLAMA_BIN" ]; then
  echo "ERROR: '$OLLAMA_BIN' is not an executable ollama binary."
  echo ""
  echo "To get 0.20.6 (overwritten by the 0.30.x app upgrade), reinstall to a separate path:"
  echo "  https://github.com/ollama/ollama/releases/tag/v0.20.6"
  echo "Or find any existing copy:"
  echo "  find /Applications /usr/local /opt \$HOME -iname 'ollama' -type f 2>/dev/null \\"
  echo "    -exec sh -c 'echo \"\$1:\"; \"\$1\" --version 2>/dev/null' _ {} \\;"
  exit 1
fi

echo "Reported binary version:"; "$OLLAMA_BIN" --version 2>&1 || true

# --- start the daemon with pinned flags (match factorial serve config) ---------
pkill -f "ollama serve" 2>/dev/null || true
sleep 3
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
echo "Ollama daemon up (PID=$OLLAMA_PID)."

# --- pin the live digest + version for this version phase -----------------------
.venv/bin/python scripts/verify_digests.py --write "$OUTDIR/manifest_${VLABEL}.json" || true
echo ""

for MODEL in "${MODELS[@]}"; do
  MS="$(safe "$MODEL")"
  # Capture effective runtime defaults for THIS model on THIS version (mechanism pin).
  .venv/bin/python scripts/dump_ollama_runtime.py \
    --model "$MODEL" --out "$OUTDIR/runtime_${MS}__${VLABEL}.json" || true

  OUT="$OUTDIR/${MS}__${VLABEL}.jsonl"
  echo "---- cell: model=$MODEL version=$VERSION_LABEL -> $OUT ----"
  EXPECT_ARGS=()
  # Only apply the digest guard when a single model is under test (guard is per-model).
  if [ -n "$EXPECT_DIGEST" ] && [ "${#MODELS[@]}" -eq 1 ]; then
    EXPECT_ARGS=(--expect-digest "$EXPECT_DIGEST")
  fi
  .venv/bin/python scripts/run_version_ab_cell.py \
    --model "$MODEL" --n "$N" --out "$OUT" "${EXPECT_ARGS[@]}"
done

kill "$OLLAMA_PID" 2>/dev/null || true
wait "$OLLAMA_PID" 2>/dev/null || true

echo ""
echo "=============================================="
echo " DONE (version=$VERSION_LABEL). Results in $OUTDIR/"
echo " After running BOTH versions, classify:"
echo "   .venv/bin/python scripts/classify_version_ab.py $OUTDIR/*.jsonl"
echo "=============================================="
