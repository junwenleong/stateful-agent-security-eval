#!/usr/bin/env bash
# ============================================================================
# qwen3.5:122b version x suffix x num_predict A/B
#
# Isolates whether the sprint's 100%->0% ASR flip for reasoning models is driven
# by Ollama VERSION (0.20.6 vs 0.30.11), the trigger SUFFIX, or a generation-cap
# (num_predict) TRUNCATION artifact.
#
# This script runs ONE Ollama version per invocation (version switching is done
# externally, exactly as the qwq isolation battery did it). Run it twice:
#
#   # 1) current stack (0.30.11) -- the ollama on PATH / Ollama.app
#   bash scripts/run_122b_version_ab.sh "$(which ollama)" 0.30.11
#
#   # 2) old stack (0.20.6) -- requires the 0.20.6 binary (see availability check)
#   bash scripts/run_122b_version_ab.sh /path/to/ollama-0.20.6 0.20.6
#
# Then classify:
#   .venv/bin/python scripts/classify_122b_ab.py results/122b_version_ab/*.jsonl
#
# Cells run per invocation (N=20 each):
#   with/default     - baseline for this version
#   without/default  - suffix control
#   with/raised      - TRUNCATION control (num_predict=8192); decisive for cell E
#
# Decisive reads:
#   - If with/raised restores high ASR while with/default is 0% -> the "0%" is a
#     generation-cap truncation artifact, NOT a version or suffix safety effect.
#   - If 0.20.6 with/default = 100% and 0.30.11 with/default = 0% (raised also 0%)
#     -> genuine version-driven flip.
#   - If without/default differs from with/default within a version -> suffix effect.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OLLAMA_BIN="${1:?Usage: run_122b_version_ab.sh <ollama-binary-path> <version-label>}"
VERSION_LABEL="${2:?Usage: run_122b_version_ab.sh <ollama-binary-path> <version-label>}"
N="${N:-20}"
RAISED_NUM_PREDICT="${RAISED_NUM_PREDICT:-8192}"
EXPECT_DIGEST="${EXPECT_DIGEST:-}"   # optional; leave empty to skip the digest guard

VLABEL="$(echo "$VERSION_LABEL" | tr '.' 'p')"
OUTDIR="results/122b_version_ab"
mkdir -p "$OUTDIR"

echo "=============================================="
echo " qwen3.5:122b A/B  |  version=$VERSION_LABEL  |  N=$N  |  $(date)"
echo " ollama binary: $OLLAMA_BIN"
echo "=============================================="

# --- 0.20.6 availability check -------------------------------------------------
if [ ! -x "$OLLAMA_BIN" ]; then
  echo "ERROR: '$OLLAMA_BIN' is not an executable ollama binary."
  echo ""
  echo "If you need 0.20.6 and it was overwritten by the 0.30.11 app upgrade, find any copy:"
  echo "  find /Applications /usr/local /opt \$HOME -iname 'ollama' -type f 2>/dev/null \\"
  echo "    -exec sh -c 'echo \"\$1:\"; \"\$1\" --version 2>/dev/null' _ {} \\;"
  echo "Or reinstall it to a separate path from https://github.com/ollama/ollama/releases (tag v0.20.6)."
  exit 1
fi

echo "Reported binary version:"
"$OLLAMA_BIN" --version 2>&1 || true

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

# --- record the live digest + version for this version phase --------------------
.venv/bin/python scripts/verify_digests.py --write "$OUTDIR/manifest_${VLABEL}.json" || true
echo ""

run_cell () {
  local suffix="$1" np="$2" tag="$3"
  local out="$OUTDIR/${VLABEL}_${tag}.jsonl"
  echo "---- cell: version=$VERSION_LABEL suffix=$suffix num_predict=$np -> $out ----"
  local expect_args=()
  [ -n "$EXPECT_DIGEST" ] && expect_args=(--expect-digest "$EXPECT_DIGEST")
  .venv/bin/python scripts/run_122b_ab_cell.py \
    --suffix "$suffix" --num-predict "$np" --n "$N" --out "$out" "${expect_args[@]}"
}

run_cell with    default              "with_default"
run_cell without default              "without_default"
run_cell with    "$RAISED_NUM_PREDICT" "with_raised"

# --- cleanup -------------------------------------------------------------------
kill "$OLLAMA_PID" 2>/dev/null || true
wait "$OLLAMA_PID" 2>/dev/null || true

echo ""
echo "=============================================="
echo " DONE (version=$VERSION_LABEL). Results in $OUTDIR/"
echo " After running BOTH versions, classify:"
echo "   .venv/bin/python scripts/classify_122b_ab.py $OUTDIR/*.jsonl"
echo "=============================================="
