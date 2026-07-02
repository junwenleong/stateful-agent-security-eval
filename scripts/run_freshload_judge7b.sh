#!/usr/bin/env bash
# ============================================================================
# Fresh-load 7B Judge Suite (re-run #2)
#
# History of invalidations:
#   1. Original run: judge_7b.yaml was missing the injection query as
#      benign_queries[0] -> session 0 never retrieved the malicious doc.
#      Fixed in commit 255786b.
#   2. Re-run #1 (Iteration 60): the qwen2.5:7b judge model was NOT served,
#      so every /api/chat judge call returned 404. The judge defense fails
#      OPEN (is_malicious=False on error), so all 640 judge decisions passed
#      through silently. judge_model_hash was None on all 120 judge-arm runs.
#      "0 flagged" was a wiring artifact, not a capability finding.
#
# This re-run adds a PRE-FLIGHT judge-reachability probe that ABORTS loudly if
# qwen2.5:7b is not served, so a missing judge can never again fail open into
# invalid data. It also runs with MAX_LOADED_MODELS=2 so the small judge
# coexists with the agent model instead of thrashing (evict/reload the agent
# on every RAG call under =1), and writes a reproducibility manifest per model.
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
JUDGE_MODEL="qwen2.5:7b"   # must match judge_model in judge_7b.yaml

OUT_DIR="results/freshload_judge7b"
mkdir -p "$OUT_DIR"

start_fresh () {
  pkill -f "ollama serve" 2>/dev/null || true
  sleep 4
  # MAX_LOADED_MODELS=2: agent model + qwen2.5:7b judge resident together.
  # Largest pair 122b(75GB)+7b(~5GB)=80GB < 96GB unified memory -> safe, no thrash.
  OLLAMA_HOST=0.0.0.0:11434 \
  OLLAMA_CONTEXT_LENGTH=16384 \
  OLLAMA_NUM_PARALLEL=1 \
  OLLAMA_MAX_LOADED_MODELS=2 \
  OLLAMA_KEEP_ALIVE=5m \
  OLLAMA_FLASH_ATTENTION=1 \
  "$OLLAMA_BIN" serve &
  OLLAMA_PID=$!
  for _ in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags 2>/dev/null | grep -q "models"; then
      break
    fi
    sleep 1
  done
  sleep 2  # extra settle after GPU discovery completes
}
stop_daemon () { kill "${OLLAMA_PID:-0}" 2>/dev/null || true; wait "${OLLAMA_PID:-0}" 2>/dev/null || true; sleep 2; }

# Pre-flight: the judge MUST answer /api/chat. Abort loudly on anything but 200.
# This is the guard whose absence invalidated re-run #1 (silent 404 fail-open).
probe_judge () {
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/chat \
    -d "{\"model\":\"${JUDGE_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"reply with OK\"}],\"stream\":false}" \
    2>/dev/null || echo "000")
  if [ "$code" != "200" ]; then
    echo "FATAL: judge model '${JUDGE_MODEL}' not reachable on /api/chat (HTTP ${code})."
    echo "       Run: ollama pull ${JUDGE_MODEL}   and verify: ollama list | grep ${JUDGE_MODEL}"
    stop_daemon
    exit 1
  fi
  echo "[probe] judge ${JUDGE_MODEL} reachable (HTTP 200) and warm."
}

safe () { echo "$1" | tr ':/.' '___'; }

# Reproducibility manifest per model. Uses `|| true` throughout so a manifest
# hiccup can never abort the run (set -e safe). Full `ollama list` (with digests)
# is written to a sidecar .txt so the exact agent+judge blob hashes are recorded.
write_manifest () {
  local agent="$1" out="$2"
  local ver bin_sha commit dirty
  ver="$($OLLAMA_BIN --version 2>/dev/null | head -1 || true)"
  bin_sha="$(shasum -a 256 "$OLLAMA_BIN" 2>/dev/null | awk '{print $1}' || true)"
  commit="$(git rev-parse HEAD 2>/dev/null || true)"
  dirty=false; [ -n "$(git status --porcelain 2>/dev/null || true)" ] && dirty=true
  $OLLAMA_BIN list > "${out%.json}.ollama_list.txt" 2>/dev/null || true
  cat > "$out" <<EOF || true
{
  "agent_model": "${agent}",
  "judge_model": "${JUDGE_MODEL}",
  "ollama_version": "${ver}",
  "ollama_bin_sha256": "${bin_sha}",
  "git_commit": "${commit}",
  "git_dirty": ${dirty},
  "env": {"OLLAMA_CONTEXT_LENGTH": 16384, "OLLAMA_NUM_PARALLEL": 1, "OLLAMA_MAX_LOADED_MODELS": 2, "OLLAMA_KEEP_ALIVE": "5m", "OLLAMA_FLASH_ATTENTION": 1},
  "ollama_list_sidecar": "${out%.json}.ollama_list.txt"
}
EOF
}

echo "=============================================="
echo " FRESH-LOAD 7B JUDGE (re-run #2) | N=$N/arm | $(date)"
echo " ollama: $OLLAMA_BIN | judge: $JUDGE_MODEL | MAX_LOADED_MODELS=2"
echo "=============================================="

for m in "${JUDGE_MODELS[@]}"; do
  echo ""
  echo "###### 7B Judge fresh load: $m ######"
  start_fresh
  probe_judge   # aborts the whole run if the judge is missing
  write_manifest "$m" "$OUT_DIR/manifest_$(safe "$m").json"
  .venv/bin/python scripts/run_freshload_cell.py \
    --config experiments/configs/judge_7b.yaml \
    --model "$m" --n "$N" --out "$OUT_DIR/$(safe "$m").jsonl"
  stop_daemon
done

echo ""
echo "=============================================="
echo " 7B JUDGE COMPLETE | $(date)"
echo " Verify: .venv/bin/python scripts/freshload_deep_analysis.py"
echo " CHECK judge_model_hash is populated and judge_reason has NO 404 before trusting results."
echo "=============================================="
