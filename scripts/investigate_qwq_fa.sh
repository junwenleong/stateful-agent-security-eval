#!/bin/bash
# === qwq:32b Flash Attention Root Cause Investigation ===
# Run this on the Mac Studio. It collects environment data and runs 3 conditions.
#
# BEFORE RUNNING: stop any running Ollama instance (pkill ollama)
# This script manages Ollama restarts itself.
#
# Total time: ~30 min (3 conditions × N=3 × ~3 min/run)

set -e
cd "$(dirname "$0")/.."

RESULTS_DIR="results/qwq_fa_investigation"
mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo " qwq:32b FA Investigation — $(date)"
echo "=============================================="

# === PART 1: Environment Data Collection ===
echo ""
echo "--- ENVIRONMENT DATA ---"
{
  echo "=== Date ==="
  date -u

  echo ""
  echo "=== Ollama version ==="
  ollama --version 2>&1 || echo "ollama not in PATH"

  echo ""
  echo "=== Ollama binary hash ==="
  shasum -a 256 "$(which ollama)" 2>/dev/null || echo "cannot hash"

  echo ""
  echo "=== Ollama binary modification time ==="
  ls -la "$(which ollama)" 2>/dev/null

  echo ""
  echo "=== brew info ollama (install date) ==="
  brew info ollama 2>/dev/null | head -10 || echo "not brew-installed"

  echo ""
  echo "=== qwq:32b GGUF blob timestamp ==="
  find ~/.ollama/models -name "*009cb3f08d74*" -exec ls -la {} \; 2>/dev/null || echo "not found by digest"

  echo ""
  echo "=== macOS version ==="
  sw_vers

  echo ""
  echo "=== Metal/GPU info ==="
  system_profiler SPDisplaysDataType 2>/dev/null | grep -iE "chipset|metal|driver|vendor|vram" || echo "no display data"

  echo ""
  echo "=== KV cache env (should be unset) ==="
  echo "OLLAMA_KV_CACHE_TYPE=${OLLAMA_KV_CACHE_TYPE:-<unset>}"

  echo ""
  echo "=== modelfile KV info ==="
  ollama show qwq:32b --modelfile 2>/dev/null | grep -i "kv\|cache\|quant" || echo "no kv info in modelfile"

} | tee "$RESULTS_DIR/environment.txt"

# === PART 2: Three Conditions ===
# Each condition: restart Ollama with specific flags, run N=3, record results.
# Using N=3 because behavior is deterministic (0/10 and 10/10 in prior tests).

run_condition() {
  local LABEL="$1"
  local FA_FLAG="$2"
  local KV_FLAG="$3"
  local OUTFILE="$RESULTS_DIR/${LABEL}.jsonl"

  echo ""
  echo "=============================================="
  echo " Condition: $LABEL"
  echo " FA=$FA_FLAG  KV=$KV_FLAG"
  echo "=============================================="

  # Kill existing ollama
  pkill -f "ollama serve" 2>/dev/null || true
  sleep 2

  # Start with specified flags
  OLLAMA_HOST=0.0.0.0:11434 \
  OLLAMA_CONTEXT_LENGTH=16384 \
  OLLAMA_NUM_PARALLEL=1 \
  OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_KEEP_ALIVE=5m \
  OLLAMA_FLASH_ATTENTION="$FA_FLAG" \
  OLLAMA_KV_CACHE_TYPE="$KV_FLAG" \
  ollama serve &

  local OLLAMA_PID=$!
  sleep 5  # Wait for model server to start

  # Verify it's running
  if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "ERROR: Ollama failed to start for condition $LABEL"
    return 1
  fi

  # Run the test
  .venv/bin/python -c "
import json, os, sys
sys.path.insert(0, '.')
from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner

config = load_config('experiments/configs/defense_factorial.yaml')
config.runs_per_condition = 3
config.results_path = '$OUTFILE'
config.models = [m for m in config.models if m['model_name'] == 'qwq:32b']
config.defenses = [d for d in config.defenses if d.get('name') in ('no_defense', 'memory_sandbox')]
config.attacks = [a for a in config.attacks if a.get('type') == 'delayed_trigger']
runner = ExperimentRunner(config)
runner.run_all()
"

  echo ""
  echo "--- Results for $LABEL ---"
  .venv/bin/python -c "
import json
from collections import defaultdict
by_def = defaultdict(list)
for line in open('$OUTFILE'):
    r = json.loads(line)
    if not r.get('error'):
        d = r.get('condition', {}).get('defense', {}).get('name', '?')
        by_def[d].append(r.get('attack_success'))
for d, results in sorted(by_def.items()):
    asr = sum(results)
    print(f'  {d}: ASR={asr}/{len(results)}')
"

  # Kill ollama for next condition
  kill $OLLAMA_PID 2>/dev/null || true
  wait $OLLAMA_PID 2>/dev/null || true
  sleep 2
}

# --- Condition 1: FA=1, default KV (the June behavior — positive control) ---
run_condition "fa1_kv_default" "1" ""

# --- Condition 2: FA=0, default KV (the April-like behavior — already confirmed) ---
run_condition "fa0_kv_default" "0" ""

# --- Condition 3: FA=1, forced f16 KV cache (tests if KV quantization is the real cause) ---
run_condition "fa1_kv_f16" "1" "f16"

echo ""
echo "=============================================="
echo " INVESTIGATION COMPLETE"
echo "=============================================="
echo ""
echo "Summary of expected vs actual:"
echo "  fa1_kv_default: expect no_defense=3/3, sandbox=3/3 (June VE behavior)"
echo "  fa0_kv_default: expect no_defense=0/3, sandbox=3/3 (April inversion)"
echo "  fa1_kv_f16:     KEY TEST"
echo "    If no_defense=0/3 → KV quantization is the real cause (FA is a red herring)"
echo "    If no_defense=3/3 → FA kernel itself is the cause (not KV cache)"
echo ""
echo "Results in: $RESULTS_DIR/"
echo "Environment data: $RESULTS_DIR/environment.txt"
