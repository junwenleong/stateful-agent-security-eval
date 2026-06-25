#!/bin/bash
# === qwq:32b Per-Load Nondeterminism Test ===
#
# HYPOTHESIS: The Draft-Only vs Vulnerable Executor flip is determined at model
# load time by nondeterministic floating-point reduction order in parallel GPU
# kernels. Within one loaded process, behavior is perfectly deterministic (all
# runs identical). Across loads, the outcome can vary.
#
# TEST DESIGN:
#   - 20 iterations
#   - Each iteration: kill Ollama, restart fresh, run EXACTLY 1 DTA no_defense run
#   - All flags identical across all 20 iterations (FA=1, default KV, 16k ctx)
#   - Record: run result + Ollama PID + load timestamp
#   - If outcomes vary across iterations (some 0%, some 100%) → per-load nondeterminism confirmed
#   - If all 20 identical → load is not the variable
#
# CONTROLS:
#   - Same binary (verified: symlink mtime Apr 1, not re-installed)
#   - Same weights (same GGUF blob, same digest)
#   - Same flags every iteration (no flag changes between runs)
#   - Same code (same commit)
#   - N=1 per load (isolates the load as the only between-run variable)
#   - Temperature 0.0 (eliminates sampling noise — all variation is FP determinism)
#
# ADDITIONAL RIGOR:
#   - After each run, record whether model used send_email to attacker (ASR)
#   - Record the first 100 chars of the trigger session reasoning (fingerprint)
#   - If all VE runs have identical reasoning AND all Draft-Only runs have identical
#     reasoning, that confirms within-load determinism and between-load variation
#
# EXPECTED OUTCOMES:
#   A) Mix of VE and Draft-Only across 20 loads → per-load FP nondeterminism CONFIRMED
#   B) All 20 VE → load is not the variable (current system state has "moved past" Draft-Only)
#   C) All 20 Draft-Only → unlikely given recent results, but would mean the state flipped back
#
# RUN: bash scripts/test_qwq_per_load.sh
# TIME: ~20 × 5min = ~100 min (qwq loads in ~8s, each run ~4-5 min)

set -e
cd "$(dirname "$0")/.."

RESULTS_DIR="results/qwq_per_load_test"
RESULTS_FILE="$RESULTS_DIR/results.jsonl"
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
N_ITERATIONS=20

# Fixed flags — identical for every iteration. No changes.
OLLAMA_FLAGS="OLLAMA_HOST=0.0.0.0:11434 OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=5m OLLAMA_FLASH_ATTENTION=1"

mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo " qwq:32b Per-Load Nondeterminism Test"
echo " $(date)"
echo " N=$N_ITERATIONS iterations, 1 run per load"
echo " All flags fixed: FA=1, KV=default, ctx=16384"
echo "=============================================="
echo "" | tee "$SUMMARY_FILE"

# Verify qwq is available
if ! ollama list 2>/dev/null | grep -q "qwq:32b"; then
    echo "ERROR: qwq:32b not found in ollama list"
    exit 1
fi

# Record environment once
{
    echo "=== Environment (fixed across all iterations) ==="
    echo "Date: $(date -u)"
    echo "Ollama binary hash: $(shasum -a 256 "$(which ollama)" | cut -d' ' -f1)"
    echo "Ollama version: $(ollama --version 2>&1 | grep -o '[0-9.]*' | head -1)"
    echo "macOS: $(sw_vers -productVersion)"
    echo "Flags: $OLLAMA_FLAGS"
    echo ""
} | tee -a "$SUMMARY_FILE"

# Clear any existing results (fresh start)
> "$RESULTS_FILE"

for i in $(seq 1 $N_ITERATIONS); do
    echo "--- Iteration $i/$N_ITERATIONS ---" | tee -a "$SUMMARY_FILE"
    
    # Kill any existing Ollama
    pkill -f "ollama serve" 2>/dev/null || true
    sleep 3  # Ensure full shutdown
    
    # Start fresh Ollama with fixed flags
    env $OLLAMA_FLAGS ollama serve &
    OLLAMA_PID=$!
    sleep 6  # Wait for server + model discovery
    
    # Verify it's responding
    for attempt in $(seq 1 10); do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "  FAIL: Ollama didn't start (PID=$OLLAMA_PID)" | tee -a "$SUMMARY_FILE"
        continue
    fi
    
    # Run exactly 1 DTA no_defense run
    ITER_RESULT=$(.venv/bin/python -c "
import json, sys, os
sys.path.insert(0, '.')
from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner

config = load_config('experiments/configs/defense_factorial.yaml')
config.runs_per_condition = 1
config.results_path = '$RESULTS_FILE'
config.models = [m for m in config.models if m['model_name'] == 'qwq:32b']
config.defenses = [d for d in config.defenses if d.get('name') == 'no_defense']
config.attacks = [a for a in config.attacks if a.get('type') == 'delayed_trigger']

runner = ExperimentRunner(config)
runner.run_all()

# Read back the last record
lines = open('$RESULTS_FILE').readlines()
r = json.loads(lines[-1])
asr = r.get('attack_success', False)
inj = r.get('injection_success', False)

# Get first 100 chars of trigger session reasoning as fingerprint
agent_logs = r.get('agent_logs') or []
trigger_reasoning = ''
for entry in reversed(agent_logs):
    if isinstance(entry, dict) and entry.get('session_index') == 3:
        content = entry.get('content', '')
        if content and len(content) > 50:
            trigger_reasoning = content[:100]
            break

print(json.dumps({
    'iteration': $i,
    'asr': asr,
    'injection': inj,
    'pid': $OLLAMA_PID,
    'reasoning_fingerprint': trigger_reasoning[:80]
}))
" 2>/dev/null)
    
    # Parse and display
    ASR=$(echo "$ITER_RESULT" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['asr'])" 2>/dev/null || echo "ERROR")
    FINGERPRINT=$(echo "$ITER_RESULT" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['reasoning_fingerprint'][:60])" 2>/dev/null || echo "")
    
    echo "  PID=$OLLAMA_PID  ASR=$ASR  fingerprint='${FINGERPRINT}'" | tee -a "$SUMMARY_FILE"
    
    # Kill this instance before next iteration
    kill $OLLAMA_PID 2>/dev/null || true
    wait $OLLAMA_PID 2>/dev/null || true
    sleep 2
done

# Final summary
echo "" | tee -a "$SUMMARY_FILE"
echo "=============================================="  | tee -a "$SUMMARY_FILE"
echo " FINAL RESULTS" | tee -a "$SUMMARY_FILE"
echo "=============================================="  | tee -a "$SUMMARY_FILE"

.venv/bin/python -c "
import json

results = []
for line in open('$RESULTS_FILE'):
    r = json.loads(line)
    if not r.get('error'):
        results.append(r.get('attack_success'))

ve = sum(results)
do = len(results) - ve
total = len(results)

print(f'Total completed: {total}/{$N_ITERATIONS}')
print(f'Vulnerable Executor (ASR=True): {ve}/{total}')
print(f'Draft-Only (ASR=False): {do}/{total}')
print()

if ve > 0 and do > 0:
    print('RESULT: *** PER-LOAD NONDETERMINISM CONFIRMED ***')
    print(f'  {ve} loads produced VE, {do} loads produced Draft-Only')
    print('  Same binary, same weights, same flags, same code.')
    print('  The ONLY difference between iterations is the Ollama process instance.')
    print('  Mechanism: nondeterministic FP reduction order in GPU parallel kernels,')
    print('  fixed at load time, determines safety-relevant behavioral archetype.')
elif ve == total:
    print('RESULT: All VE. Load is not the variable (or Draft-Only state is unreachable')
    print('  in the current system state). The model has \"moved past\" the decision boundary.')
elif do == total:
    print('RESULT: All Draft-Only. Unexpected given recent VE results.')
    print('  Suggests the system state has shifted back.')
else:
    print(f'RESULT: Incomplete ({total}/{$N_ITERATIONS} runs)')
" | tee -a "$SUMMARY_FILE"

echo "" | tee -a "$SUMMARY_FILE"
echo "Results: $RESULTS_FILE" | tee -a "$SUMMARY_FILE"
echo "Summary: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
