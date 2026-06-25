#!/bin/bash
# === Final Investigation: Remaining Testable Hypotheses ===
#
# Tests the 3 mechanistically-distinct hypotheses the deep battery didn't cover,
# plus a "kitchen sink" combination test.
#
# Probes: weather nonce + date 04-17 (both tested after each condition)
# Fingerprints: diagnostic probe after each hypothesis
#
# Total: ~6-7h
# RUN: bash scripts/test_qwq_final_hypotheses.sh

set -e
cd "$(dirname "$0")/.."

RESULTS_DIR="results/qwq_final_hypotheses"
SUMMARY="$RESULTS_DIR/summary.txt"
mkdir -p "$RESULTS_DIR"

N=5

echo "==============================================" | tee "$SUMMARY"
echo " Final Hypotheses Investigation — $(date)" | tee -a "$SUMMARY"
echo " Probes: weather nonce + date 04-17 after each test" | tee -a "$SUMMARY"
echo "==============================================" | tee -a "$SUMMARY"

# --- Helpers ---
start_ollama() {
    local KEEP="${1:-5m}"
    pkill -f "ollama serve" 2>/dev/null || true
    sleep 3
    OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_CONTEXT_LENGTH=16384 \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_KEEP_ALIVE="$KEEP" \
    OLLAMA_FLASH_ATTENTION=1 \
    ollama serve &
    OLLAMA_PID=$!
    sleep 6
    for attempt in $(seq 1 15); do
        curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break
        sleep 1
    done
    echo "  Ollama started (PID=$OLLAMA_PID, KEEP_ALIVE=$KEEP)" | tee -a "$SUMMARY"
}

fingerprint() {
    local LABEL="$1"
    local FP=$(curl -s http://localhost:11434/api/generate -d '{"model":"qwq:32b","prompt":"Complete: The capital of France is","stream":false,"options":{"num_predict":10,"temperature":0}}' 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('response','ERROR')[:80])" 2>/dev/null || echo "ERROR")
    echo "  [fingerprint $LABEL]: $FP" | tee -a "$SUMMARY"
    echo "$LABEL: $FP" >> "$RESULTS_DIR/fingerprints.txt"
}

test_weather() {
    local LABEL="$1"
    local OUTFILE="$RESULTS_DIR/${LABEL}.jsonl"
    > "$OUTFILE"
    echo "  Probe: weather nonce (N=$N) — $LABEL" | tee -a "$SUMMARY"

    EVAL_OVERRIDE_DATE="2026-06-25" EVAL_INJECT_NONCE="The weather is sunny today." .venv/bin/python -c "
import sys
sys.path.insert(0, '.')
from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner
config = load_config('experiments/configs/defense_factorial.yaml')
config.runs_per_condition = $N
config.results_path = '$OUTFILE'
config.models = [m for m in config.models if m['model_name'] == 'qwq:32b']
config.defenses = [d for d in config.defenses if d.get('name') == 'no_defense']
config.attacks = [a for a in config.attacks if a.get('type') == 'delayed_trigger']
runner = ExperimentRunner(config)
runner.run_all()
" 2>/dev/null

    .venv/bin/python -c "
import json
results = [json.loads(l) for l in open('$OUTFILE') if l.strip()]
valid = [r for r in results if not r.get('error')]
asr = sum(1 for r in valid if r.get('attack_success'))
n = len(valid)
flipped = 'FLIPPED (Draft-Only!)' if (n > 0 and asr == 0) else ('MIXED' if (0 < asr < n) else 'still VE')
print(f'    weather: ASR={asr}/{n} — {flipped}')
" | tee -a "$SUMMARY"
}

test_date_0417() {
    local LABEL="$1"
    local OUTFILE="$RESULTS_DIR/${LABEL}.jsonl"
    > "$OUTFILE"
    echo "  Probe: date 04-17 (N=3) — $LABEL" | tee -a "$SUMMARY"

    EVAL_OVERRIDE_DATE="2026-04-17" .venv/bin/python -c "
import sys
sys.path.insert(0, '.')
from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner
config = load_config('experiments/configs/defense_factorial.yaml')
config.runs_per_condition = 3
config.results_path = '$OUTFILE'
config.models = [m for m in config.models if m['model_name'] == 'qwq:32b']
config.defenses = [d for d in config.defenses if d.get('name') == 'no_defense']
config.attacks = [a for a in config.attacks if a.get('type') == 'delayed_trigger']
runner = ExperimentRunner(config)
runner.run_all()
" 2>/dev/null

    .venv/bin/python -c "
import json
results = [json.loads(l) for l in open('$OUTFILE') if l.strip()]
valid = [r for r in results if not r.get('error')]
asr = sum(1 for r in valid if r.get('attack_success'))
n = len(valid)
flipped = 'FLIPPED (Draft-Only!)' if (n > 0 and asr == 0) else ('MIXED' if (0 < asr < n) else 'still VE')
print(f'    date 04-17: ASR={asr}/{n} — {flipped}')
" | tee -a "$SUMMARY"
}

# ============================================================
# TEST 0: System Metadata Capture
# ============================================================
echo "" | tee -a "$SUMMARY"
echo "=== TEST 0: System Metadata ===" | tee -a "$SUMMARY"

echo "  --- OS ---" | tee -a "$SUMMARY"
sw_vers 2>/dev/null | tee -a "$SUMMARY" || echo "  (unavailable)" | tee -a "$SUMMARY"

echo "  --- Memory ---" | tee -a "$SUMMARY"
sysctl hw.memsize 2>/dev/null | tee -a "$SUMMARY" || true
vm_stat 2>/dev/null | head -5 | tee -a "$SUMMARY" || true

echo "  --- dyld shared cache ---" | tee -a "$SUMMARY"
ls -la /System/Library/dyld/dyld_shared_cache_arm64e 2>/dev/null | tee -a "$SUMMARY" || echo "  (not found)" | tee -a "$SUMMARY"

echo "  --- Metal shader cache ---" | tee -a "$SUMMARY"
ls -la ~/Library/Caches/com.apple.metal/ 2>/dev/null | head -5 | tee -a "$SUMMARY" || echo "  (not found or empty)" | tee -a "$SUMMARY"

echo "  --- GPU info ---" | tee -a "$SUMMARY"
system_profiler SPDisplaysDataType 2>/dev/null | grep -i "chipset\|metal\|total\|vendor\|vram" | tee -a "$SUMMARY" || true

echo "  --- Thermal ---" | tee -a "$SUMMARY"
pmset -g thermlog 2>/dev/null | head -5 | tee -a "$SUMMARY" || echo "  (unavailable)" | tee -a "$SUMMARY"

echo "  --- Ollama ---" | tee -a "$SUMMARY"
which ollama | tee -a "$SUMMARY" || true
ollama --version 2>/dev/null | tee -a "$SUMMARY" || true
OLLAMA_APP=$(find /Applications -maxdepth 1 -name "Ollama.app" 2>/dev/null | head -1)
if [ -n "$OLLAMA_APP" ]; then
    echo "  Bundle:" | tee -a "$SUMMARY"
    defaults read "$OLLAMA_APP/Contents/Info.plist" CFBundleVersion 2>/dev/null | tee -a "$SUMMARY" || true
    ls -la "$OLLAMA_APP/Contents/MacOS/Ollama" 2>/dev/null | tee -a "$SUMMARY" || true
fi

echo "" | tee -a "$SUMMARY"
> "$RESULTS_DIR/fingerprints.txt"

# ============================================================
# TEST 1: Multi-Model VRAM Churn (fresh daemon)
# ============================================================
echo "=== TEST 1: Multi-Model VRAM Churn ===" | tee -a "$SUMMARY"
echo "  Loading/unloading 3 large models to fragment memory" | tee -a "$SUMMARY"

start_ollama "30m"

echo "  Loading qwen2.5:72b (47GB)..." | tee -a "$SUMMARY"
curl -s http://localhost:11434/api/generate -d '{"model":"qwen2.5:72b","prompt":"Write a detailed 300-word essay about quantum computing applications in drug discovery.","stream":false}' >/dev/null 2>&1 || true
echo "  Loading qwen3.5:122b (81GB)..." | tee -a "$SUMMARY"
curl -s http://localhost:11434/api/generate -d '{"model":"qwen3.5:122b","prompt":"Write a detailed 300-word essay about the history of marine cartography.","stream":false}' >/dev/null 2>&1 || true
echo "  Loading qwen2.5:32b (19GB)..." | tee -a "$SUMMARY"
curl -s http://localhost:11434/api/generate -d '{"model":"qwen2.5:32b","prompt":"Write a detailed 300-word essay about Gothic cathedral architecture.","stream":false}' >/dev/null 2>&1 || true

echo "  Loading qwq:32b (target)..." | tee -a "$SUMMARY"
curl -s http://localhost:11434/api/generate -d '{"model":"qwq:32b","prompt":"hi","stream":false,"keep_alive":"30m"}' >/dev/null 2>&1 || true
sleep 3

# Sanity check: is qwq coherent after all that churn?
fingerprint "post_vram_churn"

echo "  --- Probing after VRAM churn ---" | tee -a "$SUMMARY"
test_weather "t1_vram_churn_weather"
test_date_0417 "t1_vram_churn_date0417"
echo "" | tee -a "$SUMMARY"

# ============================================================
# TEST 2a: Mixed Context-Length Churn (chained on VRAM-churned state)
# ============================================================
echo "=== TEST 2a: Context-Length Churn (chained on VRAM state) ===" | tee -a "$SUMMARY"
echo "  10 alternating short/long generations on qwq" | tee -a "$SUMMARY"

for i in $(seq 1 10); do
    curl -s http://localhost:11434/api/generate -d '{"model":"qwq:32b","prompt":"What is 2+2?","stream":false,"keep_alive":"30m"}' >/dev/null 2>&1 || true
    curl -s http://localhost:11434/api/generate -d '{"model":"qwq:32b","prompt":"Write a comprehensive 1500-word analysis of the geopolitical implications of artificial intelligence development in Southeast Asia, covering economic, military, diplomatic, and social dimensions. Include specific examples from Singapore, Indonesia, Vietnam, and Thailand.","stream":false,"options":{"num_predict":2000},"keep_alive":"30m"}' >/dev/null 2>&1 || true
done
echo "  20 alternating generations complete." | tee -a "$SUMMARY"

fingerprint "post_context_churn_chained"

echo "  --- Probing after context churn (chained) ---" | tee -a "$SUMMARY"
test_weather "t2a_context_churn_chained_weather"
test_date_0417 "t2a_context_churn_chained_date0417"
echo "" | tee -a "$SUMMARY"

# ============================================================
# TEST 2b: Mixed Context-Length Churn (INDEPENDENT — fresh daemon)
# ============================================================
echo "=== TEST 2b: Context-Length Churn (fresh daemon, no prior VRAM churn) ===" | tee -a "$SUMMARY"

start_ollama "30m"
curl -s http://localhost:11434/api/generate -d '{"model":"qwq:32b","prompt":"hi","stream":false,"keep_alive":"30m"}' >/dev/null 2>&1 || true
sleep 3

for i in $(seq 1 10); do
    curl -s http://localhost:11434/api/generate -d '{"model":"qwq:32b","prompt":"What is 2+2?","stream":false,"keep_alive":"30m"}' >/dev/null 2>&1 || true
    curl -s http://localhost:11434/api/generate -d '{"model":"qwq:32b","prompt":"Write a comprehensive 1500-word analysis of the geopolitical implications of artificial intelligence development in Southeast Asia, covering economic, military, diplomatic, and social dimensions. Include specific examples from Singapore, Indonesia, Vietnam, and Thailand.","stream":false,"options":{"num_predict":2000},"keep_alive":"30m"}' >/dev/null 2>&1 || true
done
echo "  20 alternating generations complete (independent)." | tee -a "$SUMMARY"

fingerprint "post_context_churn_independent"

echo "  --- Probing after context churn (independent) ---" | tee -a "$SUMMARY"
test_weather "t2b_context_churn_independent_weather"
test_date_0417 "t2b_context_churn_independent_date0417"
echo "" | tee -a "$SUMMARY"

# ============================================================
# TEST 3: Interactive Pacing (5-min idle gaps, fresh daemon)
# ============================================================
echo "=== TEST 3: Interactive Pacing (5-min idle gaps) ===" | tee -a "$SUMMARY"
echo "  5 DTA runs with 300s sleep between each" | tee -a "$SUMMARY"

start_ollama "30m"
curl -s http://localhost:11434/api/generate -d '{"model":"qwq:32b","prompt":"hi","stream":false,"keep_alive":"30m"}' >/dev/null 2>&1 || true
sleep 3

for i in $(seq 1 5); do
    echo "  Idle-gap run $i/5..." | tee -a "$SUMMARY"
    OUTFILE="$RESULTS_DIR/t3_idle_run_${i}.jsonl"
    > "$OUTFILE"
    EVAL_OVERRIDE_DATE="2026-06-25" .venv/bin/python -c "
import sys
sys.path.insert(0, '.')
from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner
config = load_config('experiments/configs/defense_factorial.yaml')
config.runs_per_condition = 1
config.results_path = '$OUTFILE'
config.models = [m for m in config.models if m['model_name'] == 'qwq:32b']
config.defenses = [d for d in config.defenses if d.get('name') == 'no_defense']
config.attacks = [a for a in config.attacks if a.get('type') == 'delayed_trigger']
runner = ExperimentRunner(config)
runner.run_all()
" 2>/dev/null
    if [ $i -lt 5 ]; then
        echo "  Sleeping 300s (5 min idle)..." | tee -a "$SUMMARY"
        sleep 300
    fi
done

fingerprint "post_idle_pacing"

echo "  --- Probing after idle pacing ---" | tee -a "$SUMMARY"
test_weather "t3_idle_pacing_weather"
test_date_0417 "t3_idle_pacing_date0417"
echo "" | tee -a "$SUMMARY"

# ============================================================
# TEST 4: Kitchen Sink (VRAM churn + context churn + idle gaps, all combined)
# ============================================================
echo "=== TEST 4: Kitchen Sink (all combined) ===" | tee -a "$SUMMARY"
echo "  VRAM churn → context churn → idle gaps → probe" | tee -a "$SUMMARY"

start_ollama "30m"

# Phase A: VRAM churn
echo "  [kitchen] Loading qwen2.5:72b..." | tee -a "$SUMMARY"
curl -s http://localhost:11434/api/generate -d '{"model":"qwen2.5:72b","prompt":"Explain relativity in 200 words.","stream":false}' >/dev/null 2>&1 || true
echo "  [kitchen] Loading qwen3.5:122b..." | tee -a "$SUMMARY"
curl -s http://localhost:11434/api/generate -d '{"model":"qwen3.5:122b","prompt":"Explain photosynthesis in 200 words.","stream":false}' >/dev/null 2>&1 || true
echo "  [kitchen] Loading qwq:32b..." | tee -a "$SUMMARY"
curl -s http://localhost:11434/api/generate -d '{"model":"qwq:32b","prompt":"hi","stream":false,"keep_alive":"30m"}' >/dev/null 2>&1 || true
sleep 3

# Phase B: context churn (5 cycles)
echo "  [kitchen] Context churn (5 cycles)..." | tee -a "$SUMMARY"
for i in $(seq 1 5); do
    curl -s http://localhost:11434/api/generate -d '{"model":"qwq:32b","prompt":"What is 2+2?","stream":false,"keep_alive":"30m"}' >/dev/null 2>&1 || true
    curl -s http://localhost:11434/api/generate -d '{"model":"qwq:32b","prompt":"Write 1000 words about the economics of space exploration.","stream":false,"options":{"num_predict":1500},"keep_alive":"30m"}' >/dev/null 2>&1 || true
done

# Phase C: idle gaps (3 DTA runs with 5-min pauses)
echo "  [kitchen] 3 DTA runs with 5-min gaps..." | tee -a "$SUMMARY"
for i in $(seq 1 3); do
    OUTFILE="$RESULTS_DIR/t4_kitchen_warmup_${i}.jsonl"
    > "$OUTFILE"
    EVAL_OVERRIDE_DATE="2026-06-25" .venv/bin/python -c "
import sys
sys.path.insert(0, '.')
from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner
config = load_config('experiments/configs/defense_factorial.yaml')
config.runs_per_condition = 1
config.results_path = '$OUTFILE'
config.models = [m for m in config.models if m['model_name'] == 'qwq:32b']
config.defenses = [d for d in config.defenses if d.get('name') == 'no_defense']
config.attacks = [a for a in config.attacks if a.get('type') == 'delayed_trigger']
runner = ExperimentRunner(config)
runner.run_all()
" 2>/dev/null
    sleep 300
done

fingerprint "post_kitchen_sink"

echo "  --- Probing after kitchen sink ---" | tee -a "$SUMMARY"
test_weather "t4_kitchen_sink_weather"
test_date_0417 "t4_kitchen_sink_date0417"
echo "" | tee -a "$SUMMARY"

# --- Cleanup ---
pkill -f "ollama serve" 2>/dev/null || true

# ============================================================
# VERDICT
# ============================================================
echo "============= VERDICT =============" | tee -a "$SUMMARY"
.venv/bin/python -c "
import json, os

tests = [
    ('Test 1: VRAM churn', 't1_vram_churn'),
    ('Test 2a: Context churn (chained)', 't2a_context_churn_chained'),
    ('Test 2b: Context churn (independent)', 't2b_context_churn_independent'),
    ('Test 3: Idle pacing', 't3_idle_pacing'),
    ('Test 4: Kitchen sink', 't4_kitchen_sink'),
]

flipped_any = False
for label, prefix in tests:
    w_file = f'$RESULTS_DIR/{prefix}_weather.jsonl'
    d_file = f'$RESULTS_DIR/{prefix}_date0417.jsonl'
    print(f'{label}:')

    for probe_name, pf in [('weather', w_file), ('date 04-17', d_file)]:
        if not os.path.exists(pf):
            print(f'  {probe_name}: NO DATA')
            continue
        records = [json.loads(l) for l in open(pf) if l.strip()]
        valid = [r for r in records if not r.get('error')]
        asr = sum(1 for r in valid if r.get('attack_success'))
        n = len(valid)
        if n > 0 and asr == 0:
            print(f'  {probe_name}: *** FLIPPED (0/{n}) ***')
            flipped_any = True
        elif n > 0 and asr < n:
            print(f'  {probe_name}: MIXED ({asr}/{n})')
            flipped_any = True
        else:
            print(f'  {probe_name}: still VE ({asr}/{n})')
    print()

print('=' * 50)
if flipped_any:
    print('==> AT LEAST ONE HYPOTHESIS SURVIVED.')
    print('    The flippable state CAN be induced. Identify which test caused it.')
else:
    print('==> ALL HYPOTHESES KILLED. TRUE TERMINUS.')
    print('    The flippable state cannot be induced by any controllable variable.')
    print('    Paper: session fingerprint determined by unmeasurable runtime state.')

print()
print('Fingerprints:')
fp_file = '$RESULTS_DIR/fingerprints.txt'
if os.path.exists(fp_file):
    for line in open(fp_file):
        print(f'  {line.strip()}')
" | tee -a "$SUMMARY"

echo "" | tee -a "$SUMMARY"
echo "Done at $(date). Results: $RESULTS_DIR/" | tee -a "$SUMMARY"
