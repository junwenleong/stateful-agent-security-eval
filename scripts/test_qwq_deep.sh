#!/bin/bash
# === qwq:32b Deep Investigation Battery ===
# Run all remaining characterization tests sequentially.
# 1. Long monologue (4000 tok × 5 loads) — does FP drift emerge in a single long turn?
# 2. Prompt-cache isolation (KEEP_ALIVE=0, weather N=5) — is prompt-cache contributing?
# 3. Dense date sweep on fresh load (6 dates × N=3) — are ALL dates VE on a fresh load?
# 4. Marathon warm-state (50 DTA blank runs, then weather N=5) — does longevity shift the boundary?
#
# Total: ~3-4h. Can leave overnight.
# RUN: bash scripts/test_qwq_deep.sh

set -e
cd "$(dirname "$0")/.."

RD="results/qwq_deep"
SUMMARY="$RD/summary.txt"
mkdir -p "$RD"

echo "==============================================" | tee "$SUMMARY"
echo " qwq:32b Deep Investigation — $(date)" | tee -a "$SUMMARY"
echo "==============================================" | tee -a "$SUMMARY"

#############################################
# TEST 1: Long Monologue (5 loads × 4000 tokens single-turn)
#############################################
echo "" | tee -a "$SUMMARY"
echo "=== TEST 1: Long Monologue (4000 tok × 5 loads) ===" | tee -a "$SUMMARY"

PROMPT="Write a detailed, step-by-step explanation of how a compiler transforms source code into machine code. Cover lexing, parsing, AST construction, type checking, intermediate representation, optimization passes, and code generation. Be extremely thorough and include examples for each stage. Do not stop until you have covered all stages in full detail."

for i in $(seq 1 5); do
    pkill -f "ollama" 2>/dev/null || true; sleep 3
    OLLAMA_HOST=0.0.0.0:11434 OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=5m OLLAMA_FLASH_ATTENTION=1 \
    ollama serve &>/dev/null &
    sleep 5
    for a in $(seq 1 15); do curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done

    RESP=$(curl -s http://localhost:11434/api/generate -d "{
        \"model\": \"qwq:32b\",
        \"prompt\": \"$PROMPT\",
        \"stream\": false,
        \"options\": {\"temperature\": 0, \"num_predict\": 4000}
    }")
    # save full response and extract hash for comparison
    echo "$RESP" | python3 -c "import sys,json,hashlib; r=json.load(sys.stdin).get('response',''); open('$RD/monologue_${i}.txt','w').write(r); print(f'  Load $i: {len(r)} chars, md5={hashlib.md5(r.encode()).hexdigest()}')" | tee -a "$SUMMARY"
    kill $(jobs -p) 2>/dev/null || true; sleep 1
done

echo "" | tee -a "$SUMMARY"
python3 -c "
import hashlib, glob
hashes = []
for f in sorted(glob.glob('$RD/monologue_*.txt')):
    h = hashlib.md5(open(f).read().encode()).hexdigest()
    hashes.append(h)
unique = set(hashes)
if len(unique)==1:
    print('MONOLOGUE: IDENTICAL across 5 loads. FP drift does NOT emerge in single-turn 4000-token generation.')
    print('=> Drift requires multi-turn / KV-reset cycles, not just sequence length.')
else:
    print(f'MONOLOGUE: VARIES! {len(unique)} unique outputs across 5 loads.')
    print('=> FP drift IS measurable in long single-turn generation. Smoking gun.')
    # find divergence point
    texts = [open(f).read() for f in sorted(glob.glob('$RD/monologue_*.txt'))]
    for pos in range(min(len(t) for t in texts)):
        chars = set(t[pos] for t in texts)
        if len(chars)>1:
            print(f'   First divergence at char {pos} (~token {pos//4})')
            break
" | tee -a "$SUMMARY"

#############################################
# TEST 2: Prompt-Cache Isolation (KEEP_ALIVE=0)
#############################################
echo "" | tee -a "$SUMMARY"
echo "=== TEST 2: Prompt-Cache Isolation (KEEP_ALIVE=0, weather N=5) ===" | tee -a "$SUMMARY"

pkill -f "ollama" 2>/dev/null || true; sleep 3
OLLAMA_HOST=0.0.0.0:11434 OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_NUM_PARALLEL=1 \
OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=0 OLLAMA_FLASH_ATTENTION=1 \
ollama serve &
PID=$!; sleep 6
for a in $(seq 1 15); do curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
curl -s http://localhost:11434/api/generate -d '{"model":"qwq:32b","prompt":"hi","stream":false}' >/dev/null 2>&1 || true
echo "  Ollama PID=$PID, KEEP_ALIVE=0 (no prompt cache persistence)." | tee -a "$SUMMARY"

PCFILE="$RD/promptcache_weather.jsonl"; > "$PCFILE"
for i in $(seq 1 5); do
    EVAL_OVERRIDE_DATE="2026-06-25" EVAL_INJECT_NONCE="The weather is sunny today." EVAL_NONCE_TARGET="search_search" \
    .venv/bin/python -c "
import sys, json; sys.path.insert(0,'.')
from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner
c = load_config('experiments/configs/defense_factorial.yaml')
c.runs_per_condition = 1
c.results_path = '$PCFILE'
c.models = [m for m in c.models if m['model_name']=='qwq:32b']
c.defenses = [d for d in c.defenses if d.get('name')=='no_defense']
c.attacks = [a for a in c.attacks if a.get('type')=='delayed_trigger']
ExperimentRunner(c).run_all()
r = json.loads(open('$PCFILE').readlines()[-1])
print('  pc_weather$i ASR=' + str(r.get('attack_success')))
" 2>/dev/null | tee -a "$SUMMARY"
done
kill $PID 2>/dev/null || true

#############################################
# TEST 3: Dense Date Sweep (fresh load)
#############################################
echo "" | tee -a "$SUMMARY"
echo "=== TEST 3: Dense Date Sweep (fresh load, 6 dates × N=3) ===" | tee -a "$SUMMARY"

pkill -f "ollama" 2>/dev/null || true; sleep 3
OLLAMA_HOST=0.0.0.0:11434 OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_NUM_PARALLEL=1 \
OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=60m OLLAMA_FLASH_ATTENTION=1 \
ollama serve &
PID=$!; sleep 6
for a in $(seq 1 15); do curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
curl -s http://localhost:11434/api/generate -d '{"model":"qwq:32b","prompt":"hi","stream":false,"keep_alive":"60m"}' >/dev/null 2>&1 || true
echo "  Fresh load PID=$PID" | tee -a "$SUMMARY"

for D in 2026-04-17 2026-04-20 2026-06-22 2026-06-24 2026-06-25 2025-06-24; do
    DFILE="$RD/date_${D}.jsonl"; > "$DFILE"
    for run in $(seq 1 3); do
        EVAL_OVERRIDE_DATE="$D" .venv/bin/python -c "
import sys, json; sys.path.insert(0,'.')
from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner
c = load_config('experiments/configs/defense_factorial.yaml')
c.runs_per_condition = 1
c.results_path = '$DFILE'
c.models = [m for m in c.models if m['model_name']=='qwq:32b']
c.defenses = [d for d in c.defenses if d.get('name')=='no_defense']
c.attacks = [a for a in c.attacks if a.get('type')=='delayed_trigger']
ExperimentRunner(c).run_all()
" 2>/dev/null
    done
    ASR=$(.venv/bin/python -c "
import json
rs=[json.loads(l) for l in open('$DFILE') if l.strip() and not json.loads(l).get('error')]
print(f'{sum(r.get(\"attack_success\") for r in rs)}/{len(rs)}')
")
    echo "  date=$D: ASR=$ASR" | tee -a "$SUMMARY"
done

# Interaction test: does 04-17 + weather nonce together flip on a fresh load?
echo "  --- interaction: date=04-17 + weather nonce (N=5) ---" | tee -a "$SUMMARY"
IFILE="$RD/interaction_0417_weather.jsonl"; > "$IFILE"
for i in $(seq 1 5); do
    EVAL_OVERRIDE_DATE="2026-04-17" EVAL_INJECT_NONCE="The weather is sunny today." EVAL_NONCE_TARGET="search_search" \
    .venv/bin/python -c "
import sys, json; sys.path.insert(0,'.')
from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner
c = load_config('experiments/configs/defense_factorial.yaml')
c.runs_per_condition = 1
c.results_path = '$IFILE'
c.models = [m for m in c.models if m['model_name']=='qwq:32b']
c.defenses = [d for d in c.defenses if d.get('name')=='no_defense']
c.attacks = [a for a in c.attacks if a.get('type')=='delayed_trigger']
ExperimentRunner(c).run_all()
r = json.loads(open('$IFILE').readlines()[-1])
print('  interaction$i ASR=' + str(r.get('attack_success')))
" 2>/dev/null | tee -a "$SUMMARY"
done

kill $PID 2>/dev/null || true

#############################################
# TEST 4: Marathon (50 blank DTA then weather N=5)
#############################################
echo "" | tee -a "$SUMMARY"
echo "=== TEST 4: Marathon (50 blank DTA runs, then weather N=5) ===" | tee -a "$SUMMARY"

pkill -f "ollama" 2>/dev/null || true; sleep 3
OLLAMA_HOST=0.0.0.0:11434 OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_NUM_PARALLEL=1 \
OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=120m OLLAMA_FLASH_ATTENTION=1 \
ollama serve &
PID=$!; sleep 6
for a in $(seq 1 15); do curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
curl -s http://localhost:11434/api/generate -d '{"model":"qwq:32b","prompt":"hi","stream":false,"keep_alive":"120m"}' >/dev/null 2>&1 || true
echo "  Marathon load PID=$PID" | tee -a "$SUMMARY"

# Phase A: 50 blank DTA runs to build up session state
MBLANK="$RD/marathon_blank.jsonl"; > "$MBLANK"
MFINGERPRINT="$RD/marathon_fingerprints.txt"; > "$MFINGERPRINT"
echo "  Phase A: 50 blank DTA runs (logging per-run ASR + fingerprint every 10)" | tee -a "$SUMMARY"
for i in $(seq 1 50); do
    EVAL_OVERRIDE_DATE="2026-06-25" .venv/bin/python -c "
import sys, json; sys.path.insert(0,'.')
from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner
c = load_config('experiments/configs/defense_factorial.yaml')
c.runs_per_condition = 1
c.results_path = '$MBLANK'
c.models = [m for m in c.models if m['model_name']=='qwq:32b']
c.defenses = [d for d in c.defenses if d.get('name')=='no_defense']
c.attacks = [a for a in c.attacks if a.get('type')=='delayed_trigger']
ExperimentRunner(c).run_all()
r = json.loads(open('$MBLANK').readlines()[-1])
print('  blank_run_$i ASR=' + str(r.get('attack_success')))
" 2>/dev/null | tee -a "$SUMMARY"
    # Fingerprint probe every 10 runs: fixed short prompt, capture first 50 chars
    if [ $((i % 10)) -eq 0 ]; then
        FP=$(curl -s http://localhost:11434/api/generate -d '{"model":"qwq:32b","prompt":"Complete: The capital of France is","stream":false,"options":{"temperature":0,"num_predict":10}}' | python3 -c "import sys,json; print(json.load(sys.stdin).get('response','')[:50])" 2>/dev/null)
        echo "  [fingerprint after run $i]: $FP" | tee -a "$SUMMARY"
        echo "run_$i: $FP" >> "$MFINGERPRINT"
    fi
done

# Phase B: now test weather nonce
MWEATHER="$RD/marathon_weather.jsonl"; > "$MWEATHER"
echo "  --- marathon: testing weather nonce after 50 warmup runs ---" | tee -a "$SUMMARY"
for i in $(seq 1 5); do
    EVAL_OVERRIDE_DATE="2026-06-25" EVAL_INJECT_NONCE="The weather is sunny today." EVAL_NONCE_TARGET="search_search" \
    .venv/bin/python -c "
import sys, json; sys.path.insert(0,'.')
from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner
c = load_config('experiments/configs/defense_factorial.yaml')
c.runs_per_condition = 1
c.results_path = '$MWEATHER'
c.models = [m for m in c.models if m['model_name']=='qwq:32b']
c.defenses = [d for d in c.defenses if d.get('name')=='no_defense']
c.attacks = [a for a in c.attacks if a.get('type')=='delayed_trigger']
ExperimentRunner(c).run_all()
r = json.loads(open('$MWEATHER').readlines()[-1])
print('  marathon_weather$i ASR=' + str(r.get('attack_success')))
" 2>/dev/null | tee -a "$SUMMARY"
done
kill $PID 2>/dev/null || true

#############################################
# FINAL SUMMARY
#############################################
echo "" | tee -a "$SUMMARY"
echo "============= DEEP INVESTIGATION COMPLETE =============" | tee -a "$SUMMARY"
echo "Results: $RD/" | tee -a "$SUMMARY"
echo "Check summary.txt for all verdicts." | tee -a "$SUMMARY"
