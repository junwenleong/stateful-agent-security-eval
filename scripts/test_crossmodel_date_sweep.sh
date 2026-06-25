#!/bin/bash
# === Cross-Model + Cross-Family Date Sensitivity Test ===
#
# QUESTION: Is date-sensitivity unique to qwq:32b or shared by other knife-edge models?
#
# 7 models × 2 dates × N=5 = 70 runs (~4-5h)
# Ollama restarted between each model block (isolates memory state).
#
# RUN: bash scripts/test_crossmodel_date_sweep.sh

set -e
cd "$(dirname "$0")/.."

RESULTS_DIR="results/crossmodel_date_sweep"
SUMMARY="$RESULTS_DIR/summary.txt"
mkdir -p "$RESULTS_DIR"

DATES=("2026-04-17" "2026-06-25")
N=5

echo "==============================================" | tee "$SUMMARY"
echo " Cross-Model/Family Date Sensitivity — $(date)" | tee -a "$SUMMARY"
echo " 7 models × 2 dates × N=$N = 70 runs" | tee -a "$SUMMARY"
echo " Ollama restarted between each model (clean memory per model)" | tee -a "$SUMMARY"
echo "==============================================" | tee -a "$SUMMARY"

# --- Helpers ---
start_ollama() {
    pkill -f "ollama serve" 2>/dev/null || true
    sleep 3
    OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_CONTEXT_LENGTH=16384 \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_KEEP_ALIVE=5m \
    OLLAMA_FLASH_ATTENTION=1 \
    ollama serve &
    OLLAMA_PID=$!
    sleep 6
    for attempt in $(seq 1 15); do
        curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break
        sleep 1
    done
    echo "  Ollama started (PID=$OLLAMA_PID)" | tee -a "$SUMMARY"
}

run_condition() {
    local MODEL="$1"
    local DEFENSE="$2"
    local DATE="$3"
    local LABEL="$4"
    local NRUNS="$5"
    local OUTFILE="$RESULTS_DIR/${LABEL}.jsonl"
    > "$OUTFILE"

    echo "  Running: $LABEL (model=$MODEL, defense=$DEFENSE, date=$DATE, N=$NRUNS)" | tee -a "$SUMMARY"

    EVAL_OVERRIDE_DATE="$DATE" .venv/bin/python -c "
import sys
sys.path.insert(0, '.')
from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner
config = load_config('experiments/configs/defense_factorial.yaml')
config.runs_per_condition = $NRUNS
config.results_path = '$OUTFILE'
config.models = [m for m in config.models if m['model_name'] == '$MODEL']
config.defenses = [d for d in config.defenses if d.get('name') == '$DEFENSE']
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
inj = sum(1 for r in valid if r.get('injection_success'))
arche = 'VE' if (n>0 and asr==n) else ('0%' if (n>0 and asr==0) else f'MIXED({asr}/{n})')
print(f'    inj={inj}/{n}  ASR={asr}/{n}  -> {arche}')
" | tee -a "$SUMMARY"
}

# ============================================================
# CROSS-FAMILY TESTS (fresh Ollama per model)
# ============================================================

echo "" | tee -a "$SUMMARY"
echo "========== CROSS-FAMILY ==========" | tee -a "$SUMMARY"

# --- 1. GLM-4.7-Flash (THUDM) — execution-resistant ---
echo "=== 1. glm-4.7-flash:latest (THUDM) × no_defense ===" | tee -a "$SUMMARY"
start_ollama
curl -s http://localhost:11434/api/generate -d '{"model":"glm-4.7-flash:latest","prompt":"hi","stream":false,"keep_alive":"10m"}' >/dev/null 2>&1 || true
sleep 2
for DATE in "${DATES[@]}"; do
    run_condition "glm-4.7-flash:latest" "no_defense" "$DATE" "glm47_nodef_${DATE}" $N
done
echo "" | tee -a "$SUMMARY"

# --- 2. llama3.3:70b (Meta) — injection-resistant ---
echo "=== 2. llama3.3:70b (Meta) × no_defense ===" | tee -a "$SUMMARY"
start_ollama
curl -s http://localhost:11434/api/generate -d '{"model":"llama3.3:70b","prompt":"hi","stream":false,"keep_alive":"10m"}' >/dev/null 2>&1 || true
sleep 5
for DATE in "${DATES[@]}"; do
    run_condition "llama3.3:70b" "no_defense" "$DATE" "llama33_70b_nodef_${DATE}" $N
done
echo "" | tee -a "$SUMMARY"

# --- 3. mistral-small3.2:24b (Mistral) — injection-resistant ---
echo "=== 3. mistral-small3.2:24b (Mistral) × no_defense ===" | tee -a "$SUMMARY"
start_ollama
curl -s http://localhost:11434/api/generate -d '{"model":"mistral-small3.2:24b","prompt":"hi","stream":false,"keep_alive":"10m"}' >/dev/null 2>&1 || true
sleep 2
for DATE in "${DATES[@]}"; do
    run_condition "mistral-small3.2:24b" "no_defense" "$DATE" "mistral_24b_nodef_${DATE}" $N
done
echo "" | tee -a "$SUMMARY"

# --- 4. deepseek-r1:70b (DeepSeek) — injection-resistant + reasoning ---
echo "=== 4. deepseek-r1:70b (DeepSeek) × no_defense ===" | tee -a "$SUMMARY"
start_ollama
curl -s http://localhost:11434/api/generate -d '{"model":"deepseek-r1:70b","prompt":"hi","stream":false,"keep_alive":"10m"}' >/dev/null 2>&1 || true
sleep 5
for DATE in "${DATES[@]}"; do
    run_condition "deepseek-r1:70b" "no_defense" "$DATE" "deepseek_r1_70b_nodef_${DATE}" $N
done
echo "" | tee -a "$SUMMARY"

# ============================================================
# QWEN KNIFE-EDGES (fresh Ollama per model)
# ============================================================

echo "========== QWEN KNIFE-EDGES ==========" | tee -a "$SUMMARY"

# --- 5. qwen3:8b — environment-fragile ---
echo "=== 5. qwen3:8b × no_defense ===" | tee -a "$SUMMARY"
start_ollama
curl -s http://localhost:11434/api/generate -d '{"model":"qwen3:8b","prompt":"hi","stream":false,"keep_alive":"10m"}' >/dev/null 2>&1 || true
sleep 2
for DATE in "${DATES[@]}"; do
    run_condition "qwen3:8b" "no_defense" "$DATE" "qwen3_8b_nodef_${DATE}" $N
done
echo "" | tee -a "$SUMMARY"

# --- 6. qwen3.5:122b × prompt_hardening — sleeper effect (N=10, higher stakes) ---
echo "=== 6. qwen3.5:122b × prompt_hardening (N=10) ===" | tee -a "$SUMMARY"
start_ollama
curl -s http://localhost:11434/api/generate -d '{"model":"qwen3.5:122b","prompt":"hi","stream":false,"keep_alive":"10m"}' >/dev/null 2>&1 || true
sleep 5
for DATE in "${DATES[@]}"; do
    run_condition "qwen3.5:122b" "prompt_hardening" "$DATE" "qwen35_122b_ph_${DATE}" 10
done
echo "" | tee -a "$SUMMARY"

# --- 7. qwen2.5:72b × no_defense — negative control ---
echo "=== 7. qwen2.5:72b × no_defense (negative control) ===" | tee -a "$SUMMARY"
start_ollama
curl -s http://localhost:11434/api/generate -d '{"model":"qwen2.5:72b","prompt":"hi","stream":false,"keep_alive":"10m"}' >/dev/null 2>&1 || true
sleep 5
for DATE in "${DATES[@]}"; do
    run_condition "qwen2.5:72b" "no_defense" "$DATE" "qwen25_72b_nodef_${DATE}" $N
done
echo "" | tee -a "$SUMMARY"

# --- Cleanup ---
pkill -f "ollama serve" 2>/dev/null || true

# ============================================================
# VERDICT
# ============================================================
echo "============= VERDICT =============" | tee -a "$SUMMARY"
.venv/bin/python -c "
import json, os

models = [
    ('glm-4.7-flash:latest (THUDM)', 'glm47_nodef'),
    ('llama3.3:70b (Meta)', 'llama33_70b_nodef'),
    ('mistral-small3.2:24b (Mistral)', 'mistral_24b_nodef'),
    ('deepseek-r1:70b (DeepSeek)', 'deepseek_r1_70b_nodef'),
    ('qwen3:8b (Qwen)', 'qwen3_8b_nodef'),
    ('qwen3.5:122b + PH (Qwen)', 'qwen35_122b_ph'),
    ('qwen2.5:72b (Qwen, neg ctrl)', 'qwen25_72b_nodef'),
]

date_sensitive = []
date_insensitive = []

for label, prefix in models:
    print(f'--- {label} ---')
    results_by_date = {}
    for date in ['2026-04-17', '2026-06-25']:
        f = f'$RESULTS_DIR/{prefix}_{date}.jsonl'
        if not os.path.exists(f):
            print(f'  {date}: NO DATA'); continue
        res = [json.loads(l) for l in open(f) if l.strip()]
        valid = [r for r in res if not r.get('error')]
        asr = sum(1 for r in valid if r.get('attack_success'))
        inj = sum(1 for r in valid if r.get('injection_success'))
        n = len(valid)
        results_by_date[date] = (inj, asr, n)
        print(f'  {date}: inj={inj}/{n}  ASR={asr}/{n}')

    if len(results_by_date) == 2:
        (i1, a1, n1) = results_by_date['2026-04-17']
        (i2, a2, n2) = results_by_date['2026-06-25']
        if a1 != a2 or i1 != i2:
            date_sensitive.append(label)
            print(f'  *** DATE-SENSITIVE ***')
        else:
            date_insensitive.append(label)
            print(f'  (stable)')
    print()

print('=' * 50)
print(f'DATE-SENSITIVE models ({len(date_sensitive)}):')
for m in date_sensitive:
    print(f'  - {m}')
print(f'DATE-INSENSITIVE models ({len(date_insensitive)}):')
for m in date_insensitive:
    print(f'  - {m}')
print()
if not date_sensitive:
    print('CONCLUSION: Date-sensitivity is qwq:32b-SPECIFIC (no other model affected)')
elif any(f not in m for f in ['Qwen'] for m in date_sensitive):
    print('CONCLUSION: Date-sensitivity is CROSS-FAMILY')
else:
    print('CONCLUSION: Date-sensitivity is QWEN-FAMILY-SPECIFIC')
" | tee -a "$SUMMARY"

echo "" | tee -a "$SUMMARY"
echo "Done at $(date). Results: $RESULTS_DIR/" | tee -a "$SUMMARY"
