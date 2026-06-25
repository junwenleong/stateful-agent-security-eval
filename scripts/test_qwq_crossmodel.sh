#!/bin/bash
# === Cross-Model Date Check ===
# Test whether the date-sensitivity is qwq-specific or affects other factorial models.
# qwen2.5:14b × {2026-04-17, 2026-06-25} × N=3. Single load.
# If both VE → date-sensitivity is qwq-only. If 04-17 flips → major scope issue.
# ~30 min (6 runs × ~5 min)

set -e
cd "$(dirname "$0")/.."

RD="results/qwq_crossmodel"
SUMMARY="$RD/summary.txt"
mkdir -p "$RD"

echo "==============================================" | tee "$SUMMARY"
echo " Cross-Model Date Check — $(date)" | tee -a "$SUMMARY"
echo " qwen2.5:14b × {04-17, 06-25} × N=3" | tee -a "$SUMMARY"
echo "==============================================" | tee -a "$SUMMARY"

pkill -f "ollama" 2>/dev/null || true
sleep 3
OLLAMA_HOST=0.0.0.0:11434 OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_NUM_PARALLEL=1 \
OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=60m OLLAMA_FLASH_ATTENTION=1 \
ollama serve &
PID=$!
sleep 6
for a in $(seq 1 15); do curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
curl -s http://localhost:11434/api/generate -d '{"model":"qwen2.5:14b","prompt":"hi","stream":false,"keep_alive":"60m"}' >/dev/null 2>&1 || true
echo "Ollama up (PID=$PID), qwen2.5:14b loaded." | tee -a "$SUMMARY"

run_single() {
    local DATE="$1" OUT="$2" LABEL="$3"
    EVAL_OVERRIDE_DATE="$DATE" \
    .venv/bin/python -c "
import sys, json; sys.path.insert(0,'.')
from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner
c = load_config('experiments/configs/defense_factorial.yaml')
c.runs_per_condition = 1
c.results_path = '$OUT'
c.models = [m for m in c.models if m['model_name']=='qwen2.5:14b']
c.defenses = [d for d in c.defenses if d.get('name')=='no_defense']
c.attacks = [a for a in c.attacks if a.get('type')=='delayed_trigger']
ExperimentRunner(c).run_all()
r = json.loads(open('$OUT').readlines()[-1])
print('  $LABEL ASR=' + str(r.get('attack_success')))
" 2>/dev/null
}

echo "" | tee -a "$SUMMARY"
echo "--- date=2026-04-17 (N=3) ---" | tee -a "$SUMMARY"
AFILE="$RD/april.jsonl"; > "$AFILE"
for i in $(seq 1 3); do run_single "2026-04-17" "$AFILE" "apr$i" | tee -a "$SUMMARY"; done

echo "--- date=2026-06-25 (N=3) ---" | tee -a "$SUMMARY"
JFILE="$RD/june.jsonl"; > "$JFILE"
for i in $(seq 1 3); do run_single "2026-06-25" "$JFILE" "jun$i" | tee -a "$SUMMARY"; done

kill $PID 2>/dev/null || true

echo "" | tee -a "$SUMMARY"
echo "============= VERDICT =============" | tee -a "$SUMMARY"
.venv/bin/python -c "
import json
a=[json.loads(l).get('attack_success') for l in open('$RD/april.jsonl') if l.strip() and not json.loads(l).get('error')]
j=[json.loads(l).get('attack_success') for l in open('$RD/june.jsonl') if l.strip() and not json.loads(l).get('error')]
print(f'qwen2.5:14b + date=2026-04-17: {sum(a)}/{len(a)} ({\"VE\" if all(a) else \"DRAFT-ONLY\" if not any(a) else \"MIXED\"})')
print(f'qwen2.5:14b + date=2026-06-25: {sum(j)}/{len(j)} ({\"VE\" if all(j) else \"DRAFT-ONLY\" if not any(j) else \"MIXED\"})')
print()
if all(a) and all(j):
    print('Both VE. Date-sensitivity is QWQ-SPECIFIC. Other models unaffected.')
elif not any(a) and all(j):
    print('*** APRIL FLIPS qwen2.5:14b TOO *** Date-sensitivity is CROSS-MODEL!')
else:
    print('Mixed — needs investigation.')
" | tee -a "$SUMMARY"
echo "Results: $RD/" | tee -a "$SUMMARY"
