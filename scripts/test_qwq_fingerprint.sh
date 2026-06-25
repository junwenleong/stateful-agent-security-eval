#!/bin/bash
# === qwq:32b Load Fingerprinting ===
# 20 fresh daemon loads. Each: single fixed prompt, record first generated token.
# If token varies across loads → Metal FP nondeterminism is measurable at single-token.
# If identical → nondeterminism only emerges in long sequences (attention accumulation).
# ~30 min (20 loads × ~90s each for load+1 inference)

set -e
cd "$(dirname "$0")/.."

RD="results/qwq_fingerprint"
SUMMARY="$RD/summary.txt"
mkdir -p "$RD"

PROMPT="Complete this sentence with exactly one word: The capital of France is"

echo "==============================================" | tee "$SUMMARY"
echo " qwq:32b Load Fingerprinting — $(date)" | tee -a "$SUMMARY"
echo " 20 fresh loads, single-token probe, temp=0" | tee -a "$SUMMARY"
echo "==============================================" | tee -a "$SUMMARY"

for i in $(seq 1 20); do
    pkill -f "ollama" 2>/dev/null || true
    sleep 3
    OLLAMA_HOST=0.0.0.0:11434 OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=5m OLLAMA_FLASH_ATTENTION=1 \
    ollama serve &>/dev/null &
    sleep 5
    for a in $(seq 1 15); do curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done

    RESP=$(curl -s http://localhost:11434/api/generate -d "{
        \"model\": \"qwq:32b\",
        \"prompt\": \"$PROMPT\",
        \"stream\": false,
        \"options\": {\"temperature\": 0, \"num_predict\": 20}
    }")
    TOKEN=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('response','ERROR')[:80])" 2>/dev/null)
    echo "  Load $i: $TOKEN" | tee -a "$SUMMARY"
    echo "$TOKEN" >> "$RD/tokens.txt"
    kill $(jobs -p) 2>/dev/null || true
    sleep 1
done

echo "" | tee -a "$SUMMARY"
echo "============= ANALYSIS =============" | tee -a "$SUMMARY"
python3 -c "
tokens = [l.strip() for l in open('$RD/tokens.txt')]
unique = set(tokens)
print(f'Total loads: {len(tokens)}')
print(f'Unique responses: {len(unique)}')
if len(unique) == 1:
    print(f'IDENTICAL across all loads: {repr(tokens[0])}')
    print('=> Single-token inference is deterministic. Nondeterminism emerges in long sequences.')
else:
    print('VARIES across loads:')
    for t in sorted(unique):
        print(f'  {repr(t)}: {tokens.count(t)} times')
    print('=> Metal FP nondeterminism is measurable at single-token level!')
" | tee -a "$SUMMARY"
echo "Results: $RD/" | tee -a "$SUMMARY"
