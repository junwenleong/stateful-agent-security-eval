#!/bin/bash
# === Top-Tier Sprint — Combined Mac Studio Runner ===
#
# Runs all Day 1-5 experiments sequentially:
#   Phase 1: Payload variants (5 framings × N=10) — ~1h
#   Phase 2: RATG factorial (9 models × 2 defenses × DTA × N=40) — ~2 days
#   Phase 3: 7B Judge (3 models × 2 defenses × DTA × N=40) — ~8h
#
# Prerequisites:
#   - Ollama NOT running (this script starts its own)
#   - qwen2.5:7b pulled (for judge): ollama pull qwen2.5:7b
#   - All factorial models already pulled
#
# RUN: bash scripts/run_sprint_mac_studio.sh

set -e
cd "$(dirname "$0")/.."

echo "=============================================="
echo " TOP-TIER SPRINT — $(date)"
echo "=============================================="

# Start Ollama with factorial settings
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
echo "Ollama started (PID=$OLLAMA_PID)"

# ============================================================
# PHASE 1: Payload Variants (5 framings × N=10 on qwen2.5:14b)
# ============================================================
echo ""
echo "=== PHASE 1: Payload Variants (5 framings × N=10) ==="
echo "Started: $(date)"
.venv/bin/python scripts/run_payload_variants.py
echo "Phase 1 complete: $(date)"

# ============================================================
# PHASE 2: RATG Factorial (9 models × ratg + no_defense × DTA × N=40)
# ============================================================
echo ""
echo "=== PHASE 2: RATG Factorial (9 models × 2 defenses × N=40) ==="
echo "Started: $(date)"
.venv/bin/python scripts/run_ratg_factorial.py
echo "Phase 2 complete: $(date)"

# ============================================================
# PHASE 3: 7B Judge (3 models × judge_7b + no_defense × DTA × N=40)
# ============================================================
echo ""
echo "=== PHASE 3: 7B Judge Test (3 models × N=40) ==="
echo "Started: $(date)"

# Ensure qwen2.5:7b is pulled (needed as judge)
echo "Verifying qwen2.5:7b is available..."
ollama pull qwen2.5:7b 2>/dev/null || true

.venv/bin/python scripts/run_judge_7b.py
echo "Phase 3 complete: $(date)"

# ============================================================
# DONE
# ============================================================
kill $OLLAMA_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo " SPRINT COMPLETE — $(date)"
echo "=============================================="
echo "Results:"
echo "  Phase 1: results/payload_variants/"
echo "  Phase 2: results/ratg_factorial/"
echo "  Phase 3: results/judge_7b/"
echo ""
echo "Next: git add results/ && git commit && git push"
