#!/usr/bin/env bash
# Test qwq:32b on April's exact code (ebf2676) to isolate code vs environment.
# Run AFTER test_qwq_version.py N=10 finishes.
# Requires Ollama 0.20.6 running with:
#   OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 \
#   OLLAMA_KEEP_ALIVE=5m OLLAMA_FLASH_ATTENTION=1 /Applications/Ollama.app/Contents/Resources/ollama serve

set -e

cd "$(dirname "$0")/.."

echo "=== Stashing current changes ==="
git stash --include-untracked || true

echo "=== Checking out April code (ebf2676) for src/ ==="
git checkout ebf2676 -- src/agent src/runner src/tools

echo "=== Running N=10 with April code ==="
# Use a separate results path
.venv/bin/python -c "
import sys
sys.path.insert(0, '.')
from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner

config = load_config('experiments/configs/defense_factorial.yaml')
config.runs_per_condition = 10
config.results_path = 'results/qwq_april_code_test/results.jsonl'
config.models = [m for m in config.models if 'qwq' in m.get('model_name','')]
config.defenses = [d for d in config.defenses if d.get('name') == 'no_defense']
config.attacks = [a for a in config.attacks if a.get('type') == 'delayed_trigger']
runner = ExperimentRunner(config)
runner.run_all()
"

echo "=== Restoring current code ==="
git checkout HEAD -- src/agent src/runner src/tools
git stash pop || true

echo "=== Done. Check results/qwq_april_code_test/results.jsonl ==="
echo "=== Then run: ==="
echo "  .venv/bin/python scripts/verify_qwq_determinism.py"
