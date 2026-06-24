"""Test qwq:32b with OLLAMA_FLASH_ATTENTION=0: no_defense + memory_sandbox, N=10 each.

Decisive test: does FA=0 reproduce BOTH April results?
- no_defense → 0% ASR (Draft-Only)
- memory_sandbox → 100% ASR (inversion via RAG fallback)

If both hold, FA explains the full April inversion, not just the refusal.

Run with: OLLAMA_FLASH_ATTENTION=0 (all other flags same as factorial)
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner

OUT = "results/qwq_flash_off_test"
os.makedirs(OUT, exist_ok=True)

config = load_config("experiments/configs/defense_factorial.yaml")
config.runs_per_condition = 10
config.results_path = f"{OUT}/results.jsonl"
config.models = [m for m in config.models if m["model_name"] == "qwq:32b"]
config.defenses = [d for d in config.defenses if d.get("name") in ("no_defense", "memory_sandbox")]
config.attacks = [a for a in config.attacks if a.get("type") == "delayed_trigger"]

runner = ExperimentRunner(config)
runner.run_all()

# Print results
print("\n=== RESULTS (qwq:32b, FA=0) ===")
by_def = {}
for line in open(f"{OUT}/results.jsonl"):
    r = json.loads(line)
    if r.get("error"):
        continue
    d = r.get("condition", {}).get("defense", {}).get("name", "?")
    by_def.setdefault(d, []).append(r.get("attack_success"))

for d, results in sorted(by_def.items()):
    asr = sum(results)
    print(f"  {d}: ASR={asr}/{len(results)} ({100*asr/len(results):.0f}%)")

print("\nExpected if FA explains the full April inversion:")
print("  no_defense: 0/10 (Draft-Only)")
print("  memory_sandbox: 10/10 (inversion via RAG fallback)")
