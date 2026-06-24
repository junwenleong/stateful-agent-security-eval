"""Test qwq:32b with OLLAMA_FLASH_ATTENTION=0 to check if Draft-Only returns."""
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
config.runs_per_condition = 3
config.results_path = f"{OUT}/results.jsonl"
config.models = [m for m in config.models if m["model_name"] == "qwq:32b"]
config.defenses = [d for d in config.defenses if d.get("name") == "no_defense"]
config.attacks = [a for a in config.attacks if a.get("type") == "delayed_trigger"]

runner = ExperimentRunner(config)
runner.run_all()

# Print results
print("\n=== RESULTS ===")
for line in open(f"{OUT}/results.jsonl"):
    r = json.loads(line)
    if not r.get("error"):
        print(f"  ASR={r['attack_success']}  inj={r['injection_success']}")
    else:
        print(f"  ERROR: {r['error'][:80]}")
