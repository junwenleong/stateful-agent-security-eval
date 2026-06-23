"""N=1 Bedrock field verification: confirm new logging fields populate in RunResult JSONL."""
import sys, json
from pathlib import Path
sys.path.insert(0, '.')
from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner

# Minimal config for one run
config = load_config('experiments/configs/defense_factorial.yaml')
config.runs_per_condition = 1
config.results_path = 'results/bedrock_field_verify/results.jsonl'

# Use gpt-oss-20b (cheapest Tier 1 model)
config.models = [{
    "provider": "bedrock",
    "model_name": "openai.gpt-oss-20b-1:0",
    "aws_region": "us-east-1",
    "aws_profile": "tra-sso",
    "inference_profile": "openai.gpt-oss-20b-1:0",
    "rag_session_query_limit": 15,
    "rate_limit": {"max_concurrent": 1, "rpm_limit": 999},
}]
config.defenses = [d for d in config.defenses if d.get('name') == 'no_defense']
config.attacks = [a for a in config.attacks if a.get('type') == 'delayed_trigger']

print("Running N=1 on openai.gpt-oss-20b-1:0 via Bedrock...")
runner = ExperimentRunner(config)
runner.run_all()

# Verify fields
result_path = Path('results/bedrock_field_verify/results.jsonl')
if result_path.exists():
    record = json.loads(result_path.read_text().splitlines()[0])
    print("\n=== FIELD VERIFICATION ===")
    fields = ['bedrock_model_id', 'bedrock_region', 'bedrock_inference_profile', 'final_stop_reason']
    all_ok = True
    for f in fields:
        val = record.get(f)
        status = "✅" if val is not None else "❌ MISSING"
        print(f"  {f}: {val} {status}")
        if val is None:
            all_ok = False
    print(f"\n{'✅ All fields populated — ready for Tier 1' if all_ok else '❌ Some fields missing — check wiring'}")
    print(f"  attack_success: {record.get('attack_success')}")
    print(f"  injection_success: {record.get('injection_success')}")
else:
    print("❌ No results file produced")
