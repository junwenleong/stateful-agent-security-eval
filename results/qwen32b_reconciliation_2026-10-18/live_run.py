"""Run one qwen3:32b cell through the repo's own ExperimentRunner (April factorial path by default).

usage: live_run.py CONFIG DEFENSE_NAME N OUT_JSONL [THINK]
Environment: OLLAMA_BASE_URL (tunnel), EVAL_OVERRIDE_DATE (date string put in the calendar tool description).
"""
import sys

sys.path.insert(0, ".")
from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner

cfg_path, defense, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
think = None if len(sys.argv) < 6 else (sys.argv[5].lower() == "true")

config = load_config(cfg_path)
config.runs_per_condition = n
config.results_path = out
config.models = [
    m for m in config.models
    if m["model_name"] == "qwen3:32b" and (think is None or m.get("think", False) == think)
]
config.defenses = [d for d in config.defenses if d.get("name") == defense]
config.attacks = [a for a in config.attacks if a.get("type") == "delayed_trigger"]
assert len(config.models) == 1 and len(config.defenses) == 1 and len(config.attacks) == 1, (
    len(config.models), len(config.defenses), len(config.attacks))
ExperimentRunner(config).run_all()
