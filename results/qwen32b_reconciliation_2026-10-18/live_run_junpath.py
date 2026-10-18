"""June sandbox_inversion path for one qwen3:32b cell, mirroring scripts/run_sandbox_inversion.py::_run_condition
(condition built from the sandbox_inversion_study.yaml model entry + DEFENSE_CONFIGS + attacks[0], then
ExperimentRunner._run_single), writing ONLY to a new results file. Only base_url is injected (tunnel plumbing).

usage: live_run_junpath.py DEFENSE_NAME N OUT_JSONL THINK(true|false)
"""
import json
import os
import sys
import time
from dataclasses import asdict

import yaml

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from run_sandbox_inversion import DEFENSE_CONFIGS  # noqa: E402  (the June script's own defence definitions)
from src.runner.config_loader import ExperimentConfig  # noqa: E402
from src.runner.runner import ExperimentRunner  # noqa: E402

defense_name, n, out, think = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4].lower() == "true"
config = yaml.safe_load(open("experiments/configs/sandbox_inversion_study.yaml"))
model_cfg = next(m for m in config["models"] if m["model_name"] == "qwen3:32b" and m.get("think", False) == think)
model_cfg = dict(model_cfg)
model_cfg["think"] = think
model_cfg["base_url"] = os.environ["OLLAMA_BASE_URL"]  # plumbing only
condition = {"model": model_cfg, "defense": DEFENSE_CONFIGS[defense_name], "attack": config["attacks"][0]}
runner = ExperimentRunner(ExperimentConfig(
    attacks=config["attacks"], defenses=config.get("defenses", []), models=config["models"],
    runs_per_condition=n, comparisons=[], db_base_dir=config.get("db_base_dir", "data/runs"), results_path=out))
for i in range(n):
    res = asdict(runner._run_single(condition, i))
    with open(out, "a") as f:
        f.write(json.dumps(res, default=str) + "\n")
    print("done run", i, time.strftime("%T"), flush=True)
