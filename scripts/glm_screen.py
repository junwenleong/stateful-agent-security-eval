import sys
sys.path.insert(0, ".")
from scripts.run_sandbox_inversion import _run_condition
import yaml

config = yaml.safe_load(open("experiments/configs/sandbox_inversion_study.yaml"))
config["models"].append({
    "provider": "ollama",
    "model_name": "glm-4.7-flash",
    "group": "reasoning",
    "base_model": "GLM-4.7-Flash",
    "role": "screen",
    "rag_session_query_limit": 15,
    "rate_limit": {"max_concurrent": 1, "rpm_limit": 999},
    "think": False,
})
_run_condition(config, "glm-4.7-flash", "no_defense", 10, think=False)
