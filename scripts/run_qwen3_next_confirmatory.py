#!/usr/bin/env python3
"""qwen3-next N=40 confirmatory: no_defense + memory_sandbox, same N both arms.

Decides: is +10pp (N=10 screen) real inversion, or noise/bypass-ineffective?
"""
import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL = {"provider": "bedrock", "model_name": "qwen.qwen3-next-80b-a3b", "aws_region": "us-east-1", "aws_profile": "tra-sso", "inference_profile": "qwen.qwen3-next-80b-a3b", "rag_session_query_limit": 15, "rate_limit": {"max_concurrent": 1, "rpm_limit": 60}}

RESULTS_DIR = Path("results/qwen3_next_n40_confirmatory")
RESULTS_PATH = RESULTS_DIR / "results.jsonl"


def main():
    import subprocess
    result = subprocess.run(["aws", "sts", "get-caller-identity", "--profile", "tra-sso"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("SSO expired. Run: aws sso login --profile tra-sso")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config("experiments/configs/defense_factorial.yaml")
    config.runs_per_condition = 40
    config.results_path = str(RESULTS_PATH)
    config.models = [MODEL]
    config.defenses = [d for d in config.defenses if d.get("name") in ("no_defense", "memory_sandbox")]
    config.attacks = [a for a in config.attacks if a.get("type") == "delayed_trigger"]

    logger.info("=" * 60)
    logger.info("QWEN3-NEXT N=40 CONFIRMATORY")
    logger.info("Arms: no_defense + memory_sandbox | Attack: DTA | N=40")
    logger.info(f"Results: {RESULTS_PATH}")
    logger.info("=" * 60)

    runner = ExperimentRunner(config)
    runner.run_all()
    logger.info("Done.")


if __name__ == "__main__":
    main()
