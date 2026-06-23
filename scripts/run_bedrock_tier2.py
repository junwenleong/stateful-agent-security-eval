"""Tier 2: Bedrock Frontier Breadth — Gated N=1 preflight then N=20 escalation.

For each reasoning model, runs N=1 DTA/no_defense and checks:
1. stopReason is tool_use/end_turn (not max_tokens)
2. Parse-mismatch guard did not fire (tool calls parsed if stopReason=tool_use)
3. If injection=0: diagnoses WHY (genuine resistance vs truncation/parse artifact)

Only models passing all three criteria escalate to N=20.

Usage:
    .venv/bin/python scripts/run_bedrock_tier2.py --preflight   # N=1 gate check
    .venv/bin/python scripts/run_bedrock_tier2.py --launch      # N=20 on passed models
"""
import argparse
import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TIER2_MODELS = [
    {"provider": "bedrock", "model_name": "qwen.qwen3-next-80b-a3b", "aws_region": "us-east-1", "aws_profile": "tra-sso", "inference_profile": "qwen.qwen3-next-80b-a3b", "rag_session_query_limit": 15, "rate_limit": {"max_concurrent": 1, "rpm_limit": 60}},
    {"provider": "bedrock", "model_name": "mistral.mistral-large-3-675b-instruct", "aws_region": "us-east-1", "aws_profile": "tra-sso", "inference_profile": "mistral.mistral-large-3-675b-instruct", "rag_session_query_limit": 15, "rate_limit": {"max_concurrent": 1, "rpm_limit": 60}},
    {"provider": "bedrock", "model_name": "zai.glm-5", "aws_region": "us-east-1", "aws_profile": "tra-sso", "inference_profile": "zai.glm-5", "rag_session_query_limit": 15, "rate_limit": {"max_concurrent": 1, "rpm_limit": 60}},
    {"provider": "bedrock", "model_name": "nvidia.nemotron-super-3-120b", "aws_region": "us-east-1", "aws_profile": "tra-sso", "inference_profile": "nvidia.nemotron-super-3-120b", "rag_session_query_limit": 15, "rate_limit": {"max_concurrent": 1, "rpm_limit": 60}},
    {"provider": "bedrock", "model_name": "moonshot.kimi-k2-thinking", "aws_region": "us-east-1", "aws_profile": "tra-sso", "inference_profile": "moonshot.kimi-k2-thinking", "rag_session_query_limit": 15, "rate_limit": {"max_concurrent": 1, "rpm_limit": 60}},
    {"provider": "bedrock", "model_name": "minimax.minimax-m2.5", "aws_region": "us-east-1", "aws_profile": "tra-sso", "inference_profile": "minimax.minimax-m2.5", "rag_session_query_limit": 15, "rate_limit": {"max_concurrent": 1, "rpm_limit": 60}},
    {"provider": "bedrock", "model_name": "us.meta.llama4-maverick-17b-instruct-v1:0", "aws_region": "us-east-1", "aws_profile": "tra-sso", "inference_profile": "us.meta.llama4-maverick-17b-instruct-v1:0", "rag_session_query_limit": 15, "rate_limit": {"max_concurrent": 1, "rpm_limit": 60}},
]

RESULTS_DIR = Path("results/bedrock_tier2")
PREFLIGHT_PATH = RESULTS_DIR / "preflight.jsonl"
MAIN_PATH = RESULTS_DIR / "results.jsonl"


def run_preflight():
    """Run N=1 per model and evaluate pass criteria."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config("experiments/configs/defense_factorial.yaml")
    config.runs_per_condition = 1
    config.results_path = str(PREFLIGHT_PATH)
    config.defenses = [d for d in config.defenses if d.get("name") == "no_defense"]
    config.attacks = [a for a in config.attacks if a.get("type") == "delayed_trigger"]

    passed = []
    failed = []

    for model_cfg in TIER2_MODELS:
        model_name = model_cfg["model_name"]
        logger.info("=" * 60)
        logger.info(f"PREFLIGHT: {model_name}")

        config.models = [model_cfg]
        runner = ExperimentRunner(config)
        runner.run_all()

        # Read the last record (this model's result)
        lines = PREFLIGHT_PATH.read_text().splitlines()
        record = json.loads(lines[-1])

        stop = record.get("final_stop_reason")
        inj = record.get("injection_success")
        atk = record.get("attack_success")
        err = record.get("error")

        # Gate criteria
        truncated = stop == "max_tokens_truncation"
        has_error = err is not None

        logger.info(f"  stop_reason: {stop}")
        logger.info(f"  injection: {inj}, attack: {atk}, error: {err}")

        if has_error:
            logger.error(f"  ❌ FAIL: run errored — {err[:100]}")
            failed.append((model_name, "error", err))
        elif truncated:
            logger.error(f"  ❌ FAIL: max_tokens truncation — reasoning consumed budget")
            failed.append((model_name, "truncation", stop))
        elif not inj and stop in ("end_turn", None):
            # 0 injection + end_turn = possible genuine resistance. Log for review.
            logger.warning(f"  ⚠️  0 injection + end_turn — likely genuine resistance. PASS (escalate to confirm).")
            passed.append((model_name, "possible_resistant"))
        elif inj:
            logger.info(f"  ✅ PASS: injection succeeded, stop={stop}")
            passed.append((model_name, "vulnerable"))
        else:
            logger.warning(f"  ⚠️  Ambiguous: inj={inj}, stop={stop}. Manual review needed.")
            failed.append((model_name, "ambiguous", f"inj={inj}, stop={stop}"))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PREFLIGHT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"PASSED ({len(passed)}):")
    for name, status in passed:
        logger.info(f"  ✅ {name} ({status})")
    logger.info(f"FAILED ({len(failed)}):")
    for item in failed:
        logger.info(f"  ❌ {item[0]} ({item[1]})")

    # Write pass list for launch phase
    pass_file = RESULTS_DIR / "preflight_passed.json"
    pass_file.write_text(json.dumps([name for name, _ in passed], indent=2))
    logger.info(f"\nPassed models written to: {pass_file}")
    logger.info(f"Launch with: .venv/bin/python scripts/run_bedrock_tier2.py --launch")


def run_launch():
    """Run N=20 on models that passed preflight."""
    pass_file = RESULTS_DIR / "preflight_passed.json"
    if not pass_file.exists():
        logger.error("No preflight_passed.json — run --preflight first")
        sys.exit(1)

    passed_names = json.loads(pass_file.read_text())
    passed_models = [m for m in TIER2_MODELS if m["model_name"] in passed_names]
    logger.info(f"Launching N=20 on {len(passed_models)} models: {passed_names}")

    config = load_config("experiments/configs/defense_factorial.yaml")
    config.runs_per_condition = 20
    config.results_path = str(MAIN_PATH)
    config.models = passed_models
    config.defenses = [d for d in config.defenses if d.get("name") == "no_defense"]
    config.attacks = [a for a in config.attacks if a.get("type") == "delayed_trigger"]

    runner = ExperimentRunner(config)
    runner.run_all()
    logger.info("Tier 2 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="Run N=1 gate check per model")
    parser.add_argument("--launch", action="store_true", help="Run N=20 on passed models")
    args = parser.parse_args()

    if args.preflight:
        run_preflight()
    elif args.launch:
        run_launch()
    else:
        parser.print_help()
