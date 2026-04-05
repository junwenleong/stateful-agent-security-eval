#!/usr/bin/env python3
"""Run full Bedrock Claude factorial: N=40 per model.

Tests DTA attack on 2 frontier models via Bedrock.
2 models × 1 attack × 1 defense × 40 runs = 80 runs total.

Usage:
    .venv/bin/python scripts/run_bedrock_factorial.py
    
Requires:
    - AWS credentials configured (SSO)
    - Bedrock access in ap-southeast-1
    - Claude Haiku and Sonnet models enabled
    - IAM permissions: bedrock:InvokeModel
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner.config_loader import load_config
from src.runner.parallel_runner import ParallelExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG = "experiments/configs/bedrock_claude_factorial.yaml"
OUTPUT_DIR = "results/bedrock_claude_factorial"


def _verify_bedrock_access() -> None:
    """Verify Bedrock is accessible."""
    try:
        import boto3
        
        # Use icpo-assistant profile
        session = boto3.Session(profile_name="icpo-assistant")
        client = session.client("bedrock-runtime", region_name="ap-southeast-1")
        
        # Try a simple invoke to verify access with Sonnet 4.6
        response = client.converse(
            modelId="global.anthropic.claude-sonnet-4-6",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 10}
        )
        logger.info("✓ Bedrock access verified (icpo-assistant profile, Sonnet 4.6)")
    except Exception as e:
        logger.error("✗ Bedrock access failed: %s", e)
        raise


def _setup_file_logging(output_dir: str) -> None:
    """Add a file handler so logs survive overnight runs."""
    log_path = Path(output_dir) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info("Logging to %s", log_path)


def main():
    logger.info("="*70)
    logger.info("BEDROCK CLAUDE FACTORIAL: N=40 per model")
    logger.info("="*70)
    
    _verify_bedrock_access()
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_file_logging(OUTPUT_DIR)
    results_file = output_dir / "results.jsonl"
    
    logger.info("Config: %s", CONFIG)
    logger.info("Output: %s", results_file)
    logger.info("Expected: 80 runs (40 Sonnet + 40 Haiku)")
    logger.info("Estimated cost: ~$2-5 USD")
    logger.info("Estimated time: 1-4 hours")
    logger.info("="*70)
    
    config = load_config(CONFIG)
    runner = ParallelExperimentRunner(config, num_workers=1)  # Bedrock is API-bound, use 1 worker
    results = runner.run_all(results_path=str(results_file), dry_run=False)
    
    results_dicts = [r if isinstance(r, dict) else r.__dict__ for r in results]
    successful = sum(1 for r in results_dicts if not r.get("error"))
    failed = len(results_dicts) - successful
    
    # Analyze by model
    models = {}
    for r in results_dicts:
        model_name = r.get("condition", {}).get("model", {}).get("model_name", "unknown")
        if model_name not in models:
            models[model_name] = {"total": 0, "successful": 0, "injection": 0, "attack": 0}
        models[model_name]["total"] += 1
        if not r.get("error"):
            models[model_name]["successful"] += 1
            if r.get("injection_success"):
                models[model_name]["injection"] += 1
            if r.get("attack_success"):
                models[model_name]["attack"] += 1
    
    logger.info("\n" + "="*70)
    logger.info("RESULTS BY MODEL")
    logger.info("="*70)
    
    for model_name, stats in sorted(models.items()):
        if stats["successful"] > 0:
            inj_pct = 100 * stats["injection"] / stats["successful"]
            atk_pct = 100 * stats["attack"] / stats["successful"]
            logger.info(
                "%s: %d/%d successful | injection: %.0f%% | attack: %.0f%%",
                model_name.split(".")[-1],
                stats["successful"],
                stats["total"],
                inj_pct,
                atk_pct
            )
        else:
            logger.info("%s: 0/%d successful (all failed)", model_name.split(".")[-1], stats["total"])
    
    logger.info("="*70)
    logger.info("OVERALL SUMMARY")
    logger.info("="*70)
    logger.info("Total runs: %d", len(results_dicts))
    logger.info("Successful: %d (%.0f%%)", successful, 100 * successful / len(results_dicts) if results_dicts else 0)
    logger.info("Failed: %d (%.0f%%)", failed, 100 * failed / len(results_dicts) if results_dicts else 0)
    logger.info("Results file: %s", results_file)
    logger.info("="*70)
    
    # Calculate confidence intervals
    logger.info("\nCalculating confidence intervals...")
    try:
        from src.stats.bootstrap_engine import BootstrapEngine
        
        for model_name in sorted(models.keys()):
            model_results = [r for r in results_dicts if r.get("condition", {}).get("model", {}).get("model_name") == model_name and not r.get("error")]
            if model_results:
                injection_outcomes = [r.get("injection_success", False) for r in model_results]
                attack_outcomes = [r.get("attack_success", False) for r in model_results]
                
                engine = BootstrapEngine()
                inj_ci = engine.compute_ci_bca(injection_outcomes)
                atk_ci = engine.compute_ci_bca(attack_outcomes)
                
                logger.info(
                    "%s: injection [%.1f%%, %.1f%%] | attack [%.1f%%, %.1f%%]",
                    model_name.split(".")[-1],
                    100 * inj_ci[0],
                    100 * inj_ci[1],
                    100 * atk_ci[0],
                    100 * atk_ci[1]
                )
    except Exception as e:
        logger.warning("Could not compute CIs: %s", e)
    
    logger.info("="*70)
    logger.info("Bedrock factorial complete!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
