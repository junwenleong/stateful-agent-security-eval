#!/usr/bin/env python3
"""Bedrock APAC smoke test — v2 rerun (2026-04-21).

Conditions match the defense factorial exactly. See experiments/configs/bedrock_apac_smoke.yaml.

Old results (2026-04-05, pre-v2 codebase):
    results/bedrock_apac_smoke/results_v1_pre_2026-04-11.jsonl  ← archived, do not cite

Usage:
    .venv/bin/python scripts/run_bedrock_apac_smoke.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG = "experiments/configs/bedrock_apac_smoke.yaml"
OUTPUT_DIR = "results/bedrock_apac_smoke"

MODELS = {
    "global.anthropic.claude-sonnet-4-6": "Sonnet 4.6",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": "Haiku 4.5",
}


def _verify_bedrock_access() -> None:
    import boto3
    session = boto3.Session(profile_name="icpo-assistant")
    client = session.client("bedrock-runtime", region_name="ap-southeast-1")
    client.converse(
        modelId="global.anthropic.claude-sonnet-4-6",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        inferenceConfig={"temperature": 0.0, "maxTokens": 10},
    )
    logger.info("✓ Bedrock access verified (ap-southeast-1)")


def _setup_file_logging(output_dir: str) -> None:
    log_path = Path(output_dir) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info("Logging to %s", log_path)


def main():
    logger.info("=" * 70)
    logger.info("BEDROCK APAC SMOKE TEST v2 — conditions match defense factorial")
    logger.info("Sonnet 4.6 + Haiku 4.5 — ap-southeast-1")
    logger.info("2 models × 1 defense × 2 attacks × 100 runs = 400 runs")
    logger.info("=" * 70)

    _verify_bedrock_access()

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_file_logging(OUTPUT_DIR)

    config = load_config(CONFIG)
    results_file = config.results_path

    logger.info("Config:   %s", CONFIG)
    logger.info("Output:   %s", results_file)
    logger.info("=" * 70)

    runner = ExperimentRunner(config)
    results = runner.run_all(results_path=results_file, dry_run=False)

    results_dicts = [r if isinstance(r, dict) else r.__dict__ for r in results]

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    for model_id, label in MODELS.items():
        for attack_type in ("delayed_trigger", "no_attack"):
            ok = [
                r for r in results_dicts
                if r.get("condition", {}).get("model", {}).get("model_name") == model_id
                and r.get("condition", {}).get("attack", {}).get("type") == attack_type
                and not r.get("error")
            ]
            total = sum(
                1 for r in results_dicts
                if r.get("condition", {}).get("model", {}).get("model_name") == model_id
                and r.get("condition", {}).get("attack", {}).get("type") == attack_type
            )
            if ok:
                inj = sum(1 for r in ok if r.get("injection_success"))
                atk = sum(1 for r in ok if r.get("attack_success"))
                btcr = sum(1 for r in ok if r.get("btcr_success"))
                logger.info(
                    "%-12s %-16s %d/%d ok | inj %d/%d (%.0f%%) | asr %d/%d (%.0f%%) | btcr %d/%d (%.0f%%)",
                    label, attack_type,
                    len(ok), total,
                    inj, len(ok), 100 * inj / len(ok),
                    atk, len(ok), 100 * atk / len(ok),
                    btcr, len(ok), 100 * btcr / len(ok),
                )
            else:
                logger.warning("%-12s %-16s 0/%d ok — all failed", label, attack_type, total)

    successful = sum(1 for r in results_dicts if not r.get("error"))
    errors = sum(1 for r in results_dicts if r.get("error"))
    logger.info("=" * 70)
    logger.info("Total: %d runs | %d ok | %d errors", len(results_dicts), successful, errors)
    logger.info("Results: %s", results_file)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
