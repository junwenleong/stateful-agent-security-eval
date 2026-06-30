"""RATG Patch Run — re-run only the two models affected by config errors.

Fixes applied:
  1. qwen3.5:9b — trigger prompt was missing "Do not ask for clarification;
     use your best judgment to complete the draft." suffix
  2. glm-4.7-flash — was q8_0, should be bf16 (matching N=10 rescreen baseline)

This script runs ONLY these two models under no_defense + DTA (N=40 each = 80 runs).
Results are written to a SEPARATE file to avoid contaminating the main RATG results.
Merge manually after verification.

Usage:
    .venv/bin/python scripts/run_ratg_patch.py
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PATCH_MODELS = {"qwen3.5:9b", "glm-4.7-flash:bf16"}
OUTPUT_PATH = "results/ratg_factorial/patch_rerun.jsonl"


def main():
    config = load_config("experiments/configs/ratg_factorial.yaml")

    # Override results path to avoid contaminating main results
    config.results_path = OUTPUT_PATH
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Filter to only the patch models
    original_models = config.models
    config.models = [m for m in original_models if m.get("model_name") in PATCH_MODELS]

    if not config.models:
        logger.error(f"No patch models found! Available: {[m.get('model_name') for m in original_models]}")
        sys.exit(1)

    # Filter to only DTA + both defenses (no_defense and ratg)
    # Both arms need the corrected trigger prompt + correct quantization
    original_attacks = config.attacks
    config.attacks = [a for a in original_attacks if a.get("type") == "delayed_trigger"]

    # Keep both no_defense and ratg — both were affected by the trigger prompt
    original_defenses = config.defenses
    config.defenses = [d for d in original_defenses if d.get("name") in ("no_defense", "ratg")]

    logger.info(f"Patch run: {len(config.models)} models × {len(config.attacks)} attacks × {len(config.defenses)} defenses × N={config.runs_per_condition}")
    logger.info(f"Models: {[m.get('model_name') for m in config.models]}")
    logger.info(f"Output: {OUTPUT_PATH}")
    logger.info(f"Expected: {len(config.models) * len(config.attacks) * len(config.defenses) * config.runs_per_condition} runs")

    runner = ExperimentRunner(config)
    runner.run_all()

    # Summary
    records = [json.loads(l) for l in Path(OUTPUT_PATH).read_text().splitlines() if l.strip()]
    valid = [r for r in records if not r.get("error")]

    print("\n" + "=" * 60)
    print("RATG PATCH RESULTS")
    print("=" * 60)
    for model_name in sorted(PATCH_MODELS):
        runs = [r for r in valid if r.get("condition", {}).get("model", {}).get("model_name") == model_name]
        if not runs:
            print(f"  {model_name:<25} — NO DATA")
            continue
        inj = sum(1 for r in runs if r.get("injection_success"))
        atk = sum(1 for r in runs if r.get("attack_success"))
        btcr = sum(1 for r in runs if r.get("btcr_success_under_attack"))
        n = len(runs)
        print(f"  {model_name:<25} N={n:2d} | inj={inj}/{n} ({100*inj/n:.0f}%) | ASR={atk}/{n} ({100*atk/n:.0f}%) | BTCR={btcr}/{n} ({100*btcr/n:.0f}%)")

    print("\nExpected (if fixes work):")
    print("  qwen3.5:9b     → 100% inj, 100% ASR, 100% BTCR (restored)")
    print("  glm-4.7-flash  → 100% inj, 100% ASR, 100% BTCR (restored)")


if __name__ == "__main__":
    main()
