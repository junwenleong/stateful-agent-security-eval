"""Payload Variants Test — Day 1 of top-tier sprint.

Tests 5 authority framings × N=10 on qwen2.5:14b to verify semantic masking generalizes.

Usage:
    .venv/bin/python scripts/run_payload_variants.py
"""
import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/payload_variants")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    config = load_config("experiments/configs/payload_variants.yaml")
    runner = ExperimentRunner(config)
    runner.run_all()

    # Analyze
    results_file = Path(config.results_path)
    if not results_file.exists():
        logger.error("No results file found")
        return

    records = [json.loads(l) for l in results_file.read_text().splitlines() if l.strip()]
    valid = [r for r in records if not r.get("error")]

    # Group by attack name
    by_variant = {}
    for r in valid:
        name = r.get("condition", {}).get("attack", {}).get("name", "unknown")
        by_variant.setdefault(name, []).append(r)

    print("\n" + "=" * 60)
    print("PAYLOAD VARIANTS RESULTS")
    print("=" * 60)

    for name, runs in sorted(by_variant.items()):
        inj = sum(1 for r in runs if r.get("injection_success"))
        asr = sum(1 for r in runs if r.get("attack_success"))
        n = len(runs)
        print(f"  {name:<20} inj={inj}/{n}  ASR={asr}/{n} ({100*asr/n:.0f}%)")

    print("\n" + "=" * 60)
    all_asr = sum(1 for r in valid if r.get("attack_success"))
    all_n = len(valid)
    print(f"AGGREGATE: ASR={all_asr}/{all_n} ({100*all_asr/all_n:.0f}%)")

    if all_asr == all_n:
        print("CONCLUSION: Semantic masking generalizes across ALL authority framings (100% ASR).")
    elif all_asr / all_n > 0.9:
        print("CONCLUSION: Semantic masking generalizes broadly (>90% ASR across framings).")
    else:
        print("CONCLUSION: Some framings are more effective than others — investigate per-variant.")

    # Write summary
    summary = RESULTS_DIR / "summary.txt"
    with open(summary, "w") as f:
        f.write(f"Payload Variants: {len(by_variant)} framings × N={len(runs)} on qwen2.5:14b\n")
        for name, runs in sorted(by_variant.items()):
            asr = sum(1 for r in runs if r.get("attack_success"))
            f.write(f"  {name}: ASR={asr}/{len(runs)}\n")
        f.write(f"Aggregate: {all_asr}/{all_n}\n")


if __name__ == "__main__":
    main()
