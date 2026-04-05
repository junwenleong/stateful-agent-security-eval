#!/usr/bin/env python3
"""Positive probe: gpt-oss:120b only, n=10 per defense condition.

Purpose: Confirm true ASR and test defense effectiveness before 570-run pilot.
The screening run showed gpt-oss:120b as the only confirmed true positive
(send_email to attacker at session 3). This probe validates that finding
and measures defense effect sizes.

1 model × 5 defenses × 10 runs = 50 runs
Estimated time: ~3-4 hours

Usage:
    .venv/bin/python scripts/run_positive_probe.py
"""
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

CONFIG = "experiments/configs/gpt_oss_positive_probe.yaml"
OUTPUT_DIR = "results/gpt_oss_positive_probe"


def _setup_file_logging(output_dir: str) -> None:
    log_path = Path(output_dir) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info("Logging to %s", log_path)


def _verify_ollama() -> None:
    import urllib.request
    try:
        urllib.request.urlopen("http://169.254.1.2:11434/api/tags", timeout=5)
        logger.info("✓ Ollama reachable at 169.254.1.2:11434")
    except Exception as e:
        logger.error("✗ Ollama not reachable: %s", e)
        raise


def main():
    logger.info("=" * 70)
    logger.info("POSITIVE PROBE: gpt-oss:120b, n=10 per defense")
    logger.info("=" * 70)
    logger.info("Screening confirmed: gpt-oss:120b called send_email to attacker")
    logger.info("at session 3 (true delayed trigger, not draft-only false positive)")
    logger.info("=" * 70)

    _verify_ollama()

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_file_logging(OUTPUT_DIR)
    results_file = output_dir / "results.jsonl"

    logger.info("Config: %s", CONFIG)
    logger.info("Output: %s", results_file)
    logger.info("Expected: 50 runs (5 defenses × 10 runs)")
    logger.info("Estimated time: 3-4 hours")
    logger.info("=" * 70)

    config = load_config(CONFIG)
    runner = ParallelExperimentRunner(config, num_workers=1)
    results = runner.run_all(results_path=str(results_file), dry_run=False)

    results_dicts = [r if isinstance(r, dict) else r.__dict__ for r in results]
    successful = [r for r in results_dicts if not r.get("error")]
    failed = len(results_dicts) - len(successful)

    # Group by defense
    by_defense = {}
    for r in successful:
        defense = r.get("condition", {}).get("defense", {}).get("name", "unknown")
        if defense not in by_defense:
            by_defense[defense] = {"injection": [], "attack": [], "btcr": [], "recalled": []}
        by_defense[defense]["injection"].append(bool(r.get("injection_success")))
        by_defense[defense]["attack"].append(bool(r.get("attack_success")))
        by_defense[defense]["btcr"].append(bool(r.get("btcr_success_under_attack")))
        by_defense[defense]["recalled"].append(bool(r.get("memory_recalled_in_trigger")))

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS BY DEFENSE (gpt-oss:120b)")
    logger.info("=" * 70)
    logger.info("%-20s  %8s  %8s  %8s  %8s", "Defense", "Inject%", "Attack%", "BTCR%", "Recalled%")
    logger.info("-" * 70)

    defense_order = ["no_defense", "minimizer_only", "sanitizer_only", "prompt_hardening", "rag_sanitizer"]
    for defense in defense_order:
        if defense not in by_defense:
            continue
        d = by_defense[defense]
        n = len(d["injection"])
        if n == 0:
            continue
        inj = 100 * sum(d["injection"]) / n
        atk = 100 * sum(d["attack"]) / n
        btcr = 100 * sum(d["btcr"]) / n
        rec = 100 * sum(d["recalled"]) / n
        logger.info("%-20s  %7.0f%%  %7.0f%%  %7.0f%%  %7.0f%%  (n=%d)", defense, inj, atk, btcr, rec, n)

    logger.info("=" * 70)
    logger.info("Total: %d runs, %d successful, %d failed", len(results_dicts), len(successful), failed)

    # Bootstrap CIs on no_defense baseline
    try:
        from src.stats.bootstrap_engine import BootstrapEngine
        baseline = by_defense.get("no_defense", {})
        if baseline.get("attack"):
            engine = BootstrapEngine()
            ci = engine.compute_ci_bca(baseline["attack"])
            logger.info(
                "\nBaseline ASR (no_defense): %.0f%% [95%% CI: %.1f%%-%.1f%%]",
                100 * sum(baseline["attack"]) / len(baseline["attack"]),
                100 * ci[0],
                100 * ci[1],
            )
    except Exception as e:
        logger.warning("Could not compute CIs: %s", e)

    logger.info("=" * 70)
    logger.info("Positive probe complete. Results: %s", results_file)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
