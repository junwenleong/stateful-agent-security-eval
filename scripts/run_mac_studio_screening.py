#!/usr/bin/env python3
"""Mac Studio screening run: all 19 models, n=5, delayed_trigger + no_defense only.

Purpose: detect which models show ANY vulnerability before committing to full pilot.
95 runs total. Est. 4-6 hours on M3 Ultra 96GB.

Recommended Ollama config:
    OLLAMA_HOST=0.0.0.0:11434 \\
    OLLAMA_CONTEXT_LENGTH=32768 \\
    OLLAMA_NUM_PARALLEL=2 \\
    OLLAMA_MAX_LOADED_MODELS=1 \\
    OLLAMA_FLASH_ATTENTION=1 \\
    ollama serve

Usage:
    .venv/bin/python scripts/run_mac_studio_screening.py
    .venv/bin/python scripts/run_mac_studio_screening.py --dry-run   # 1 run per model
"""
import argparse
import json
import logging
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG = "experiments/configs/mac_studio_screening.yaml"
OUTPUT_DIR = "results/mac_studio_part1_screening"


def _setup_file_logging(output_dir: str) -> None:
    log_path = Path(output_dir) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info("Logging to %s", log_path)


def _check_ollama(base_url: str) -> bool:
    import requests
    try:
        requests.get(f"{base_url}/api/tags", timeout=5).raise_for_status()
        return True
    except Exception as e:
        logger.error("Cannot reach Ollama at %s: %s", base_url, e)
        return False


def _ollama_model_ids() -> dict:
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        ids = {}
        for line in out.stdout.split("\n")[1:]:
            parts = line.split()
            if len(parts) >= 2:
                ids[parts[0]] = parts[1]
        return ids
    except Exception:
        return {}


def _run_model_sequential(
    config,
    model: dict,
    attack: dict,
    defense: dict,
    n_runs: int,
    results_path: str,
    dry_run: bool,
) -> list[dict]:
    """Run all n_runs for a single model sequentially. Returns list of result dicts."""
    model_name = model.get("model_name", "unknown")
    runs = 1 if dry_run else n_runs
    results = []

    # Build a single-model config for the runner
    from src.runner.config_loader import ExperimentConfig
    single_config = ExperimentConfig(
        attacks=[attack],
        defenses=[defense],
        models=[model],
        runs_per_condition=runs,
        results_path=results_path,
        db_base_dir=config.db_base_dir,
        effect_size=config.effect_size,
        alpha=config.alpha,
        power=config.power,
        n_bootstrap=config.n_bootstrap,
        bootstrap_seed=config.bootstrap_seed,
        injection_similarity_threshold=config.injection_similarity_threshold,
        detection=config.detection,
        btcr_criteria=config.btcr_criteria,
        comparisons=[],
    )
    runner = ExperimentRunner(single_config)
    condition = {"attack": attack, "defense": defense, "model": model}

    for i in range(runs):
        run_id = str(uuid.uuid4())
        try:
            result = runner._run_single(condition, run_id)
            result_dict = asdict(result)
            results.append(result_dict)
            inj = "✓" if result.injection_success else "✗"
            atk = "✓" if result.attack_success else "✗"
            logger.info("[%s] run %d/%d  injection=%s  attack=%s  btcr=%s  %.1fs",
                        model_name, i + 1, runs, inj, atk,
                        "✓" if result.btcr_success else "✗",
                        result.timing_ms / 1000)
        except Exception as e:
            logger.error("[%s] run %d/%d FAILED: %s", model_name, i + 1, runs, e)
            results.append({
                "run_id": run_id,
                "condition": condition,
                "error": str(e),
                "attack_success": False,
                "btcr_success": False,
                "injection_success": None,
                "timing_ms": 0.0,
            })

    # Append to JSONL
    with open(results_path, "a") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    return results


def _print_summary(all_results: list[dict]) -> None:
    """Print per-model summary table."""
    by_model: dict[str, list] = {}
    for r in all_results:
        name = r.get("condition", {}).get("model", {}).get("model_name", "unknown")
        by_model.setdefault(name, []).append(r)

    logger.info("\n" + "=" * 75)
    logger.info("SCREENING RESULTS — delayed_trigger + no_defense")
    logger.info("=" * 75)
    logger.info("%-32s %8s %8s %8s %8s", "Model", "Inj/N", "Inj%", "Atk/N", "Atk%")
    logger.info("-" * 75)

    vulnerable = []
    resistant = []
    errored = []

    for model_name in sorted(by_model):
        runs = by_model[model_name]
        ok = [r for r in runs if not r.get("error")]
        errs = len(runs) - len(ok)
        inj = sum(1 for r in ok if r.get("injection_success"))
        atk = sum(1 for r in ok if r.get("attack_success"))
        n = len(ok)
        inj_pct = inj / n * 100 if n else 0
        atk_pct = atk / n * 100 if n else 0

        err_note = f" ({errs} err)" if errs else ""
        logger.info("%-32s %4d/%-3d %7.0f%% %4d/%-3d %7.0f%%%s",
                    model_name, inj, n, inj_pct, atk, n, atk_pct, err_note)

        if atk > 0:
            vulnerable.append(model_name)
        elif inj > 0:
            resistant.append(f"{model_name} (injects but no attack)")
        elif errs == len(runs):
            errored.append(model_name)
        else:
            resistant.append(model_name)

    logger.info("=" * 75)

    if vulnerable:
        logger.info("\n🔴 VULNERABLE (attack_success > 0) — candidates for full pilot:")
        for m in vulnerable:
            logger.info("   %s", m)

    if resistant:
        logger.info("\n🟢 RESISTANT / LATENT (attack_success = 0):")
        for m in resistant:
            logger.info("   %s", m)

    if errored:
        logger.info("\n⚠️  ALL RUNS ERRORED:")
        for m in errored:
            logger.info("   %s", m)

    logger.info("\nNext step: run full pilot (n=3, all 5 defenses) on vulnerable models.")


def main():
    parser = argparse.ArgumentParser(description="Mac Studio screening: 19 models, n=5, DTA only")
    parser.add_argument("--dry-run", action="store_true", help="1 run per model (smoke test)")
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args()

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_file_logging(OUTPUT_DIR)
    results_path = str(output_dir / "results.jsonl")

    config = load_config(args.config)
    base_url = config.models[0].get("base_url", "http://169.254.1.2:11434")

    logger.info("=" * 75)
    logger.info("MAC STUDIO SCREENING: %d models × %d runs = %d total",
                len(config.models), config.runs_per_condition,
                len(config.models) * (1 if args.dry_run else config.runs_per_condition))
    logger.info("Attack: delayed_trigger | Defense: no_defense | Dry-run: %s", args.dry_run)
    logger.info("=" * 75)

    if not _check_ollama(base_url):
        logger.error("Aborting — Ollama not reachable at %s", base_url)
        sys.exit(1)

    model_ids = _ollama_model_ids()
    logger.info("Ollama model IDs: %s", model_ids)

    attack = config.attacks[0]
    defense = config.defenses[0]
    all_results = []
    start = time.monotonic()

    for i, model in enumerate(config.models):
        model_name = model.get("model_name", "unknown")
        elapsed = time.monotonic() - start
        remaining_models = len(config.models) - i
        avg_per_model = elapsed / max(i, 1)
        est_remaining_h = avg_per_model * remaining_models / 3600

        logger.info("\n[%d/%d] %-32s  elapsed=%.0fm  est_remaining=%.1fh",
                    i + 1, len(config.models), model_name,
                    elapsed / 60, est_remaining_h)

        results = _run_model_sequential(
            config, model, attack, defense,
            config.runs_per_condition, results_path, args.dry_run
        )
        all_results.extend(results)

        # Quick per-model summary after each model completes
        ok = [r for r in results if not r.get("error")]
        if ok:
            inj = sum(1 for r in ok if r.get("injection_success"))
            atk = sum(1 for r in ok if r.get("attack_success"))
            logger.info("  → injection: %d/%d  attack: %d/%d", inj, len(ok), atk, len(ok))

    _print_summary(all_results)

    metadata = {
        "total_runs": len(all_results),
        "successful": sum(1 for r in all_results if not r.get("error")),
        "failed": sum(1 for r in all_results if r.get("error")),
        "ollama_model_ids": model_ids,
        "tested_models": [m.get("model_name") for m in config.models],
        "dry_run": args.dry_run,
        "elapsed_minutes": (time.monotonic() - start) / 60,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("\nResults: %s", results_path)


if __name__ == "__main__":
    main()
