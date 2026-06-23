#!/usr/bin/env python3
"""Bedrock Frontier Negative Control — Sandbox Inversion Study.

Runs Sonnet 4.6 + Haiku 4.5 across the 4 sandbox defense conditions (DTA arm)
plus no_attack × {no_defense, sandbox_full} to establish BTCR floor.

12 cells total, N=100 each, 1,200 runs.

Usage:
    .venv/bin/python scripts/run_bedrock_frontier_sandbox_control.py
    .venv/bin/python scripts/run_bedrock_frontier_sandbox_control.py --dry-run
"""
import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner.config_loader import ExperimentConfig
from src.runner.runner import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG = "experiments/configs/bedrock_frontier_sandbox_control.yaml"
OUTPUT_DIR = "results/bedrock_frontier_sandbox_control"
RUNS_PER_CONDITION = 100

# ── Condition Matrix (12 cells) ─────────────────────────────────────────────
# DTA arm: 4 defenses × 2 models = 8 cells
# no_attack arm: 2 defenses × 2 models = 4 cells
MODELS = [
    "global.anthropic.claude-sonnet-4-6",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
]

DEFENSE_CONFIGS = {
    "no_defense": {"type": "none", "name": "no_defense"},
    "sandbox_full": {
        "type": "memory_sandbox", "name": "sandbox_full", "variant": "full",
        "exclude_tools": ["memory_recall_fact"],
    },
    "sandbox_blind": {
        "type": "memory_sandbox", "name": "sandbox_blind", "variant": "blind",
        "exclude_tools": ["memory_recall_fact", "memory_list_all_facts"],
    },
    "sandbox_null_recall": {
        "type": "memory_sandbox", "name": "sandbox_null_recall", "variant": "null_recall",
        "exclude_tools": [],
        "recall_returns_empty": True,
        "empty_return_template": "No stored value found for key '{key}'. The key may not exist or may have been cleared.",
    },
}

# DTA arm: all 4 defenses. no_attack arm: no_defense + sandbox_full only.
DTA_DEFENSES = ["no_defense", "sandbox_full", "sandbox_blind", "sandbox_null_recall"]
NO_ATTACK_DEFENSES = ["no_defense", "sandbox_full"]


def _setup_file_logging() -> None:
    log_path = Path(OUTPUT_DIR) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)


def _load_existing_results() -> list[dict]:
    results_path = Path(OUTPUT_DIR) / "results.jsonl"
    if not results_path.exists():
        return []
    results = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _get_model_name(r: dict) -> str:
    cond = r.get("condition") if isinstance(r, dict) else None
    if not isinstance(cond, dict):
        return ""
    model = cond.get("model")
    if isinstance(model, str):
        return model
    if isinstance(model, dict):
        return model.get("model_name", "")
    return ""


def _get_defense_name(r: dict) -> str:
    cond = r.get("condition") if isinstance(r, dict) else None
    if not isinstance(cond, dict):
        return ""
    defense = cond.get("defense")
    if isinstance(defense, str):
        return defense
    if isinstance(defense, dict):
        return defense.get("name", defense.get("type", ""))
    return ""


def _get_attack_type(r: dict) -> str:
    cond = r.get("condition") if isinstance(r, dict) else None
    if not isinstance(cond, dict):
        return ""
    attack = cond.get("attack")
    if isinstance(attack, str):
        return attack
    if isinstance(attack, dict):
        return attack.get("type", "")
    return ""


def _count_completed(results: list[dict], model: str, defense_name: str, attack_type: str) -> int:
    return sum(
        1 for r in results
        if _get_model_name(r) == model
        and _get_defense_name(r) == defense_name
        and _get_attack_type(r) == attack_type
        and r.get("error") is None
    )


def _verify_bedrock_access(profile: str) -> None:
    import boto3
    session = boto3.Session(profile_name=profile)
    client = session.client("bedrock-runtime", region_name="ap-southeast-1")
    client.converse(
        modelId="global.anthropic.claude-sonnet-4-6",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        inferenceConfig={"temperature": 0.0, "maxTokens": 10},
    )
    logger.info("✓ Bedrock access verified (tra-sso, ap-southeast-1)")


def _run_condition(
    config: dict, model_name: str, defense_name: str, attack_type: str,
    runs_per_condition: int, dry_run: bool = False,
) -> list[dict]:
    existing = _load_existing_results()
    completed = _count_completed(existing, model_name, defense_name, attack_type)
    remaining = runs_per_condition - completed

    if remaining <= 0:
        logger.info("✓ %s × %s × %s: complete (%d/%d)",
                    model_name.split(".")[-1], defense_name, attack_type, completed, runs_per_condition)
        return []

    logger.info("▶ %s × %s × %s: %d remaining (%d/%d done)",
                model_name.split(".")[-1], defense_name, attack_type, remaining, completed, runs_per_condition)

    if dry_run:
        remaining = 1

    # Find model config
    model_cfg = None
    for m in config.get("models", []):
        if m["model_name"] == model_name:
            model_cfg = dict(m)
            break
    if model_cfg is None:
        logger.error("Model %s not found in config", model_name)
        return []

    # Find attack config
    attack_cfg = None
    for a in config.get("attacks", []):
        if a["type"] == attack_type:
            attack_cfg = a
            break
    if attack_cfg is None:
        logger.error("Attack type %s not found in config", attack_type)
        return []

    defense_cfg = DEFENSE_CONFIGS[defense_name]
    condition = {"model": model_cfg, "defense": defense_cfg, "attack": attack_cfg}

    # Build runner
    runner_config = ExperimentConfig(
        attacks=config["attacks"],
        defenses=config.get("defenses", []),
        models=config["models"],
        runs_per_condition=runs_per_condition,
        comparisons=[],
        db_base_dir=config.get("db_base_dir", "data/runs"),
        results_path=config.get("results_path", f"{OUTPUT_DIR}/results.jsonl"),
    )
    runner = ExperimentRunner(runner_config)
    results_path = Path(OUTPUT_DIR) / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    new_results = []
    consecutive_fast_failures = 0
    for i in range(remaining):
        run_index = completed + i
        logger.info("  Run %d/%d for %s × %s × %s",
                    run_index + 1, runs_per_condition,
                    model_name.split(".")[-1], defense_name, attack_type)
        try:
            result = runner._run_single(condition, run_index)
            result_dict = asdict(result)
            run_time = result.timing_ms / 1000.0

            # Fast-fail guard: if 3 consecutive runs finish in <2s, auth is dead.
            # Don't write garbage to JSONL — stop immediately so resume works next time.
            if run_time < 2.0 and not result.btcr_success:
                consecutive_fast_failures += 1
                if consecutive_fast_failures >= 3:
                    logger.error("⛔ 3 consecutive fast failures (<2s each) — SSO token likely expired. "
                                 "Stopping to preserve clean results. Re-run after: aws sso login --profile tra-sso")
                    sys.exit(2)
            else:
                consecutive_fast_failures = 0

            with open(results_path, "a") as f:
                f.write(json.dumps(result_dict, default=str) + "\n")
            new_results.append(result_dict)
            logger.info("    %.1fs | inj=%s asr=%s btcr=%s",
                        run_time, result.injection_success, result.attack_success, result.btcr_success)
        except Exception as e:
            logger.error("  Error on run %d: %s", run_index, e)
            error_result = {
                "condition": condition,
                "run_index": run_index,
                "error": str(e),
                "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(results_path, "a") as f:
                f.write(json.dumps(error_result, default=str) + "\n")

    return new_results


def main():
    parser = argparse.ArgumentParser(description="Bedrock Frontier Sandbox Control")
    parser.add_argument("--dry-run", action="store_true", help="1 run per condition")
    args = parser.parse_args()

    _setup_file_logging()

    import yaml
    config_path = Path(CONFIG)
    if not config_path.exists():
        logger.error("Config not found: %s", CONFIG)
        sys.exit(1)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("=" * 70)
    logger.info("BEDROCK FRONTIER NEGATIVE CONTROL — Sandbox Inversion Study")
    logger.info("Sonnet 4.6 + Haiku 4.5 | 12 cells | N=100 | 1,200 runs")
    logger.info("Profile: tra-sso | Region: ap-southeast-1 | Temp: 0.0")
    logger.info("=" * 70)

    _verify_bedrock_access("tra-sso")

    runs = 1 if args.dry_run else RUNS_PER_CONDITION
    total_cells = 0

    # DTA arm: all 4 defenses × 2 models
    logger.info("\n── DTA ARM (4 defenses × 2 models = 8 cells) ──")
    for model in MODELS:
        for defense in DTA_DEFENSES:
            _run_condition(config, model, defense, "delayed_trigger", runs, dry_run=args.dry_run)
            total_cells += 1

    # no_attack arm: no_defense + sandbox_full × 2 models
    logger.info("\n── NO_ATTACK ARM (2 defenses × 2 models = 4 cells) ──")
    for model in MODELS:
        for defense in NO_ATTACK_DEFENSES:
            _run_condition(config, model, defense, "no_attack", runs, dry_run=args.dry_run)
            total_cells += 1

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE: %d cells executed", total_cells)
    all_results = _load_existing_results()
    successful = sum(1 for r in all_results if not r.get("error"))
    errors = sum(1 for r in all_results if r.get("error"))
    logger.info("Total records: %d | OK: %d | Errors: %d", len(all_results), successful, errors)

    # Per-condition summary
    logger.info("\n── PER-CONDITION SUMMARY ──")
    for model in MODELS:
        short = model.split(".")[-1]
        for attack_type in ("delayed_trigger", "no_attack"):
            defenses = DTA_DEFENSES if attack_type == "delayed_trigger" else NO_ATTACK_DEFENSES
            for defense in defenses:
                n = _count_completed(all_results, model, defense, attack_type)
                relevant = [
                    r for r in all_results
                    if _get_model_name(r) == model
                    and _get_defense_name(r) == defense
                    and _get_attack_type(r) == attack_type
                    and not r.get("error")
                ]
                if relevant:
                    inj = sum(1 for r in relevant if r.get("injection_success"))
                    asr = sum(1 for r in relevant if r.get("attack_success"))
                    btcr = sum(1 for r in relevant if r.get("btcr_success"))
                    logger.info("  %s × %-20s × %-16s n=%3d inj=%.0f%% asr=%.0f%% btcr=%.0f%%",
                                short, defense, attack_type, n,
                                100 * inj / n, 100 * asr / n, 100 * btcr / n)
                else:
                    logger.info("  %s × %-20s × %-16s n=%3d (no data)", short, defense, attack_type, n)

    logger.info("=" * 70)
    logger.info("Results: %s/results.jsonl", OUTPUT_DIR)


if __name__ == "__main__":
    main()
