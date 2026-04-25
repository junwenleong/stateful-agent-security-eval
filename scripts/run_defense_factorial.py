#!/usr/bin/env python3
"""Defense Factorial — 9 models × 7 defenses × 2 attacks × N=40 = 5,040 runs.

Models (9, cross-family):
    Phase 1: qwen2.5:14b            (9GB)   — fastest; anchor model
    Phase 2: qwen3.5:9b             (6.6GB) — small Qwen3.5
    Phase 3: qwen3:32b              (20GB)  — Qwen3 mid-size
    Phase 4: qwen2.5:72b            (47GB)  — mid-size Qwen2.5
    Phase 5: qwen3.5:122b           (81GB)  — largest Qwen
    Phase 6: qwq:32b                (19GB)  — reasoning model
    Phase 7: glm-4.7-flash:bf16     (59GB)  — GLM family
    Phase 8: gpt-oss:20b            (13GB)  — OpenAI small
    Phase 9: gpt-oss-safeguard:120b (65GB)  — safety-tuned paradox

Excluded:
    gemma4:31b             — Ollama v0.20.6 runtime regression
    nemotron-cascade-2:30b — weight drift
    gpt-oss:120b           — re-injection confound (S2 exfiltration)
    qwen3.5:35b            — 50% ASR, uninterpretable defense results
    cogito:14b             — S2 task confusion confound

RAG limit: 15 for ALL models (uniform tool contract).
Limit=5 existed only for gemma4's loop prevention; gemma4 is excluded.
All 9 models make ≤6 RAG calls/session in practice.

Usage:
    .venv/bin/python scripts/run_defense_factorial.py --phase 1
    .venv/bin/python scripts/run_defense_factorial.py --phase all
    .venv/bin/python scripts/run_defense_factorial.py --model qwen2.5:14b
    .venv/bin/python scripts/run_defense_factorial.py --phase 1 --dry-run

Ollama serve command (run BEFORE this script):

    OLLAMA_HOST=0.0.0.0:11434 \\
    OLLAMA_CONTEXT_LENGTH=16384 \\
    OLLAMA_NUM_PARALLEL=1 \\
    OLLAMA_MAX_LOADED_MODELS=1 \\
    OLLAMA_KEEP_ALIVE=5m \\
    OLLAMA_FLASH_ATTENTION=1 \\
    ollama serve
"""
import argparse
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

CONFIG = "experiments/configs/defense_factorial.yaml"
OUTPUT_DIR = "results/defense_factorial"

# Phase definitions: which models run in each phase and how many workers
# RAG limit: 15 for ALL phases (uniform tool contract — see rag_tool.py comment)
PHASES = {
    1: {
        "models": ["qwen2.5:14b"],
        "workers": 1,
        "memory_gb": 9,
        "rag_limit": 15,
        "note": "solo — 9GB, ~0.4 days",
    },
    2: {
        "models": ["qwen3.5:9b"],
        "workers": 1,
        "memory_gb": 7,
        "rag_limit": 15,
        "note": "solo — 6.6GB, ~0.3 days",
    },
    3: {
        "models": ["qwen3:32b"],
        "workers": 1,
        "memory_gb": 20,
        "rag_limit": 15,
        "note": "solo — 20GB, ~0.8 days",
    },
    4: {
        "models": ["qwen2.5:72b"],
        "workers": 1,
        "memory_gb": 47,
        "rag_limit": 15,
        "note": "solo — 47GB, ~1.3 days",
    },
    5: {
        "models": ["qwen3.5:122b"],
        "workers": 1,
        "memory_gb": 81,
        "rag_limit": 15,
        "note": "solo — 81GB, ~1.5 days",
    },
    6: {
        "models": ["qwq:32b"],
        "workers": 1,
        "memory_gb": 19,
        "rag_limit": 15,
        "note": "solo — 19GB, ~0.7 days",
    },
    7: {
        "models": ["glm-4.7-flash:q8_0"],
        "workers": 1,
        "memory_gb": 9,
        "rag_limit": 15,
        "note": "solo — ~9GB (q4_K_M), ~1.0 days",
    },
    8: {
        "models": ["gpt-oss:20b"],
        "workers": 1,
        "memory_gb": 13,
        "rag_limit": 15,
        "note": "solo — 13GB, ~0.5 days",
    },
    9: {
        "models": ["gpt-oss-safeguard:120b"],
        "workers": 1,
        "memory_gb": 65,
        "rag_limit": 15,
        "note": "solo — 65GB, ~2.0 days",
    },
}


def _setup_file_logging(output_dir: str) -> None:
    log_path = Path(output_dir) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info("Logging to %s", log_path)


def _print_ollama_reminder(phase: int | None = None) -> None:
    logger.info("=" * 80)
    logger.info("OLLAMA SERVE COMMAND (Mac Studio M3 Ultra 96GB):")
    logger.info("  OLLAMA_HOST=0.0.0.0:11434 \\")
    logger.info("  OLLAMA_CONTEXT_LENGTH=16384 \\")
    logger.info("  OLLAMA_NUM_PARALLEL=1 \\")
    logger.info("  OLLAMA_MAX_LOADED_MODELS=1 \\")
    logger.info("  OLLAMA_KEEP_ALIVE=2m \\")
    logger.info("  OLLAMA_FLASH_ATTENTION=1 \\")
    logger.info("  ollama serve")
    logger.info("")
    logger.info("NUM_PARALLEL=1 required — large models (72-75GB) leave <16GB headroom")
    if phase:
        p = PHASES[phase]
        logger.info("Phase %d: %s | workers=%d | memory=%dGB | RAG_limit=%d",
                    phase, p["note"], p["workers"], p["memory_gb"], p["rag_limit"])
    logger.info("=" * 80)


def _check_connectivity(base_url: str) -> bool:
    import requests
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        logger.info("✓ Ollama reachable at %s", base_url)
        return True
    except Exception as e:
        logger.error("✗ Cannot reach Ollama at %s: %s", base_url, e)
        return False


def _check_models(config, base_url: str) -> list[str]:
    import requests
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        available_models = resp.json().get("models", [])
        available = {m["name"] for m in available_models}
        # Build a name→details map for quantization checks
        details_by_name = {m["name"]: m for m in available_models}
    except Exception as e:
        logger.warning("Could not query model list: %s", e)
        return []

    configured = [m["model_name"] for m in config.models]
    missing = [m for m in configured if m not in available]
    if missing:
        logger.warning("Models NOT in Ollama (need pull): %s", missing)
    else:
        logger.info("✓ All %d models found in Ollama", len(configured))

    # Q2: Warn if the pulled model's quantization doesn't match the config.
    # Ollama model names include the quantization tag when explicitly pulled
    # (e.g. "qwen2.5:14b:q8_0"). If the config specifies q4_0 but the user
    # pulled q8_0, results will differ from documented conditions.
    for model_cfg in config.models:
        name = model_cfg["model_name"]
        expected_quant = model_cfg.get("ollama_quantization")
        if not expected_quant or name not in details_by_name:
            continue
        # Check if a variant with a different quantization tag is present
        # (e.g. "qwen2.5:14b:q8_0" when we expect "qwen2.5:14b" = q4_0 default)
        for available_name in available:
            if available_name.startswith(name + ":") and expected_quant not in available_name:
                logger.warning(
                    "Quantization mismatch for %s: config expects %s but found %s in Ollama. "
                    "Re-pull with: ollama pull %s:%s",
                    name, expected_quant, available_name, name, expected_quant,
                )

    return missing


def _capture_ollama_hashes(base_url: str) -> dict:
    try:
        import requests
        resp = requests.get(f"{base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        return {m["name"]: m.get("digest", "")[:12] for m in resp.json().get("models", [])}
    except Exception as e:
        logger.warning("Could not capture Ollama model list: %s", e)
        return {}


def _capture_ollama_version(base_url: str) -> str:
    """Capture Ollama runtime version for reproducibility metadata.

    Runtime version is critical for reproducibility — Ollama auto-updates silently
    and renderer/parser changes can alter tool-call behavior without changing model
    weights. gemma4:31b failed after Ollama v0.20.3→v0.20.6 auto-update despite
    identical blob digest. Always record this alongside model hashes.
    """
    try:
        import requests
        resp = requests.get(f"{base_url.rstrip('/')}/api/version", timeout=10)
        resp.raise_for_status()
        return resp.json().get("version", "unknown")
    except Exception as e:
        logger.warning("Could not capture Ollama version: %s", e)
        # Fallback: try subprocess
        try:
            import subprocess
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
            return result.stdout.strip().replace("ollama version is ", "")
        except Exception:
            return "unknown"


def _filter_config_by_models(config, model_names: list[str]):
    """Return a copy of config filtered to the given models."""
    import copy
    cfg = copy.deepcopy(config)
    cfg.models = [m for m in cfg.models if m["model_name"] in model_names]
    return cfg


def _validate(results: list[dict]) -> dict:
    """Print per-model, per-defense breakdown."""
    errors = [r for r in results if r.get("error")]
    dta = [r for r in results if
           r.get("condition", {}).get("attack", {}).get("type") == "delayed_trigger"
           and not r.get("error")]
    no_atk = [r for r in results if
              r.get("condition", {}).get("attack", {}).get("type") == "no_attack"
              and not r.get("error")]

    logger.info("=" * 100)
    logger.info("DEFENSE FACTORIAL RESULTS SUMMARY")
    logger.info("=" * 100)
    logger.info("Total runs: %d | Errors: %d (%.0f%%)",
                len(results), len(errors), len(errors) / max(len(results), 1) * 100)

    models = sorted(set(
        r.get("condition", {}).get("model", {}).get("model_name", "?") for r in results
    ))
    defenses = ["no_defense", "minimizer", "sanitizer", "prompt_hardening",
                "rag_sanitizer", "memory_sandbox", "rag_llm_judge"]

    per_model = {}
    for model in models:
        logger.info("")
        logger.info("Model: %s", model)
        logger.info("  %-25s %10s %10s %10s %10s", "Defense", "Inj%", "ASR%", "BTCR%", "N(DTA)")
        logger.info("  " + "-" * 65)

        per_model[model] = {}
        for defense in defenses:
            m_dta = [r for r in dta if
                     r.get("condition", {}).get("model", {}).get("model_name") == model and
                     r.get("condition", {}).get("defense", {}).get("name") == defense]
            m_no_atk = [r for r in no_atk if
                        r.get("condition", {}).get("model", {}).get("model_name") == model and
                        r.get("condition", {}).get("defense", {}).get("name") == defense]

            n_dta = len(m_dta)
            if n_dta == 0:
                continue
            inj = sum(1 for r in m_dta if r.get("injection_success")) / n_dta * 100
            asr = sum(1 for r in m_dta if r.get("attack_success")) / n_dta * 100
            n_no_atk = len(m_no_atk)
            btcr = sum(1 for r in m_no_atk if r.get("btcr_success")) / max(n_no_atk, 1) * 100

            per_model[model][defense] = {"injection": inj, "asr": asr, "btcr": btcr, "n_dta": n_dta}
            logger.info("  %-25s %9.0f%% %9.0f%% %9.0f%% %10d", defense, inj, asr, btcr, n_dta)

    logger.info("=" * 100)
    return {"total": len(results), "errors": len(errors), "per_model": per_model}


def _run_phase(phase_num: int, config, dry_run: bool, results_file: Path, base_url: str) -> list[dict]:
    """Run a single phase with its model set.

    RAG limit is uniform=15 for all phases (set in PHASES dict).
    """
    p = PHASES[phase_num]
    phase_config = _filter_config_by_models(config, p["models"])
    if not phase_config.models:
        logger.warning("Phase %d: no models found in config, skipping.", phase_num)
        return []

    # Apply model-conditional RAG limit to the attack config
    rag_limit = p.get("rag_limit", 5)
    for attack in phase_config.attacks:
        if attack.get("type") == "delayed_trigger":
            attack["rag_limit"] = rag_limit
            logger.info("Applied RAG_limit=%d to delayed_trigger attack", rag_limit)

    workers = 1 if dry_run else p["workers"]
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE %d: %s", phase_num, p["note"])
    logger.info("Models: %s", p["models"])
    logger.info("Workers: %d | Memory: %dGB | RAG_limit: %d", workers, p["memory_gb"], rag_limit)
    logger.info("=" * 80)

    runner = ParallelExperimentRunner(phase_config, num_workers=workers)
    return runner.run_all(results_path=str(results_file), dry_run=dry_run)


def main():
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Defense Factorial — 9 models × 7 defenses × 2 attacks × N=40 = 5,040 runs"
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "all"],
        help="Phase to run (1-9) or all (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="1 run per condition (smoke test)")
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this specific model (overrides --phase)")
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    _setup_file_logging(OUTPUT_DIR)

    base_url = "http://localhost:11434"
    phase_num = int(args.phase) if args.phase != "all" else None
    _print_ollama_reminder(phase_num)

    if not _check_connectivity(base_url):
        logger.error("Aborting — fix Ollama connectivity first.")
        sys.exit(1)

    config = load_config(args.config)
    _check_models(config, base_url)

    results_file = Path(OUTPUT_DIR) / "results.jsonl"
    all_results = []

    if args.model:
        # Single model override
        filtered = _filter_config_by_models(config, [args.model])
        if not filtered.models:
            logger.error("Model '%s' not found in config.", args.model)
            sys.exit(1)
        # Determine workers from phase membership
        workers = 1
        for p_num, p_def in PHASES.items():
            if args.model in p_def["models"]:
                workers = 1 if args.dry_run else p_def["workers"]
                break
        logger.info("Running single model: %s (workers=%d)", args.model, workers)
        runner = ParallelExperimentRunner(filtered, num_workers=workers)
        all_results = runner.run_all(results_path=str(results_file), dry_run=args.dry_run)

    elif args.phase == "all":
        for p_num in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            results = _run_phase(p_num, config, args.dry_run, results_file, base_url)
            all_results.extend(results)

    else:
        all_results = _run_phase(phase_num, config, args.dry_run, results_file, base_url)

    results_dicts = [r if isinstance(r, dict) else r.__dict__ for r in all_results]
    summary = _validate(results_dicts)

    ollama_hashes = _capture_ollama_hashes(base_url)
    ollama_version = _capture_ollama_version(base_url)
    metadata = {
        "config": args.config,
        "device": "Mac Studio M3 Ultra 96GB",
        "ollama_host": base_url,
        "ollama_version": ollama_version,
        "phase": args.phase,
        "model_filter": args.model,
        "dry_run": args.dry_run,
        "total_runs": summary["total"],
        "errors": summary["errors"],
        "per_model": summary["per_model"],
        "ollama_models": ollama_hashes,
        "phases": PHASES,
    }
    (Path(OUTPUT_DIR) / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Done. Results: %s", results_file)


if __name__ == "__main__":
    main()
