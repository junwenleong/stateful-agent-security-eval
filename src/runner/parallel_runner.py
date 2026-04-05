"""Parallel experiment runner for multi-core execution (Mac Studio, etc.)."""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from src.runner.config_loader import ExperimentConfig
from src.runner.runner import ExperimentRunner, RunResult

logger = logging.getLogger(__name__)


def _run_condition_batch_star(args: tuple) -> list[dict]:
    """Unpack args tuple for imap_unordered compatibility."""
    return _run_condition_batch(*args)


def _run_condition_batch(
    config_dict: dict,
    condition: dict,
    runs_per_condition: int,
    results_path: str,
    worker_id: int,
) -> list[dict]:
    """Worker process: run all trials for a single condition.
    
    Each worker owns one condition (attack × defense × model). With one worker
    per model, Ollama never needs to swap models mid-run.
    """
    # Configure logging in worker process
    worker_logger = logging.getLogger()
    if not worker_logger.handlers:
        fmt = logging.Formatter(
            f"%(asctime)s [W{worker_id}] %(name)s %(levelname)s %(message)s"
        )
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        worker_logger.addHandler(sh)
        worker_logger.setLevel(logging.INFO)
    
    config = ExperimentConfig(**config_dict)
    runner = ExperimentRunner(config)
    
    attack_type = condition.get("attack", {}).get("type", "unknown")
    defense_type = condition.get("defense", {}).get("type", "unknown")
    model_name = condition.get("model", {}).get("model_name", "unknown")
    worker_logger.info(
        "[Worker %d] Starting condition: attack=%s, defense=%s, model=%s (%d runs)",
        worker_id, attack_type, defense_type, model_name, runs_per_condition
    )
    
    results = []
    for run_idx in range(runs_per_condition):
        import uuid
        run_id = str(uuid.uuid4())
        try:
            result = runner._run_single(condition, run_id)
            results.append(asdict(result))
            worker_logger.info(
                "[Worker %d] Completed run %d/%d - attack_success=%s, btcr_success=%s",
                worker_id, run_idx + 1, runs_per_condition,
                result.attack_success, result.btcr_success
            )
        except Exception as e:
            worker_logger.error("[Worker %d] Run %d failed: %s", worker_id, run_idx + 1, str(e), exc_info=True)
            results.append({
                "run_id": run_id,
                "condition": condition,
                "error": str(e),
                "attack_success": False,
                "btcr_success": False,
                "btcr_mean_session": 0.0,
                "injection_success": None,
                "tool_logs": [],
                "timing_ms": 0.0,
                "temperature_used": 0.0,
                "defense_logs": [],
                "rag_logs": [],
                "agent_logs": [],
                "mechanistic_tags": {},
                "btcr_success_under_attack": None,
                "btcr_mean_under_attack": 0.0,
                "exfiltration_session_index": None,
                "injection_session_memory_calls": 0,
                "rag_called_in_injection": None,
                "memory_recalled_in_trigger": None,
                "exfiltration_recipient": None,
                "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
    
    return results


class ParallelExperimentRunner:
    """Run factorial experiment with multiprocessing parallelization.
    
    Distributes conditions across worker processes. Each worker runs all trials
    for its assigned condition sequentially (to avoid SQLite contention).
    
    Example:
        config = load_config("experiments/configs/default_factorial.yaml")
        runner = ParallelExperimentRunner(config, num_workers=8)
        results = runner.run_all(results_path="results/factorial/results.jsonl")
    """
    
    def __init__(self, config: ExperimentConfig, num_workers: Optional[int] = None):
        """Initialize parallel runner.
        
        Args:
            config: ExperimentConfig
            num_workers: Number of worker processes. If None, uses CPU count.
        """
        self.config = config
        self.num_workers = num_workers or mp.cpu_count()
        logger.info("Initialized ParallelExperimentRunner with %d workers", self.num_workers)
    
    def run_all(self, results_path: str = "", dry_run: bool = False) -> list[RunResult]:
        """Run all conditions in parallel, with resume support.
        
        Already-completed conditions (by condition hash) are skipped.
        """
        if not results_path:
            results_path = self.config.results_path
        
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        
        conditions = self._enumerate_conditions()
        runs_per = 1 if dry_run else self.config.runs_per_condition
        total_runs = len(conditions) * runs_per

        # Resume: load already-completed runs and skip finished conditions
        completed_results = self._load_partial_results(results_path)
        import hashlib
        def _condition_id(cond: dict) -> str:
            import json as _json
            return hashlib.sha256(_json.dumps(cond, sort_keys=True, default=str).encode()).hexdigest()[:16]

        completion_count: dict[str, int] = {}
        for r in completed_results:
            if not r.get("error"):
                cid = _condition_id(r.get("condition", {}))
                completion_count[cid] = completion_count.get(cid, 0) + 1

        pending_conditions = [
            c for c in conditions
            if completion_count.get(_condition_id(c), 0) < runs_per
        ]
        pending_runs_per = {
            _condition_id(c): runs_per - completion_count.get(_condition_id(c), 0)
            for c in pending_conditions
        }

        logger.info("=" * 80)
        logger.info("PARALLEL EXPERIMENT: %d conditions × %d runs = %d total | %d already done | %d pending",
                   len(conditions), runs_per, total_runs, len(completed_results), 
                   sum(pending_runs_per.values()))
        logger.info("Using %d worker processes", self.num_workers)
        logger.info("=" * 80)

        if not pending_conditions:
            logger.info("All conditions already complete — nothing to run.")
            return completed_results
        
        config_dict = {
            "attacks": self.config.attacks,
            "defenses": self.config.defenses,
            "models": self.config.models,
            "runs_per_condition": runs_per,
            "results_path": results_path,
            "db_base_dir": self.config.db_base_dir,
            "effect_size": self.config.effect_size,
            "alpha": self.config.alpha,
            "power": self.config.power,
            "n_bootstrap": self.config.n_bootstrap,
            "bootstrap_seed": self.config.bootstrap_seed,
            "injection_similarity_threshold": self.config.injection_similarity_threshold,
            "detection": self.config.detection,
            "btcr_criteria": self.config.btcr_criteria,
            "comparisons": self.config.comparisons,
        }
        
        experiment_start = time.monotonic()
        all_results = list(completed_results)
        error_count = 0
        attack_success_by_type = {}
        
        with mp.Pool(processes=self.num_workers) as pool:
            worker_args = [
                (config_dict, condition, pending_runs_per[_condition_id(condition)],
                 results_path, worker_id % self.num_workers)
                for worker_id, condition in enumerate(pending_conditions)
            ]
            for batch_results in pool.imap_unordered(
                _run_condition_batch_star, worker_args, chunksize=1
            ):
                try:
                    for result_dict in batch_results:
                        all_results.append(result_dict)
                        if result_dict.get("error"):
                            error_count += 1
                        else:
                            attack_type = result_dict["condition"].get("attack", {}).get("type", "unknown")
                            defense_type = result_dict["condition"].get("defense", {}).get("type", "unknown")
                            key = f"{attack_type}_{defense_type}"
                            if key not in attack_success_by_type:
                                attack_success_by_type[key] = {"success": 0, "total": 0}
                            attack_success_by_type[key]["total"] += 1
                            if result_dict.get("attack_success"):
                                attack_success_by_type[key]["success"] += 1
                        self._append_result_to_jsonl(result_dict, results_path)

                    completed = len(all_results)
                    elapsed_sec = time.monotonic() - experiment_start
                    new_done = completed - len(completed_results)
                    progress_pct = completed / total_runs * 100
                    avg = elapsed_sec / max(new_done, 1)
                    est_remaining = avg * (total_runs - completed) / 3600.0
                    logger.info("[PROGRESS] %.0f%% | %d/%d runs | Est. remaining: %.1fh",
                                progress_pct, completed, total_runs, est_remaining)
                except Exception as e:
                    logger.error("Condition result collection failed: %s", e)
                    error_count += 1
        
        logger.info("=" * 80)
        logger.info("COMPLETE: %d runs, %d errors", len(all_results), error_count)
        if attack_success_by_type:
            for key, stats in sorted(attack_success_by_type.items()):
                if stats["total"] > 0:
                    logger.info("  %s: %.0f%% ASR (%d/%d)", key,
                                stats["success"] / stats["total"] * 100,
                                stats["success"], stats["total"])
        logger.info("=" * 80)
        
        return all_results
    
    def _enumerate_conditions(self) -> list[dict]:
        """Generate all condition combinations."""
        from itertools import product
        conditions = []
        for attack, defense, model in product(
            self.config.attacks, self.config.defenses, self.config.models
        ):
            conditions.append({"attack": attack, "defense": defense, "model": model})
        return conditions
    
    def _load_partial_results(self, path: str) -> list[dict]:
        """Load already-completed results from JSONL for resume support."""
        p = Path(path)
        if not p.exists():
            return []
        results = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if results:
            logger.info("Resuming: loaded %d existing results from %s", len(results), path)
        return results

    def _append_result_to_jsonl(self, result: dict, path: str) -> None:
        """Append result to JSONL file."""
        with open(path, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")
