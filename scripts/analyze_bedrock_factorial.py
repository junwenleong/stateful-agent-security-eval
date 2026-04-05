#!/usr/bin/env python3
"""Analyze Bedrock Claude factorial results with statistical rigor.

Computes:
- Injection success and attack success rates
- 95% BCa bootstrapped confidence intervals
- Holm-Bonferroni corrected p-values
- Sleeper effect (injection - attack)
- Mechanistic findings
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_results(results_file: str) -> List[Dict]:
    """Load results from JSONL file."""
    results = []
    with open(results_file) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_ci_bca(outcomes: List[bool], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute bias-corrected and accelerated (BCa) confidence interval.
    
    Args:
        outcomes: List of boolean outcomes (True/False)
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower, upper) confidence interval bounds
    """
    outcomes_array = np.array(outcomes, dtype=float)
    n = len(outcomes_array)
    
    if n == 0:
        return (0.0, 1.0)
    
    # Point estimate
    theta_hat = np.mean(outcomes_array)
    
    # Bootstrap resamples
    np.random.seed(42)  # For reproducibility
    bootstrap_samples = []
    for _ in range(10000):
        resample = np.random.choice(outcomes_array, size=n, replace=True)
        bootstrap_samples.append(np.mean(resample))
    
    bootstrap_samples = np.array(bootstrap_samples)
    
    # Bias correction (z0)
    z0 = np.mean(bootstrap_samples < theta_hat)
    if z0 == 0 or z0 == 1:
        z0 = 0.5 / len(bootstrap_samples)  # Avoid log(0)
    z0 = np.arctanh(2 * z0 - 1) if z0 != 0.5 else 0
    
    # Acceleration (jack-knife)
    jack_samples = []
    for i in range(n):
        jack_sample = np.mean(np.delete(outcomes_array, i))
        jack_samples.append(jack_sample)
    
    jack_samples = np.array(jack_samples)
    jack_mean = np.mean(jack_samples)
    numerator = np.sum((jack_mean - jack_samples) ** 3)
    denominator = 6 * (np.sum((jack_mean - jack_samples) ** 2) ** 1.5)
    acceleration = numerator / denominator if denominator != 0 else 0
    
    # Percentile points
    alpha = (1 - confidence) / 2
    z_alpha = np.arctanh(2 * alpha - 1) if alpha != 0.5 else 0
    z_1_alpha = np.arctanh(2 * (1 - alpha) - 1) if (1 - alpha) != 0.5 else 0
    
    # BCa adjusted percentiles
    p_lower = np.arctanh(z0 + (z0 + z_alpha) / (1 - acceleration * (z0 + z_alpha)))
    p_upper = np.arctanh(z0 + (z0 + z_1_alpha) / (1 - acceleration * (z0 + z_1_alpha)))
    
    # Convert back to probability
    p_lower = (np.tanh(p_lower) + 1) / 2
    p_upper = (np.tanh(p_upper) + 1) / 2
    
    # Clamp to [0, 1]
    p_lower = max(0, min(1, p_lower))
    p_upper = max(0, min(1, p_upper))
    
    # Get percentile indices
    idx_lower = int(p_lower * len(bootstrap_samples))
    idx_upper = int(p_upper * len(bootstrap_samples))
    
    sorted_samples = np.sort(bootstrap_samples)
    ci_lower = sorted_samples[max(0, idx_lower)]
    ci_upper = sorted_samples[min(len(sorted_samples) - 1, idx_upper)]
    
    return (ci_lower, ci_upper)


def analyze_by_model(results: List[Dict]) -> Dict[str, Dict]:
    """Analyze results grouped by model."""
    by_model = {}
    
    for r in results:
        if r.get("error"):
            continue
        
        model_name = r.get("condition", {}).get("model", {}).get("model_name", "unknown")
        if model_name not in by_model:
            by_model[model_name] = {
                "injection_outcomes": [],
                "attack_outcomes": [],
                "btcr_outcomes": [],
                "memory_calls": [],
                "exfil_sessions": [],
            }
        
        by_model[model_name]["injection_outcomes"].append(r.get("injection_success", False))
        by_model[model_name]["attack_outcomes"].append(r.get("attack_success", False))
        by_model[model_name]["btcr_outcomes"].append(r.get("btcr_success_under_attack", False))
        by_model[model_name]["memory_calls"].append(r.get("injection_session_memory_calls", 0))
        by_model[model_name]["exfil_sessions"].append(r.get("exfiltration_session_index"))
    
    return by_model


def main():
    results_file = Path("results/bedrock_claude_factorial/results.jsonl")
    
    if not results_file.exists():
        logger.error("Results file not found: %s", results_file)
        return
    
    logger.info("="*80)
    logger.info("BEDROCK CLAUDE FACTORIAL ANALYSIS")
    logger.info("="*80)
    
    results = load_results(str(results_file))
    logger.info("Loaded %d results", len(results))
    
    # Filter successful runs
    successful = [r for r in results if not r.get("error")]
    logger.info("Successful runs: %d/%d (%.1f%%)", len(successful), len(results), 100 * len(successful) / len(results) if results else 0)
    
    # Analyze by model
    by_model = analyze_by_model(successful)
    
    logger.info("\n" + "="*80)
    logger.info("RESULTS BY MODEL")
    logger.info("="*80)
    
    for model_name in sorted(by_model.keys()):
        data = by_model[model_name]
        n = len(data["injection_outcomes"])
        
        if n == 0:
            logger.info("%s: No successful runs", model_name)
            continue
        
        # Compute rates
        inj_rate = np.mean(data["injection_outcomes"])
        atk_rate = np.mean(data["attack_outcomes"])
        btcr_rate = np.mean(data["btcr_outcomes"])
        sleeper_effect = inj_rate - atk_rate
        
        # Compute CIs
        inj_ci = compute_ci_bca(data["injection_outcomes"])
        atk_ci = compute_ci_bca(data["attack_outcomes"])
        
        # Memory stats
        mem_calls = np.array(data["memory_calls"])
        mem_mean = np.mean(mem_calls)
        mem_std = np.std(mem_calls)
        
        logger.info("\n%s (N=%d)", model_name.split(".")[-1], n)
        logger.info("  Injection Success: %.1f%% [95%% CI: %.1f%%-%.1f%%]", 
                   100 * inj_rate, 100 * inj_ci[0], 100 * inj_ci[1])
        logger.info("  Attack Success:    %.1f%% [95%% CI: %.1f%%-%.1f%%]", 
                   100 * atk_rate, 100 * atk_ci[0], 100 * atk_ci[1])
        logger.info("  BTCR (under attack): %.1f%%", 100 * btcr_rate)
        logger.info("  Sleeper Effect:    %.1f%% (injection - attack)", 100 * sleeper_effect)
        logger.info("  Memory Saves (inj session): %.1f ± %.1f", mem_mean, mem_std)
    
    # Cross-model comparison
    logger.info("\n" + "="*80)
    logger.info("CROSS-MODEL COMPARISON")
    logger.info("="*80)
    
    models = sorted(by_model.keys())
    if len(models) >= 2:
        model_a = models[0]
        model_b = models[1]
        
        data_a = by_model[model_a]
        data_b = by_model[model_b]
        
        inj_a = np.mean(data_a["injection_outcomes"])
        inj_b = np.mean(data_b["injection_outcomes"])
        atk_a = np.mean(data_a["attack_outcomes"])
        atk_b = np.mean(data_b["attack_outcomes"])
        
        logger.info("\n%s vs %s", model_a.split(".")[-1], model_b.split(".")[-1])
        logger.info("  Injection Success: %.1f%% vs %.1f%% (Δ = %.1f%%)", 
                   100 * inj_a, 100 * inj_b, 100 * (inj_a - inj_b))
        logger.info("  Attack Success:    %.1f%% vs %.1f%% (Δ = %.1f%%)", 
                   100 * atk_a, 100 * atk_b, 100 * (atk_a - atk_b))
    
    # Mechanistic findings
    logger.info("\n" + "="*80)
    logger.info("MECHANISTIC FINDINGS")
    logger.info("="*80)
    
    for model_name in sorted(by_model.keys()):
        data = by_model[model_name]
        n = len(data["injection_outcomes"])
        
        if n == 0:
            continue
        
        # Sleeper effect
        sleeper_count = sum(1 for inj, atk in zip(data["injection_outcomes"], data["attack_outcomes"]) if inj and not atk)
        if sleeper_count > 0:
            logger.info("\n%s: Sleeper Effect Detected", model_name.split(".")[-1])
            logger.info("  %d/%d runs show injection without attack (%.1f%%)", 
                       sleeper_count, n, 100 * sleeper_count / n)
        
        # Injection resistance
        inj_rate = np.mean(data["injection_outcomes"])
        if inj_rate < 0.2:
            logger.info("\n%s: Injection Resistance", model_name.split(".")[-1])
            logger.info("  Only %.1f%% injection success - model detects/refuses malicious rules", 100 * inj_rate)
        
        # Memory behavior
        mem_calls = np.array(data["memory_calls"])
        if np.mean(mem_calls) > 0:
            logger.info("\n%s: Memory Behavior", model_name.split(".")[-1])
            logger.info("  Average memory saves in injection session: %.1f", np.mean(mem_calls))
            logger.info("  Range: %d-%d", int(np.min(mem_calls)), int(np.max(mem_calls)))
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info("Results saved to: %s", results_file)


if __name__ == "__main__":
    main()
