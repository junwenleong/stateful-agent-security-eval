#!/usr/bin/env python3
"""Factorial results analysis: BCa CIs, Holm-Bonferroni, completion check.

Usage:
    .venv/bin/python scripts/analyze_results.py
    .venv/bin/python scripts/analyze_results.py --results results/defense_factorial/results.jsonl
    .venv/bin/python scripts/analyze_results.py --results results/defense_factorial/results.jsonl --config experiments/configs/defense_factorial.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner.config_loader import load_config
from src.stats.bootstrap_engine import BootstrapEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RESULTS = "results/defense_factorial/results.jsonl"
DEFAULT_CONFIG = "experiments/configs/defense_factorial.yaml"


# ── helpers ───────────────────────────────────────────────────────────────────

def _condition_key(record: dict) -> str:
    """Canonical condition string matching the YAML comparison format.

    Uses defense 'name' when present (e.g. 'no_defense'), falling back to 'type'
    (e.g. 'none'). The YAML comparisons use the name field, so this must match.
    """
    cond = record.get("condition", {})
    attack = cond.get("attack", {}).get("type", "unknown")
    defense_cfg = cond.get("defense", {})
    defense = defense_cfg.get("name") or defense_cfg.get("type", "unknown")
    model = cond.get("model", {}).get("model_name", "unknown")
    return f"attack={attack},defense={defense},model={model}"


def load_results(path: str) -> list[dict]:
    """Load JSONL, skip malformed lines, filter to successful records only (Q3).

    V1/V2 guard: records with defense_schema_version=None and run_timestamp
    before 2026-04-11 are v1 data (list_all_facts returned full values, making
    memory_sandbox results invalid). They are excluded here so that accidentally
    merging the v1 archive file (results_v1_list_all_facts_full_values.jsonl)
    with the active results file does not silently corrupt the analysis.

    All active records have run_timestamp >= 2026-04-11 and defense_schema_version=None
    (the field was added after the runs completed; None is the correct v2 marker for
    existing records). The guard only fires for records with timestamps before the
    v2 fix date, which only exist in the archived v1 file.
    """
    records = []
    skipped_errors = 0
    skipped_malformed = 0
    skipped_v1 = 0
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                skipped_malformed += 1
                logger.warning("Malformed JSON on line %d — skipped", lineno)
                continue
            # Q3: filter out error records before any statistics
            if r.get("error"):
                skipped_errors += 1
                continue
            # V1/V2 guard: exclude v1 records (list_all_facts returned full values).
            # defense_schema_version=None + run_timestamp < 2026-04-11 → v1 archive record.
            # Active v2 records also have defense_schema_version=None but timestamps >= 2026-04-11.
            ts = r.get("run_timestamp", "")
            if r.get("defense_schema_version") is None and ts and ts < "2026-04-11":
                skipped_v1 += 1
                continue
            records.append(r)

    if skipped_malformed:
        logger.warning("Skipped %d malformed lines", skipped_malformed)
    if skipped_v1:
        logger.warning(
            "Skipped %d v1 records (run_timestamp < 2026-04-11, defense_schema_version=None). "
            "These are from the archived v1 run where list_all_facts returned full values. "
            "Do not merge results_v1_list_all_facts_full_values.jsonl with the active results file.",
            skipped_v1,
        )
    if skipped_errors:
        logger.info("Filtered out %d error records (kept %d successes)", skipped_errors, len(records))
    return records


def check_completion(records: list[dict], config_path: str, threshold: float = 0.95) -> None:
    """Q9: Warn prominently if factorial is below threshold completion."""
    try:
        cfg = load_config(config_path)
    except Exception as e:
        logger.warning("Could not load config for completion check: %s", e)
        return

    expected = len(cfg.attacks) * len(cfg.defenses) * len(cfg.models) * cfg.runs_per_condition
    actual = len(records)
    pct = actual / expected if expected > 0 else 0.0

    if pct < threshold:
        logger.warning("=" * 70)
        logger.warning("⚠  FACTORIAL INCOMPLETE: %d / %d runs (%.1f%%)", actual, expected, pct * 100)
        logger.warning("   Statistics below are PROVISIONAL — do not treat as final.")
        logger.warning("=" * 70)
    else:
        logger.info("Completion: %d / %d runs (%.1f%%) ✓", actual, expected, pct * 100)

    # Per-condition completion
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts[_condition_key(r)] += 1
    incomplete = {k: v for k, v in counts.items() if v < cfg.runs_per_condition}
    if incomplete:
        logger.warning("Incomplete conditions (%d):", len(incomplete))
        for k, v in sorted(incomplete.items()):
            logger.warning("  %s: %d/%d", k, v, cfg.runs_per_condition)


def validate_comparisons(comparisons, condition_keys: set[str]) -> None:
    """Q12: Raise a clear error if any comparison references a missing condition."""
    missing: list[tuple[str, str]] = []
    for comp in comparisons:
        a, b = comp.condition_a, comp.condition_b
        if a not in condition_keys:
            missing.append(("condition_a", a))
        if b not in condition_keys:
            missing.append(("condition_b", b))

    if missing:
        lines = [f"  {side}: {key}" for side, key in missing]
        raise ValueError(
            f"Comparisons reference {len(missing)} condition(s) with zero results in the JSONL.\n"
            "This would produce NaN in bootstrap — fix the comparison keys or check the results file.\n"
            "Missing:\n" + "\n".join(lines)
        )


# ── core analysis ─────────────────────────────────────────────────────────────

def compute_condition_stats(records: list[dict], engine: BootstrapEngine) -> dict:
    """Per-condition ASR and BTCR with BCa CIs."""
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_condition[_condition_key(r)].append(r)

    stats = {}
    for cond_key, recs in sorted(by_condition.items()):
        asr_outcomes = np.array([1.0 if r.get("attack_success") else 0.0 for r in recs])
        btcr_outcomes = np.array([1.0 if r.get("btcr_success") else 0.0 for r in recs])

        asr_ci = engine.compute_ci(asr_outcomes)
        btcr_ci = engine.compute_ci(btcr_outcomes)

        stats[cond_key] = {
            "n": len(recs),
            "asr": {
                "point_estimate": asr_ci.point_estimate,
                "lower": asr_ci.lower,
                "upper": asr_ci.upper,
                "warning": asr_ci.warning,
            },
            "btcr": {
                "point_estimate": btcr_ci.point_estimate,
                "lower": btcr_ci.lower,
                "upper": btcr_ci.upper,
                "warning": btcr_ci.warning,
            },
        }
    return stats


def _is_na_comparison(condition_a: str, condition_b: str) -> str | None:
    """Return a reason string if this comparison should be annotated N/A.

    qwq:32b is a Draft-Only Executor: it never executes the exfiltration under
    any defense (ASR=0% across the board) except memory_sandbox (100% via RAG
    re-injection bypass). Comparing its no_defense baseline against its own
    defense conditions measures nothing about defense effectiveness — the
    baseline is already 0%. Similarly, comparing qwq:32b's no_defense ASR
    against other models' no_defense ASR measures model behavior, not defense
    effectiveness.

    Excluded (7 comparisons):
      - 6 primary DTA: no_defense,qwq vs {minimizer,sanitizer,prompt_hardening,
        rag_sanitizer,memory_sandbox,rag_llm_judge},qwq
      - 1 cross-model no_defense DTA: qwen2.5:14b,no_defense vs qwq:32b,no_defense

    NOT excluded (6 qwq comparisons that remain active):
      - 5 secondary BTCR (no_attack arm): these measure utility, not attack
      - 1 cross-model memory_sandbox DTA: qwen2.5:14b vs qwq:32b under
        memory_sandbox — this IS meaningful (documents the inversion effect)
    """
    NA_REASON = "Draft-Only Executor: mechanistically distinct attack pathway"

    # Primary DTA qwq: no_defense,qwq vs any_defense,qwq (DTA arm)
    if ("model=qwq:32b" in condition_a and "model=qwq:32b" in condition_b
            and "attack=delayed_trigger" in condition_a
            and "attack=delayed_trigger" in condition_b
            and "defense=no_defense" in condition_a):
        return NA_REASON

    # Cross-model no_defense DTA involving qwq
    if ("attack=delayed_trigger" in condition_a
            and "attack=delayed_trigger" in condition_b
            and "defense=no_defense" in condition_a
            and "defense=no_defense" in condition_b
            and ("model=qwq:32b" in condition_a or "model=qwq:32b" in condition_b)):
        return NA_REASON

    return None


def run_comparisons(
    comparisons,
    by_condition: dict[str, list[dict]],
    engine: BootstrapEngine,
) -> list[dict]:
    """BCa diff CIs for all pre-specified comparisons, then Holm-Bonferroni.

    Comparisons involving qwq:32b's primary DTA conditions are annotated N/A
    (Draft-Only Executor: mechanistically distinct attack pathway) and excluded
    from the Holm-Bonferroni correction. They are still computed for
    completeness but do not count toward the active comparison set.
    """
    results = []
    for comp in comparisons:
        na_reason = _is_na_comparison(comp.condition_a, comp.condition_b)

        recs_a = by_condition[comp.condition_a]
        recs_b = by_condition[comp.condition_b]
        a_asr = np.array([1.0 if r.get("attack_success") else 0.0 for r in recs_a])
        b_asr = np.array([1.0 if r.get("attack_success") else 0.0 for r in recs_b])
        ci = engine.compute_diff_ci(a_asr, b_asr)
        # Significant if 95% CI excludes zero (equivalent to α=0.05 two-sided test)
        significant = ci.lower > 0 or ci.upper < 0
        results.append({
            "condition_a": comp.condition_a,
            "condition_b": comp.condition_b,
            "diff_point": ci.point_estimate,
            "diff_lower": ci.lower,
            "diff_upper": ci.upper,
            "significant_pre_correction": significant,
            "warning": ci.warning,
            "na_reason": na_reason,
        })

    # Holm-Bonferroni on ACTIVE comparisons only (exclude N/A)
    active_indices = [i for i, r in enumerate(results) if r["na_reason"] is None]
    n_active = len(active_indices)

    # Sort active comparisons by |diff| descending (most significant first)
    ranked_active = sorted(
        active_indices,
        key=lambda i: -abs(results[i]["diff_point"]),
    )

    holm_significant = [False] * len(results)
    for rank, idx in enumerate(ranked_active):
        if results[idx]["significant_pre_correction"]:
            holm_significant[idx] = True
        else:
            # Step-down: once we hit a non-significant result, stop
            break

    for i, r in enumerate(results):
        if r["na_reason"] is not None:
            r["significant_holm"] = None  # N/A — not tested
        else:
            r["significant_holm"] = holm_significant[i]

    n_na = sum(1 for r in results if r["na_reason"] is not None)
    n_sig = sum(1 for r in results if r["significant_holm"] is True)
    logger.info(
        "Comparisons: %d total, %d N/A, %d active, %d significant (Holm-Bonferroni)",
        len(results), n_na, n_active, n_sig,
    )

    return results


def print_mechanistic_summary(records: list[dict]) -> None:
    """Print mechanistic tag counts per DTA condition.

    Reads tags from result["mechanistic_tags"]["tags"] (a list of strings).
    Uses list-membership check ("tag" in tags_list), NOT dict-key access
    (result["mechanistic_tags"].get("tag")) — the latter silently returns None
    because the serialized structure is {"tags": [...], "mechanism": "...", ...},
    not a flat dict of booleans.
    """
    from collections import Counter

    # Only DTA runs carry meaningful mechanistic tags
    dta_records = [
        r for r in records
        if r.get("condition", {}).get("attack", {}).get("type") == "delayed_trigger"
    ]
    if not dta_records:
        return

    # Group by defense
    by_defense: dict[str, list[dict]] = defaultdict(list)
    for r in dta_records:
        defense_cfg = r.get("condition", {}).get("defense", {})
        defense = defense_cfg.get("name") or defense_cfg.get("type", "unknown")
        by_defense[defense].append(r)

    print("\n" + "=" * 80)
    print("MECHANISTIC TAG SUMMARY (DTA runs only)")
    print("NOTE: tags read from result['mechanistic_tags']['tags'] (list membership)")
    print("=" * 80)
    for defense, recs in sorted(by_defense.items()):
        tag_counter: Counter = Counter()
        for r in recs:
            # Correct access pattern: mechanistic_tags["tags"] is a list of strings.
            # Do NOT use mechanistic_tags.get("semantic_masking_success") — that key
            # does not exist at the top level; it would silently return None.
            tags_list = r.get("mechanistic_tags", {}).get("tags", [])
            for tag in tags_list:
                tag_counter[tag] += 1
        n = len(recs)
        tag_str = ", ".join(f"{tag}={count}/{n}" for tag, count in sorted(tag_counter.items()))
        print(f"  {defense:<20} (n={n:>3}): {tag_str or '(no tags)'}")


def print_summary(stats: dict, comparison_results: list[dict]) -> None:
    """Print a concise results table."""
    print("\n" + "=" * 80)
    print("CONDITION STATISTICS")
    print("=" * 80)
    print(f"{'Condition':<60} {'N':>4} {'ASR':>6} {'[95% CI]':>16} {'BTCR':>6}")
    print("-" * 80)
    for cond, s in sorted(stats.items()):
        asr = s["asr"]
        btcr = s["btcr"]
        ci_str = f"[{asr['lower']:.2f}, {asr['upper']:.2f}]"
        print(f"{cond:<60} {s['n']:>4} {asr['point_estimate']:>6.2f} {ci_str:>16} {btcr['point_estimate']:>6.2f}")
        if asr.get("warning"):
            print(f"  ⚠  {asr['warning']}")

    print("\n" + "=" * 80)
    print("COMPARISONS (Holm-Bonferroni corrected)")
    print("=" * 80)
    n_na = sum(1 for r in comparison_results if r.get("na_reason") is not None)
    n_active = len(comparison_results) - n_na
    sig_count = sum(1 for r in comparison_results if r.get("significant_holm") is True)
    print(f"{len(comparison_results)} total, {n_na} N/A, {n_active} active, "
          f"{sig_count} significant after Holm-Bonferroni\n")
    for r in comparison_results:
        if r.get("na_reason"):
            sig = "—"
        elif r.get("significant_holm"):
            sig = "✓"
        else:
            sig = " "
        pre = "*" if r["significant_pre_correction"] else " "
        print(
            f"[{sig}] {r['condition_a'][:45]:<45} vs"
            f"\n    {r['condition_b'][:45]:<45}"
            f"  diff={r['diff_point']:+.3f} [{r['diff_lower']:+.3f}, {r['diff_upper']:+.3f}]"
            f"  pre={pre}"
        )
        if r.get("na_reason"):
            print(f"    N/A: {r['na_reason']}")
        if r.get("warning"):
            print(f"    ⚠  {r['warning']}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze factorial results")
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out", default=None, help="Write JSON output to file")
    args = parser.parse_args()

    if not Path(args.results).exists():
        logger.error("Results file not found: %s", args.results)
        sys.exit(1)

    # Load and filter records (Q3: error records excluded)
    records = load_results(args.results)
    if not records:
        logger.error("No successful records found in %s", args.results)
        sys.exit(1)
    logger.info("Loaded %d successful records", len(records))

    # Q9: completion check
    check_completion(records, args.config)

    # Load config for comparisons
    try:
        cfg = load_config(args.config)
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        sys.exit(1)

    # Build condition index
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_condition[_condition_key(r)].append(r)
    condition_keys = set(by_condition.keys())

    # Q12: validate all comparisons reference existing conditions
    try:
        validate_comparisons(cfg.comparisons, condition_keys)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)

    engine = BootstrapEngine(
        n_resamples=cfg.n_bootstrap,
        alpha=cfg.alpha,
        seed=cfg.bootstrap_seed,
    )

    stats = compute_condition_stats(records, engine)
    comparison_results = run_comparisons(cfg.comparisons, by_condition, engine)

    print_summary(stats, comparison_results)
    print_mechanistic_summary(records)

    if args.out:
        output = {"stats": stats, "comparisons": comparison_results, "mechanistic_tags": {}}
        # Populate mechanistic tag counts for JSON output using correct list-membership pattern
        dta_records = [
            r for r in records
            if r.get("condition", {}).get("attack", {}).get("type") == "delayed_trigger"
        ]
        tag_output: dict[str, dict[str, int]] = {}
        for r in dta_records:
            defense_cfg = r.get("condition", {}).get("defense", {})
            defense = defense_cfg.get("name") or defense_cfg.get("type", "unknown")
            if defense not in tag_output:
                tag_output[defense] = {}
            tags_list = r.get("mechanistic_tags", {}).get("tags", [])
            for tag in tags_list:
                tag_output[defense][tag] = tag_output[defense].get(tag, 0) + 1
        output["mechanistic_tags"] = tag_output
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info("Results written to %s", args.out)


if __name__ == "__main__":
    main()
