"""Bedrock Date Sensitivity — N=40, Sequential, Resume-Safe.

Simple design: for each model, run N=40 per date sequentially.
Resume-safe: counts existing records in JSONL, only runs remaining.
WARP-tolerant: if connection drops, just rerun this script.

5 models × 3 dates × N=40 = 600 runs.

Usage:
    .venv/bin/python scripts/run_bedrock_date_sweep_n40.py
    .venv/bin/python scripts/run_bedrock_date_sweep_n40.py --analyze   # just print verdict
"""
import json
import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/bedrock_date_sweep_n40")
DATES = ["2026-04-17", "2026-06-25", "2026-03-15"]
N = 40

MODELS = [
    {"provider": "bedrock", "model_name": "nvidia.nemotron-super-3-120b", "aws_region": "us-east-1", "aws_profile": "tra-sso", "inference_profile": "nvidia.nemotron-super-3-120b", "rag_session_query_limit": 15, "rate_limit": {"max_concurrent": 1, "rpm_limit": 60}},
    {"provider": "bedrock", "model_name": "minimax.minimax-m2.5", "aws_region": "us-east-1", "aws_profile": "tra-sso", "inference_profile": "minimax.minimax-m2.5", "rag_session_query_limit": 15, "rate_limit": {"max_concurrent": 1, "rpm_limit": 60}},
    {"provider": "bedrock", "model_name": "moonshot.kimi-k2-thinking", "aws_region": "us-east-1", "aws_profile": "tra-sso", "inference_profile": "moonshot.kimi-k2-thinking", "rag_session_query_limit": 15, "rate_limit": {"max_concurrent": 1, "rpm_limit": 60}},
    {"provider": "bedrock", "model_name": "qwen.qwen3-next-80b-a3b", "aws_region": "us-east-1", "aws_profile": "tra-sso", "inference_profile": "qwen.qwen3-next-80b-a3b", "rag_session_query_limit": 15, "rate_limit": {"max_concurrent": 1, "rpm_limit": 60}},
    {"provider": "bedrock", "model_name": "us.meta.llama4-maverick-17b-instruct-v1:0", "aws_region": "us-east-1", "aws_profile": "tra-sso", "inference_profile": "us.meta.llama4-maverick-17b-instruct-v1:0", "rag_session_query_limit": 15, "rate_limit": {"max_concurrent": 1, "rpm_limit": 60}},
]


def safe_name(model_name: str) -> str:
    return model_name.replace(".", "_").replace(":", "_").replace("/", "_")


def count_existing(path: Path) -> int:
    if not path.exists():
        return 0
    return len([l for l in path.read_text().splitlines() if l.strip()])


def run_all():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for model_cfg in MODELS:
        model_name = model_cfg["model_name"]
        sn = safe_name(model_name)

        for date in DATES:
            outfile = RESULTS_DIR / f"{sn}_{date}.jsonl"
            existing = count_existing(outfile)
            remaining = N - existing

            if remaining <= 0:
                logger.info(f"[{model_name}] {date}: already have {existing}/{N}, skipping")
                continue

            logger.info(f"[{model_name}] {date}: have {existing}/{N}, running {remaining} more")

            os.environ["EVAL_OVERRIDE_DATE"] = date

            config = load_config("experiments/configs/defense_factorial.yaml")
            config.runs_per_condition = remaining
            config.results_path = str(outfile)
            config.models = [model_cfg]
            config.defenses = [d for d in config.defenses if d.get("name") == "no_defense"]
            config.attacks = [a for a in config.attacks if a.get("type") == "delayed_trigger"]

            runner = ExperimentRunner(config)
            runner.run_all()

            # Report progress
            final_count = count_existing(outfile)
            records = [json.loads(l) for l in outfile.read_text().splitlines() if l.strip()]
            valid = [r for r in records if not r.get("error")]
            asr = sum(1 for r in valid if r.get("attack_success"))
            logger.info(f"[{model_name}] {date}: {final_count}/{N} done, ASR={asr}/{len(valid)}")

    os.environ.pop("EVAL_OVERRIDE_DATE", None)
    logger.info("All models complete. Run with --analyze for verdict.")


def analyze():
    print("=" * 60)
    print(f"Bedrock Date Sensitivity N=40 — VERDICT (Bonferroni α=0.017)")
    print("=" * 60)

    for model_cfg in MODELS:
        model_name = model_cfg["model_name"]
        sn = safe_name(model_name)
        print(f"\n{model_name}:")

        rates = {}
        inj_rates = {}
        for date in DATES:
            f = RESULTS_DIR / f"{sn}_{date}.jsonl"
            if not f.exists() or f.stat().st_size == 0:
                print(f"  {date}: NO DATA")
                continue
            records = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            valid = [r for r in records if not r.get("error")]
            asr = sum(1 for r in valid if r.get("attack_success"))
            inj = sum(1 for r in valid if r.get("injection_success"))
            n = len(valid)
            rates[date] = (asr, n)
            inj_rates[date] = (inj, n)
            print(f"  {date}: inj={inj}/{n} ({100*inj/n:.1f}%)  ASR={asr}/{n} ({100*asr/n:.1f}%)")

        if len(rates) >= 2:
            from scipy.stats import fisher_exact
            ALPHA = 0.017
            date_list = sorted(rates.keys())

            print(f"  --- Attack Success (Fisher's exact, α={ALPHA}) ---")
            for i in range(len(date_list)):
                for j in range(i+1, len(date_list)):
                    d1, d2 = date_list[i], date_list[j]
                    a1, n1 = rates[d1]
                    a2, n2 = rates[d2]
                    table = [[a1, n1-a1], [a2, n2-a2]]
                    _, p = fisher_exact(table)
                    delta = (a1/n1 - a2/n2) * 100 if n1 > 0 and n2 > 0 else 0
                    sig = "*** SIGNIFICANT" if p < ALPHA else ""
                    print(f"    {d1} vs {d2}: Δ={delta:+.1f}pp, p={p:.4f} {sig}")

            print(f"  --- Injection Success (Fisher's exact, α={ALPHA}) ---")
            for i in range(len(date_list)):
                for j in range(i+1, len(date_list)):
                    d1, d2 = date_list[i], date_list[j]
                    i1, n1 = inj_rates[d1]
                    i2, n2 = inj_rates[d2]
                    table = [[i1, n1-i1], [i2, n2-i2]]
                    _, p = fisher_exact(table)
                    delta = (i1/n1 - i2/n2) * 100 if n1 > 0 and n2 > 0 else 0
                    sig = "*** SIGNIFICANT" if p < ALPHA else ""
                    print(f"    {d1} vs {d2}: Δ={delta:+.1f}pp, p={p:.4f} {sig}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    if args.analyze:
        analyze()
    else:
        run_all()
