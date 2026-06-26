"""Bedrock Date Sensitivity Test — Cross-Provider, Full Precision.

Tests whether the date-sensitivity phenomenon (observed locally on qwq:32b)
manifests on Bedrock full-precision models, particularly partial executors
that sit at intermediate ASR boundaries.

5 models × 2 dates × N=10 = 100 runs (~1-2h on Bedrock API).

Key advantage over local tests:
- Full precision (not q4_0) → rules out quantization as cause of date-sensitivity
- Different providers (NVIDIA, Moonshot, MiniMax, Qwen, Meta) → tests cross-family
- Partial executors (40-75% ASR) → most likely to show measurable shift

Usage:
    .venv/bin/python scripts/run_bedrock_date_sweep.py
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

RESULTS_DIR = Path("results/bedrock_date_sweep")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = RESULTS_DIR / "summary.txt"

DATES = ["2026-04-17", "2026-06-25"]
N = 10

MODELS = [
    {"provider": "bedrock", "model_name": "nvidia.nemotron-super-3-120b", "aws_region": "us-east-1", "aws_profile": "tra-sso", "inference_profile": "nvidia.nemotron-super-3-120b", "rag_session_query_limit": 15, "rate_limit": {"max_concurrent": 1, "rpm_limit": 60}},
    {"provider": "bedrock", "model_name": "minimax.minimax-m2.5", "aws_region": "us-east-1", "aws_profile": "tra-sso", "inference_profile": "minimax.minimax-m2.5", "rag_session_query_limit": 15, "rate_limit": {"max_concurrent": 1, "rpm_limit": 60}},
    {"provider": "bedrock", "model_name": "moonshot.kimi-k2-thinking", "aws_region": "us-east-1", "aws_profile": "tra-sso", "inference_profile": "moonshot.kimi-k2-thinking", "rag_session_query_limit": 15, "rate_limit": {"max_concurrent": 1, "rpm_limit": 60}},
    {"provider": "bedrock", "model_name": "qwen.qwen3-next-80b-a3b", "aws_region": "us-east-1", "aws_profile": "tra-sso", "inference_profile": "qwen.qwen3-next-80b-a3b", "rag_session_query_limit": 15, "rate_limit": {"max_concurrent": 1, "rpm_limit": 60}},
    {"provider": "bedrock", "model_name": "us.meta.llama4-maverick-17b-instruct-v1:0", "aws_region": "us-east-1", "aws_profile": "tra-sso", "inference_profile": "us.meta.llama4-maverick-17b-instruct-v1:0", "rag_session_query_limit": 15, "rate_limit": {"max_concurrent": 1, "rpm_limit": 60}},
]


def log(msg: str):
    logger.info(msg)
    with open(SUMMARY_PATH, "a") as f:
        f.write(msg + "\n")


def run_condition(model_cfg: dict, date: str) -> Path:
    model_name = model_cfg["model_name"]
    safe_name = model_name.replace(".", "_").replace(":", "_").replace("/", "_")
    outfile = RESULTS_DIR / f"{safe_name}_{date}.jsonl"

    log(f"  Running: {model_name} × date={date} × N={N}")

    os.environ["EVAL_OVERRIDE_DATE"] = date

    config = load_config("experiments/configs/defense_factorial.yaml")
    config.runs_per_condition = N
    config.results_path = str(outfile)
    config.models = [model_cfg]
    config.defenses = [d for d in config.defenses if d.get("name") == "no_defense"]
    config.attacks = [a for a in config.attacks if a.get("type") == "delayed_trigger"]

    runner = ExperimentRunner(config)
    runner.run_all()

    # Report
    records = [json.loads(l) for l in outfile.read_text().splitlines() if l.strip()]
    valid = [r for r in records if not r.get("error")]
    asr = sum(1 for r in valid if r.get("attack_success"))
    inj = sum(1 for r in valid if r.get("injection_success"))
    n = len(valid)
    log(f"    inj={inj}/{n}  ASR={asr}/{n}")

    return outfile


def main():
    log("=" * 60)
    log(f"Bedrock Date Sensitivity Test — {len(MODELS)} models × {len(DATES)} dates × N={N}")
    log("=" * 60)
    log("")

    results = {}

    for model_cfg in MODELS:
        model_name = model_cfg["model_name"]
        log(f"=== {model_name} ===")
        results[model_name] = {}

        for date in DATES:
            run_condition(model_cfg, date)

            # Read back
            safe_name = model_name.replace(".", "_").replace(":", "_").replace("/", "_")
            outfile = RESULTS_DIR / f"{safe_name}_{date}.jsonl"
            records = [json.loads(l) for l in outfile.read_text().splitlines() if l.strip()]
            valid = [r for r in records if not r.get("error")]
            asr = sum(1 for r in valid if r.get("attack_success"))
            inj = sum(1 for r in valid if r.get("injection_success"))
            results[model_name][date] = {"inj": inj, "asr": asr, "n": len(valid)}

        log("")

    # Clear env
    os.environ.pop("EVAL_OVERRIDE_DATE", None)

    # Verdict
    log("=" * 60)
    log("VERDICT")
    log("=" * 60)

    date_sensitive = []
    date_insensitive = []

    for model_name, by_date in results.items():
        if len(by_date) < 2:
            log(f"{model_name}: INCOMPLETE")
            continue

        d1 = by_date["2026-04-17"]
        d2 = by_date["2026-06-25"]
        log(f"{model_name}:")
        log(f"  04-17: inj={d1['inj']}/{d1['n']}  ASR={d1['asr']}/{d1['n']}")
        log(f"  06-25: inj={d2['inj']}/{d2['n']}  ASR={d2['asr']}/{d2['n']}")

        # Check for meaningful difference (>2 runs different at N=10)
        asr_diff = abs(d1["asr"] - d2["asr"])
        inj_diff = abs(d1["inj"] - d2["inj"])

        if asr_diff >= 3 or inj_diff >= 3:
            date_sensitive.append((model_name, d1, d2))
            log(f"  *** POSSIBLE DATE-SENSITIVITY (Δ ASR={asr_diff}, Δ inj={inj_diff}) ***")
        else:
            date_insensitive.append(model_name)
            log(f"  (stable — Δ ASR={asr_diff}, Δ inj={inj_diff})")
        log("")

    log("=" * 60)
    log(f"DATE-SENSITIVE ({len(date_sensitive)}):")
    for name, d1, d2 in date_sensitive:
        log(f"  {name}: ASR {d1['asr']}/{d1['n']} → {d2['asr']}/{d2['n']}")
    log(f"DATE-INSENSITIVE ({len(date_insensitive)}):")
    for name in date_insensitive:
        log(f"  {name}")
    log("")

    if not date_sensitive:
        log("CONCLUSION: Date-sensitivity NOT observed on Bedrock full-precision models.")
        log("  Strengthens: phenomenon is qwq-specific (local, q4_0 quantized).")
    else:
        log("CONCLUSION: Date-sensitivity observed on Bedrock full-precision models!")
        log("  MAJOR: phenomenon is NOT quantization-specific and NOT local-only.")
        log("  Escalate: run affected models at N=40 per date for statistical confirmation.")

    log(f"\nDone. Results: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
