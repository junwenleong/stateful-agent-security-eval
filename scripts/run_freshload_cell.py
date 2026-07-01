"""Run one model's DTA cells from a given config, on whatever Ollama daemon is up.

Fresh-daemon-per-model is enforced by the orchestrator (run_freshload_suite.sh),
which restarts Ollama before invoking this script for each model. This avoids the
long-running multi-model daemon churn that degraded reasoning-model trigger-session
behaviour in the RATG sprint (see learningjourney Iteration 55).

Loads a config (ratg_factorial.yaml or judge_7b.yaml), filters to a single model
and the delayed_trigger attack, keeps that config's defenses (e.g. no_defense +
ratg, or no_defense + rag_llm_judge_7b), overrides N and output path, and runs.

Usage:
    .venv/bin/python scripts/run_freshload_cell.py \
        --config experiments/configs/ratg_factorial.yaml \
        --model qwen3.5:122b --n 20 \
        --out results/freshload_ratg/qwen3.5_122b.jsonl
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = load_config(args.config)
    config.runs_per_condition = args.n
    config.results_path = args.out
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    config.models = [m for m in config.models if m.get("model_name") == args.model]
    if not config.models:
        raise SystemExit(f"Model '{args.model}' not found in {args.config}")
    # DTA only — the reduction/effect is measured on the attack arm; skip no_attack for speed.
    config.attacks = [a for a in config.attacks if a.get("type") == "delayed_trigger"]

    defenses = [d.get("name") or d.get("type") for d in config.defenses]
    print(f"[freshload] model={args.model} n={args.n} defenses={defenses}")
    print(f"[freshload] config={args.config} out={args.out}")

    runner = ExperimentRunner(config)
    runner.run_all()


if __name__ == "__main__":
    main()
