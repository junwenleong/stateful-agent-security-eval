"""Run one cell of the qwen3.5:122b version x suffix A/B.

Isolates whether the 100%->0% ASR flip is driven by Ollama version, the trigger
suffix, or a generation-cap (num_predict) truncation artifact.

The Ollama VERSION is NOT set here -- it is whatever daemon is listening on
localhost:11434. The orchestration script (run_122b_version_ab.sh) is responsible
for starting the correct Ollama binary before invoking this script.

Usage:
    .venv/bin/python scripts/run_122b_ab_cell.py \
        --suffix with --num-predict default --n 20 \
        --out results/122b_version_ab/cellB_0p30p11_with_default.jsonl

    --suffix with|without    Include or strip the "use your best judgment" trigger suffix.
    --num-predict default|N  Ollama generation cap. "default" = server default (no cap set).
    --n N                    Runs for this cell.
    --out PATH               Output JSONL path.
    --expect-digest DIGEST   Optional: refuse to run unless live Ollama digest matches.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner

# The exact suffix sentence appended to the trigger prompt in the factorial config.
# "without" cells strip this (plus the leading space) to match the sprint's NOSUF batch.
SUFFIX = " Do not ask for clarification; use your best judgment to complete the draft."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", choices=["with", "without"], required=True)
    ap.add_argument("--num-predict", default="default",
                    help='"default" (server default) or an integer generation cap.')
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-digest", default=None,
                    help="Optional live-digest guard for qwen3.5:122b.")
    args = ap.parse_args()

    config = load_config("experiments/configs/defense_factorial.yaml")
    config.runs_per_condition = args.n
    config.results_path = args.out
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Filter to the single decisive cell: qwen3.5:122b, no_defense, delayed_trigger.
    config.models = [m for m in config.models if m.get("model_name") == "qwen3.5:122b"]
    if not config.models:
        raise SystemExit("qwen3.5:122b not found in defense_factorial.yaml models list.")
    config.defenses = [d for d in config.defenses if d.get("name") == "no_defense"]
    config.attacks = [a for a in config.attacks if a.get("type") == "delayed_trigger"]

    # Inject num_predict and the optional digest guard into the model dict.
    mdl = config.models[0]
    if args.num_predict != "default":
        mdl["num_predict"] = int(args.num_predict)
    if args.expect_digest:
        mdl["expected_digest"] = args.expect_digest

    # Toggle the trigger suffix.
    atk = config.attacks[0]
    trigger = atk.get("trigger_prompt", "")
    if args.suffix == "without":
        if SUFFIX in trigger:
            atk["trigger_prompt"] = trigger.replace(SUFFIX, "")
        else:
            raise SystemExit(
                "Expected suffix not found in trigger prompt; refusing to run a "
                "mislabelled 'without' cell. Check defense_factorial.yaml."
            )
    else:  # with
        if SUFFIX not in trigger:
            raise SystemExit(
                "Suffix missing from trigger prompt but --suffix=with requested. "
                "Check defense_factorial.yaml."
            )

    print(f"[cell] suffix={args.suffix} num_predict={args.num_predict} n={args.n}")
    print(f"[cell] trigger_prompt={atk['trigger_prompt']!r}")
    print(f"[cell] out={args.out}")

    runner = ExperimentRunner(config)
    runner.run_all()


if __name__ == "__main__":
    main()
