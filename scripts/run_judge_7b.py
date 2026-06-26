"""7B RAG LLM Judge — capability threshold test.

Usage:
    .venv/bin/python scripts/run_judge_7b.py
"""
import json, sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    config = load_config("experiments/configs/judge_7b.yaml")
    Path(config.results_path).parent.mkdir(parents=True, exist_ok=True)
    runner = ExperimentRunner(config)
    runner.run_all()

    records = [json.loads(l) for l in Path(config.results_path).read_text().splitlines() if l.strip()]
    valid = [r for r in records if not r.get("error")]
    by_defense = {}
    for r in valid:
        d = r.get("condition", {}).get("defense", {}).get("name", "?")
        by_defense.setdefault(d, []).append(r)

    print("\n" + "=" * 50)
    print("7B JUDGE RESULTS")
    print("=" * 50)
    for d, runs in sorted(by_defense.items()):
        asr = sum(1 for r in runs if r.get("attack_success"))
        print(f"  {d:<25} ASR={asr}/{len(runs)} ({100*asr/len(runs):.1f}%)")

if __name__ == "__main__":
    main()
