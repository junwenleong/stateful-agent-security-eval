import json
from pathlib import Path

def show(path, label, total):
    p = Path(path)
    if not p.exists():
        print(f"{label}: not started")
        return
    lines = p.read_text().splitlines()
    n = len(lines)
    dta = [json.loads(l) for l in lines if "delayed_trigger" in l]
    asr = sum(1 for r in dta if r.get("attack_success")) / max(len(dta), 1) * 100
    inj = sum(1 for r in dta if r.get("injection_success")) / max(len(dta), 1) * 100
    print(f"{label}: {n}/{total} | DTA:{len(dta)} Inj:{inj:.0f}% ASR:{asr:.0f}% | no_attack:{n - len(dta)}")

show("results/bedrock_tier1/results.jsonl", "Tier1", 960)
show("results/bedrock_tier2/results.jsonl", "Tier2", 140)
