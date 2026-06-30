#!/usr/bin/env python3
"""Sprint monitor — run with: watch -n 10 .venv/bin/python scripts/monitor_sprint.py"""
import json, sys
from pathlib import Path
from collections import Counter
from datetime import datetime

print(f"=== RATG Sprint {datetime.now().strftime('%H:%M')} ===\n")

files = {
    "payload_variants": ("results/payload_variants/results.jsonl", 50),
    "ratg_factorial": ("results/ratg_factorial/results.jsonl", 720),
    "judge_7b": ("results/judge_7b/results.jsonl", 240),
}

for name, (path, target) in files.items():
    p = Path(path)
    n = len(p.read_text().splitlines()) if p.exists() else 0
    pct = int(n / target * 100) if target else 0
    phase = "✅" if n >= target else "🔄" if n > 0 else "⏳"
    print(f"  {phase} {name:20s} {n:4d}/{target} ({pct}%)")

print()

# Phase 2 detail
ratg_path = Path("results/ratg_factorial/results.jsonl")
if ratg_path.exists():
    lines = ratg_path.read_text().splitlines()
    if lines:
        by_model = Counter()
        for l in lines:
            r = json.loads(l)
            by_model[r["condition"]["model"]["model_name"]] += 1

        print("--- Phase 2 by model ---")
        for m, n in sorted(by_model.items(), key=lambda x: -x[1]):
            bar = "█" * int(n / 80 * 20) + "░" * (20 - int(n / 80 * 20))
            print(f"  {m:30s} {bar} {n}/80")

        print()
        last = json.loads(lines[-1])
        m = last["condition"]["model"]["model_name"]
        d = last["condition"]["defense"]["name"]
        t = last["timing_ms"] / 1000
        asr = "✓" if last.get("attack_success") else "✗"
        print(f"  Last: {m} | {d} | {t:.0f}s | ASR={asr}")

# Sprint log
log = Path("sprint.log")
if log.exists():
    tail = log.read_text().splitlines()[-1:] 
    if tail:
        print(f"\n  Log: {tail[0][-80:]}")
