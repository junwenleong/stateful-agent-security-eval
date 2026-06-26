import json
from pathlib import Path

d = Path("results/bedrock_date_sweep_n40")
models = [("nemotron", "nvidia_nemotron-super-3-120b", 84), ("minimax", "minimax_minimax-m2_5", 131), ("kimi", "moonshot_kimi-k2-thinking", 301), ("qwen3-next", "qwen_qwen3-next-80b-a3b", 52), ("llama4", "us_meta_llama4-maverick-17b-instruct-v1_0", 5)]
dates = ["2026-04-17", "2026-06-25", "2026-03-15"]

print(f"{'Model':<12}{'N':<7}{'04-17':<10}{'06-25':<10}{'03-15':<10}ETA")
print("-" * 55)
tot = 0
for nm, pf, spd in models:
    t = 0; s = {}
    for dt in dates:
        f = d / f"{pf}_{dt}.jsonl"
        if f.exists():
            ls = [l for l in f.read_text().splitlines() if l.strip()]
            t += len(ls)
            v = [json.loads(l) for l in ls if not json.loads(l).get("error")]
            s[dt] = f"{sum(r.get('attack_success',0) for r in v)}/{len(v)}"
        else:
            s[dt] = "-"
    eta = f"{(120-t)*spd/3600:.1f}h" if t < 120 else "DONE"
    print(f"{nm:<12}{t:<7}{s.get(dates[0],'-'):<10}{s.get(dates[1],'-'):<10}{s.get(dates[2],'-'):<10}{eta}")
    tot += t
print("-" * 55)
print(f"Total: {tot}/600")
