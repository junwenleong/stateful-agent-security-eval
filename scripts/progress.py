import json
lines = open("results/bedrock_tier1/results.jsonl").readlines()
n = len(lines)
dta = [json.loads(l) for l in lines if "delayed_trigger" in l]
asr = sum(1 for r in dta if r.get("attack_success")) / max(len(dta), 1) * 100
print(f"{n}/960 | DTA:{len(dta)} ASR:{asr:.0f}% | no_attack:{n - len(dta)}")
