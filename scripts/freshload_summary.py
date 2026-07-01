"""Summarize the fresh-load suite results (RATG + 7B judge).

Reports, per model, the no_defense (fresh-load baseline) ASR and the defended ASR,
so the reduction is directly readable with a clean baseline (unlike the sprint,
whose reasoning-model baselines were degraded).

Usage:
    .venv/bin/python scripts/freshload_summary.py
"""
import json
from pathlib import Path
from collections import defaultdict


def summarize(folder: str, defended_names: set[str], label: str) -> None:
    d = Path(folder)
    files = sorted(d.glob("*.jsonl")) if d.exists() else []
    if not files:
        print(f"[{label}] no results in {folder}/")
        return
    print(f"\n=== {label}  ({folder}) ===")
    print(f"  {'model':26s} {'no_defense':>14s} {'defended':>14s}  reduction")
    print(f"  {'-'*26} {'-'*14} {'-'*14}  {'-'*10}")
    for f in files:
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        by_def = defaultdict(lambda: {"atk": 0, "n": 0})
        model = None
        for r in recs:
            model = r["condition"]["model"]["model_name"]
            name = (r["condition"].get("defense", {}) or {}).get("name") or "no_defense"
            by_def[name]["n"] += 1
            if r.get("attack_success"):
                by_def[name]["atk"] += 1
        nd = by_def.get("no_defense", {"atk": 0, "n": 0})
        defn = next((n for n in by_def if n in defended_names), None)
        dv = by_def.get(defn, {"atk": 0, "n": 0}) if defn else {"atk": 0, "n": 0}
        nd_s = f"{nd['atk']}/{nd['n']}" if nd["n"] else "-"
        dv_s = f"{dv['atk']}/{dv['n']}" if dv["n"] else "-"
        red = ""
        if nd["n"] and dv["n"]:
            red = f"{nd['atk']/nd['n']*100:.0f}% -> {dv['atk']/dv['n']*100:.0f}%"
        print(f"  {model:26s} {nd_s:>14s} {dv_s:>14s}  {red}")


def main() -> None:
    summarize("results/freshload_ratg", {"ratg"}, "Reasoning-model RATG (fresh load)")
    summarize("results/freshload_judge7b", {"rag_llm_judge_7b", "rag_llm_judge"}, "7B RAG-LLM-Judge (fresh load)")
    print("\nRead: no_defense should be ~100% (clean fresh-load baseline, unlike the")
    print("degraded sprint). Defended column shows whether the defense reduces ASR.")


if __name__ == "__main__":
    main()
