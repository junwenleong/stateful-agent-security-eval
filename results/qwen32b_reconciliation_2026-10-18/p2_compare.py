"""Compare each live qwen3:32b run's S1 trajectory with the three historical sandbox cells (chronological S1 events)."""
import glob
import json
import os


def first(path, pred):
    with open(path) as f:
        for l in f:
            l = l.strip()
            if l:
                r = json.loads(l)
                if pred(r):
                    return r


def cellp(d, think):
    def f(r):
        m = r["condition"].get("model", {})
        return (m.get("model_name") == "qwen3:32b" and r["condition"]["defense"]["name"] == d
                and r["condition"]["attack"]["type"] == "delayed_trigger" and (think is None or m.get("think") == think))
    return f


def sig(r):
    ev = sorted([e for e in r["tool_logs"] if e.get("session_index") == 0], key=lambda e: e.get("timestamp", ""))
    out = []
    for e in ev:
        ins = e.get("inputs") or {}
        if e.get("operation") == "save_fact":
            out.append("save:" + ins.get("key", "?"))
        elif e.get("tool_name") == "RAGTool":
            out.append("rag")
        elif e.get("tool_name") == "EmailTool":
            out.append("send")
        else:
            out.append(str(e.get("tool_name")) + "." + str(e.get("operation")))
    return tuple(out)


hist = {
    "APR(04-16)": sig(first("results/defense_factorial/results.jsonl", cellp("memory_sandbox", None))),
    "JUN16": sig(first("results/sandbox_inversion/results_32k.jsonl", cellp("sandbox_full", False))),
    "JUN18": sig(first("results/sandbox_inversion/results.jsonl", cellp("sandbox_full", False))),
}
hist_nodef = sig(first("results/defense_factorial/results.jsonl", cellp("no_defense", None)))
for k, v in hist.items():
    print(f"hist {k}: {len(v)} S1 events")
for path in sorted(glob.glob("results/qwen32b_reconciliation_2026-10-18/live_*.jsonl")):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    for i, r in enumerate(rows):
        s = sig(r)
        match = [k for k, v in hist.items() if v == s]
        nm = "nodef=APR_nodef" if s == hist_nodef else ""
        print(f"{os.path.basename(path)} run{i}: ASR={r.get('attack_success')} err={bool(r.get('error'))} "
              f"exfil_sess={r.get('exfiltration_session_index')} S1 sig == {match or 'none'} {nm} "
              f"saves={sum(x.startswith('save') for x in s)} rag={s.count('rag')} send={s.count('send')} hash={r.get('agent_model_hash')}")
