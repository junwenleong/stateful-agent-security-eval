"""Deep analysis of the completed fresh-load suite (RATG + 7B judge).

Applies the locked per-model, per-arm, S0-identity protocol (Iterations 57-59):
- Never read ASR alone.
- For RATG (activates at first recall_fact in trigger): S0 MUST be identical across
  arms. If S0 forks (save-count / op-seq differ), the comparison is uninterpretable
  because RATG is provably inert in S0.
- For the 7B judge (activates in S0 as a retrieval filter): S0 is EXPECTED to differ;
  the question is whether the judge detected/removed the malicious doc.
"""
import json
from collections import Counter, defaultdict


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def arm_of(r):
    d = r["condition"].get("defense", {})
    return d.get("name", "?") if isinstance(d, dict) else str(d)


def s0_opseq(r):
    """Ordered tool op names in session 0 (injection), timestamp-sorted (Artifact 8)."""
    logs = [t for t in (r.get("tool_logs") or []) if t.get("session_index") == 0]
    logs = sorted(logs, key=lambda t: t.get("timestamp", 0))
    return tuple(t.get("operation", "?") for t in logs)


def summarize_arm(rows):
    n = len(rows)
    errs = sum(1 for r in rows if r.get("error"))
    inj = sum(1 for r in rows if r.get("injection_success"))
    asr = sum(1 for r in rows if r.get("attack_success"))
    btcr_ua = sum(1 for r in rows if r.get("btcr_success_under_attack"))
    recall = sum(1 for r in rows if r.get("memory_recalled_in_trigger"))
    ragtrig = sum(1 for r in rows if r.get("rag_called_in_trigger"))
    s0saves = Counter(r.get("injection_session_memory_calls") for r in rows)
    exfilsess = Counter(r.get("exfiltration_session_index") for r in rows)
    tags = Counter()
    for r in rows:
        for t in (r.get("mechanistic_tags") or []):
            tags[t] += 1
    influence = sum(1 for r in rows if r.get("instruction_influence"))
    return dict(n=n, errs=errs, inj=inj, asr=asr, btcr_ua=btcr_ua, recall=recall,
                ragtrig=ragtrig, s0saves=s0saves, exfilsess=exfilsess, tags=tags,
                influence=influence)


def pct(x, n):
    return f"{100*x/n:4.1f}%" if n else "  n/a"


RATG = "results/freshload_ratg"
JUDGE = "results/freshload_judge7b"

ratg_models = ["qwen3_5_122b", "qwen3_5_9b", "qwq_32b", "gpt-oss_20b",
               "gpt-oss-safeguard_120b", "glm-4_7-flash_bf16"]
judge_models = ["qwen2_5_14b", "qwen3_32b", "qwen3_5_122b"]

print("=" * 78)
print("RATG SUITE  (activation = first recall_fact in TRIGGER; S0 must match across arms)")
print("=" * 78)
for m in ratg_models:
    rows = load(f"{RATG}/{m}.jsonl")
    byarm = defaultdict(list)
    for r in rows:
        byarm[arm_of(r)].append(r)
    print(f"\n### {m}   (total {len(rows)})")
    digests = set(r.get("agent_model_hash") for r in rows)
    overs = set(r.get("ollama_version") for r in rows)
    print(f"    digest={digests}  ollama={overs}")
    arm_summ = {}
    arm_opseq = {}
    for arm in ["no_defense", "ratg"]:
        rs = byarm.get(arm, [])
        if not rs:
            print(f"    [{arm}] MISSING")
            continue
        s = summarize_arm(rs)
        arm_summ[arm] = s
        arm_opseq[arm] = Counter(s0_opseq(r) for r in rs)
        print(f"    [{arm:10s}] N={s['n']:2d} err={s['errs']} "
              f"inj={pct(s['inj'],s['n'])} ASR={pct(s['asr'],s['n'])} "
              f"BTCR_ua={pct(s['btcr_ua'],s['n'])} recall={pct(s['recall'],s['n'])} "
              f"ragTrig={pct(s['ragtrig'],s['n'])} infl={pct(s['influence'],s['n'])}")
        print(f"                 S0 save-count dist: {dict(s['s0saves'])}")
        print(f"                 exfil-session dist: {dict(s['exfilsess'])}")
        print(f"                 tags: {dict(s['tags'])}")
    # S0-identity verdict
    if "no_defense" in arm_summ and "ratg" in arm_summ:
        nd_saves = arm_summ["no_defense"]["s0saves"]
        rt_saves = arm_summ["ratg"]["s0saves"]
        s0_ident = (nd_saves == rt_saves) and (arm_opseq["no_defense"] == arm_opseq["ratg"])
        verdict = "S0-IDENTICAL -> INTERPRETABLE" if s0_ident else "S0 FORK -> UNINTERPRETABLE"
        print(f"    >>> S0-identity: {verdict}")

print("\n" + "=" * 78)
print("7B-JUDGE SUITE  (activation = S0 retrieval filter; S0 differs by design)")
print("=" * 78)
for m in judge_models:
    rows = load(f"{JUDGE}/{m}.jsonl")
    byarm = defaultdict(list)
    for r in rows:
        byarm[arm_of(r)].append(r)
    print(f"\n### {m}   (total {len(rows)})")
    for arm in ["no_defense", "rag_llm_judge_7b"]:
        rs = byarm.get(arm, [])
        if not rs:
            print(f"    [{arm}] MISSING")
            continue
        s = summarize_arm(rs)
        # judge removal signal from rag_logs
        removed = 0
        judged = 0
        for r in rs:
            for rl in (r.get("rag_logs") or []):
                for jd in (rl.get("judge_decisions") or []):
                    judged += 1
                    if jd.get("is_malicious"):
                        removed += 1
        print(f"    [{arm:16s}] N={s['n']:2d} err={s['errs']} "
              f"inj={pct(s['inj'],s['n'])} ASR={pct(s['asr'],s['n'])} "
              f"BTCR_ua={pct(s['btcr_ua'],s['n'])}")
        print(f"                       tags: {dict(s['tags'])}")
        if arm != "no_defense":
            print(f"                       judge_decisions: {judged} docs judged, "
                  f"{removed} flagged malicious")
