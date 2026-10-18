import json, sys, collections

path = sys.argv[1]
rows = [json.loads(l) for l in open(path) if l.strip()]
print(f"    {path.split('/')[-1]}: runs={len(rows)}")
for i, r in enumerate(rows):
    s1 = [(e.get("tool_name"), e.get("operation")) for e in r.get("tool_logs", []) if e.get("session_index") == 0]
    c = collections.Counter(s1)
    date_seen = [e.get("date") for e in r.get("tool_logs", []) if e.get("tool_name") == "CalendarTool"]
    print(
        f"    run{i}: err={bool(r.get('error'))} ASR={r.get('attack_success')} inj={r.get('injection_success')} "
        f"exfil_session={r.get('exfiltration_session_index')} S1 saves={c.get(('MemoryTool','save_fact'),0)} "
        f"rag={c.get(('RAGTool','query'),0)} send={c.get(('EmailTool','send_email'),0)} "
        f"S1_s={r['session_timings_ms'][0]/1000:.0f} model_hash={r.get('agent_model_hash')} cal_dates={date_seen[:1]}"
    )
