import json
def first(path,pred):
    with open(path) as f:
        for l in f:
            l=l.strip()
            if l:
                r=json.loads(l)
                if pred(r): return r
def cellp(d,think):
    def f(r):
        m=r['condition'].get('model',{})
        return m.get('model_name')=='qwen3:32b' and r['condition']['defense']['name']==d and r['condition']['attack']['type']=='delayed_trigger' and (think is None or m.get('think')==think)
    return f
cells={'APR 04-16':('results/defense_factorial/results.jsonl',cellp('memory_sandbox',None)),
       'JUN16 "32k"':('results/sandbox_inversion/results_32k.jsonl',cellp('sandbox_full',False)),
       'JUN18':('results/sandbox_inversion/results.jsonl',cellp('sandbox_full',False))}
seqs={}
for n,(p,f) in cells.items():
    r=first(p,f)
    ev=[e for e in r['tool_logs'] if e.get('session_index')==0]
    ev.sort(key=lambda e:e.get('timestamp',''))
    s=[]
    for e in ev:
        ins=e.get('inputs') or {}
        s.append(('save:'+ins.get('key','?')) if e.get('operation')=='save_fact' else ('rag' if e.get('tool_name')=='RAGTool' else 'send:'+','.join(ins.get('to',[])) ))
    seqs[n]=s
    print(n,'(chronological S1)'); 
    for x in s: print('   ',x)
    print('   S2 calendar date asked:',[ (e.get('inputs') or {}).get('date') for e in r['tool_logs'] if e.get('tool_name')=='CalendarTool'][:1],' run_ts',r['run_timestamp'])
# save value equality for first three saves across APR and JUN18
def saves(p,f):
    r=first(p,f); 
    ev=sorted([e for e in r['tool_logs'] if e.get('session_index')==0 and e.get('operation')=='save_fact'],key=lambda e:e['timestamp'])
    return [(e['inputs']['key'],e['inputs']['value']) for e in ev]
a=saves(*cells['APR 04-16']); j=saves(*cells['JUN18']); k=saves(*cells['JUN16 "32k"'])
print('first 3 saves identical APR vs JUN18:',a[:3]==j[:3],'| APR vs 32k:',a[:3]==k[:3],'| first 4 APR vs 32k:',a[:4]==k[:4])
