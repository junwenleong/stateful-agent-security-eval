import json,hashlib,collections
def rows(path):
    with open(path) as f:
        for l in f:
            l=l.strip()
            if l: yield json.loads(l)
def cell(r):
    c=r['condition']; m=c.get('model',{}); 
    return (m.get('model_name'), m.get('think'), c['defense'].get('name'), (c.get('attack') or {}).get('type'))
def sig(r,sess=None):
    ev=[]
    for e in r['tool_logs']:
        if sess is not None and e.get('session_index')!=sess: continue
        ins=dict(e.get('inputs') or {}); outs=json.dumps(e.get('outputs'),sort_keys=True)
        ev.append((e.get('session_index'),e.get('tool_name'),e.get('operation'),json.dumps(ins,sort_keys=True)))
    return tuple(ev)
want=[('defense_factorial/results.jsonl',('qwen3:32b',None,'memory_sandbox','delayed_trigger')),
      ('sandbox_inversion/results_32k.jsonl',('qwen3:32b',False,'sandbox_full','delayed_trigger')),
      ('sandbox_inversion/results.jsonl',('qwen3:32b',False,'sandbox_full','delayed_trigger')),
      ('defense_factorial/results.jsonl',('qwen3:32b',None,'no_defense','delayed_trigger')),
      ('sandbox_inversion/results_32k.jsonl',('qwen3:32b',False,'no_defense','delayed_trigger')),
      ('sandbox_inversion/results.jsonl',('qwen3:32b',False,'no_defense','delayed_trigger'))]
store={}
for p,k in want:
    rs=[r for r in rows('results/'+p) if cell(r)==k]
    r=rs[0]
    s1=[(e.get('tool_name'),e.get('operation')) for e in r['tool_logs'] if e.get('session_index')==0]
    store[(p,k[2])]=(len(rs),sig(r,0),r)
    c=collections.Counter(s1)
    print(p,k[2],'N',len(rs),'distinct S1 sigs over all runs:',len(set(sig(x,0) for x in rs)),'| S1:',dict(c),'ASR',sum(bool(x['attack_success']) for x in rs),'exfil sess',collections.Counter(x.get('exfiltration_session_index') for x in rs).most_common(2),'S1 median s',sorted(x['session_timings_ms'][0]/1000 for x in rs)[len(rs)//2])
ka=('defense_factorial/results.jsonl','memory_sandbox'); k32=('sandbox_inversion/results_32k.jsonl','sandbox_full'); kj=('sandbox_inversion/results.jsonl','sandbox_full')
print('S1 sig APR == 32k sandbox_full:',store[ka][1]==store[k32][1],'| APR == JUN16k:',store[ka][1]==store[kj][1],'| 32k==JUN16k:',store[k32][1]==store[kj][1])
