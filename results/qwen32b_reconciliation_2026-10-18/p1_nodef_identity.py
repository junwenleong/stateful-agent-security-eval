import json,hashlib,re
def rows(path,pred):
    with open(path) as f:
        for l in f:
            l=l.strip()
            if l:
                r=json.loads(l)
                if pred(r): yield r
def norm(r):
    # content of tool logs with timestamps removed
    out=[]
    for e in r['tool_logs']:
        d={k:v for k,v in e.items() if k!='timestamp'}
        out.append(d)
    return hashlib.md5(json.dumps(out,sort_keys=True).encode()).hexdigest()
def p(d,think):
    def f(r):
        m=r['condition'].get('model',{})
        return m.get('model_name')=='qwen3:32b' and r['condition']['defense']['name']==d and r['condition']['attack']['type']=='delayed_trigger' and (think is None or m.get('think')==think)
    return f
a=[norm(r) for r in rows('results/defense_factorial/results.jsonl',p('no_defense',None))]
j=[norm(r) for r in rows('results/sandbox_inversion/results.jsonl',p('no_defense',False))]
print('APR no_defense distinct tool_log hashes',len(set(a)),'JUN',len(set(j)),'overlap',len(set(a)&set(j)))
a=[norm(r) for r in rows('results/defense_factorial/results.jsonl',p('memory_sandbox',None))]
j=[norm(r) for r in rows('results/sandbox_inversion/results.jsonl',p('sandbox_full',False))]
print('sandbox: APR distinct',len(set(a)),'JUN distinct',len(set(j)),'overlap',len(set(a)&set(j)))
# S1 sandbox: compare S1 events only (first 5 saves identical?)
def s1(r): 
    return [(e.get('tool_name'),e.get('operation'),json.dumps(e.get('inputs'),sort_keys=True)[:200]) for e in r['tool_logs'] if e.get('session_index')==0]
A=next(rows('results/defense_factorial/results.jsonl',p('memory_sandbox',None))); J=next(rows('results/sandbox_inversion/results.jsonl',p('sandbox_full',False)))
for nm,r in (('APR',A),('JUN',J)):
    print('--',nm,'S1 events (in logged order)')
    for e in r['tool_logs']:
        if e.get('session_index')==0:
            print('  ',e.get('timestamp','')[11:23],e.get('tool_name'),e.get('operation'),json.dumps(e.get('inputs'))[:110])
