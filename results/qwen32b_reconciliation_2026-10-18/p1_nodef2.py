import json
def rows(path,pred):
    with open(path) as f:
        for l in f:
            l=l.strip()
            if l:
                r=json.loads(l)
                if pred(r): yield r
def p(d,think):
    def f(r):
        m=r['condition'].get('model',{})
        return m.get('model_name')=='qwen3:32b' and r['condition']['defense']['name']==d and r['condition']['attack']['type']=='delayed_trigger' and (think is None or m.get('think')==think)
    return f
def ev(r):
    return [(e.get('session_index'),e.get('tool_name'),e.get('operation'),json.dumps(e.get('inputs'),sort_keys=True),json.dumps(e.get('outputs'),sort_keys=True)) for e in r['tool_logs']]
for dn,(da,dj) in {'no_defense':('no_defense','no_defense'),'sandbox':('memory_sandbox','sandbox_full')}.items():
    A=next(rows('results/defense_factorial/results.jsonl',p(da,None))); J=next(rows('results/sandbox_inversion/results.jsonl',p(dj,False)))
    ea,ej=ev(A),ev(J)
    print(dn,'event count APR',len(ea),'JUN',len(ej))
    n=min(len(ea),len(ej)); first=None
    # compare as ordered lists by session
    for i in range(n):
        if ea[i]!=ej[i]: first=i; break
    print(' first differing event index',first)
    if first is not None:
        for nm,e in (('APR',ea[first]),('JUN',ej[first])):
            print('  ',nm,e[0],e[1],e[2],e[3][:160],'->',e[4][:160])
    # no_defense: also compare agent_logs text content hash
    ta=json.dumps([x.get('content') for x in A['agent_logs'] if x.get('type')=='tool_output'],sort_keys=True); tj=json.dumps([x.get('content') for x in J['agent_logs'] if x.get('type')=='tool_output'],sort_keys=True)
    print(' agent tool_output contents identical:',ta==tj)
