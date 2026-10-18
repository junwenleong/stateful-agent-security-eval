import json,collections,sys
def rows(path):
    with open(path) as f:
        for l in f:
            l=l.strip()
            if l: yield json.loads(l)
def key(r):
    c=r['condition']; m=c.get('model',{})
    return (m.get('model_name'), m.get('think'), c['defense'].get('name'), (c.get('attack') or {}).get('type'))
for path in ['results/defense_factorial/results.jsonl','results/sandbox_inversion/results.jsonl','results/sandbox_inversion/results_32k.jsonl']:
    agg=collections.defaultdict(lambda:[0,0,0,[]])
    for r in rows(path):
        k=key(r)
        if k[0] and 'qwen3:32b' in k[0]:
            a=agg[k]; a[0]+=1; a[1]+=bool(r['attack_success']); a[2]+=bool(r['injection_success']); a[3].append(r.get('run_timestamp'))
    print('==',path)
    for k,v in sorted(agg.items(), key=lambda x:str(x[0])):
        print(k,'N',v[0],'ASR',v[1],'inj',v[2],'ts',min(v[3]),'..',max(v[3]))
