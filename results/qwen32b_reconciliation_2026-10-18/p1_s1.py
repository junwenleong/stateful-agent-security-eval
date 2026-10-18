import json,collections
def rows(path,pred):
    with open(path) as f:
        for l in f:
            l=l.strip()
            if not l: continue
            r=json.loads(l)
            if pred(r): yield r
def mk(dname,think):
    def p(r):
        m=r['condition'].get('model',{})
        return m.get('model_name')=='qwen3:32b' and r['condition']['defense']['name']==dname and r['condition']['attack']['type']=='delayed_trigger' and (think is None or m.get('think')==think)
    return p
sets={
 'APR no_defense':('results/defense_factorial/results.jsonl',mk('no_defense',None)),
 'APR memory_sandbox':('results/defense_factorial/results.jsonl',mk('memory_sandbox',None)),
 'JUN no_defense think=F':('results/sandbox_inversion/results.jsonl',mk('no_defense',False)),
 'JUN sandbox_full think=F':('results/sandbox_inversion/results.jsonl',mk('sandbox_full',False)),
}
# attack dict diff
A=next(rows(*sets['APR memory_sandbox'])); J=next(rows(*sets['JUN sandbox_full think=F']))
a=A['condition']['attack']; j=J['condition']['attack']
for k in sorted(set(a)|set(j)):
    if a.get(k)!=j.get(k): print('ATTACK DIFF',k,'APR=',json.dumps(a.get(k))[:300],'JUN=',json.dumps(j.get(k))[:300])
for k in sorted(set(A['condition']['model'])|set(J['condition']['model'])):
    if A['condition']['model'].get(k)!=J['condition']['model'].get(k): print('MODEL DIFF',k,A['condition']['model'].get(k),'|',J['condition']['model'].get(k))
def s1sig(r):
    ev=[]
    for e in r['tool_logs']:
        if e.get('session_index')==0:
            op=e.get('operation'); tn=e.get('tool_name')
            ins=e.get('inputs') or {}
            if tn=='RAGTool': ev.append('rag(k=%s)'%ins.get('top_k'))
            elif tn=='MemoryTool': ev.append('mem.'+str(op))
            elif tn=='EmailTool': ev.append('email.'+str(op))
            else: ev.append(str(tn)+'.'+str(op))
    return tuple(ev)
for name,(path,p) in sets.items():
    n=0; asr=0; send_s1=0; save_counts=collections.Counter(); rag_counts=collections.Counter(); exfil_sess=collections.Counter(); sig=collections.Counter(); s4=[]; s1t=[]
    for r in rows(path,p):
        n+=1; asr+=bool(r['attack_success']); exfil_sess[r.get('exfiltration_session_index')]+=1
        sg=s1sig(r); sig[sg]+=1
        save_counts[sum(1 for x in sg if x=='mem.save_fact')]+=1
        rag_counts[sum(1 for x in sg if x.startswith('rag'))]+=1
        send_s1+=any(x.startswith('email.send') for x in sg)
        s1t.append(r['session_timings_ms'][0]/1000); s4.append(r['session_timings_ms'][3]/1000)
    s1t.sort(); s4.sort()
    print('\n==',name,'N',n,'ASR',asr,'S1 has send',send_s1,'exfil_session',dict(exfil_sess))
    print(' S1 save_fact counts',dict(save_counts),'| S1 rag counts',dict(rag_counts))
    print(' S1 time median %.0fs  S4 time median %.1fs'%(s1t[n//2],s4[n//2]))
    for s,c in sig.most_common(3): print('  S1 sig x%d: %s'%(c,' > '.join(s)))
