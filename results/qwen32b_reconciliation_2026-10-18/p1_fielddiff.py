import json,collections
def first(path,pred,n=1):
    out=[]
    with open(path) as f:
        for l in f:
            l=l.strip()
            if not l: continue
            r=json.loads(l)
            if pred(r):
                out.append(r)
                if len(out)>=n: break
    return out
def p_apr(r):
    m=r['condition'].get('model',{}); return m.get('model_name')=='qwen3:32b' and r['condition']['defense']['name']=='memory_sandbox' and r['condition']['attack']['type']=='delayed_trigger'
def p_jun(r):
    m=r['condition'].get('model',{}); return m.get('model_name')=='qwen3:32b' and m.get('think') is False and r['condition']['defense']['name']=='sandbox_full'
A=first('results/defense_factorial/results.jsonl',p_apr)[0]
J=first('results/sandbox_inversion/results.jsonl',p_jun)[0]
for k in ['temperature_used','excluded_tools_list','defense_schema_version','agent_model_hash','judge_model_hash','runner_config','error','run_timestamp','session_timings_ms','timing_ms','exfiltration_session_index','injection_session_memory_calls','rag_called_in_injection','rag_called_in_trigger','memory_recalled_in_trigger','mechanistic_tags','trigger_steps_before_exfil']:
    print('--',k); print(' APR:',json.dumps(A.get(k))[:400]); print(' JUN:',json.dumps(J.get(k))[:400])
print('== condition.model'); print(' APR',json.dumps(A['condition']['model'])); print(' JUN',json.dumps(J['condition']['model']))
print('== condition.defense'); print(' APR',json.dumps(A['condition']['defense'])); print(' JUN',json.dumps(J['condition']['defense']))
ka=set(A['condition'].keys()); kj=set(J['condition'].keys()); print('cond keys only APR',ka-kj,'only JUN',kj-ka)
for k in sorted(ka&kj):
    if k in('model','defense'): continue
    a=json.dumps(A['condition'][k],sort_keys=True); j=json.dumps(J['condition'][k],sort_keys=True)
    print('cond',k,'SAME' if a==j else 'DIFFERENT')
    if a!=j: print('   APR',a[:700]); print('   JUN',j[:700])
sa=A.get('system_prompt_used'); sj=J.get('system_prompt_used')
print('== system_prompt_used same?', sa==sj, len(sa or ''), len(sj or ''))
if sa!=sj:
    import difflib
    for l in difflib.unified_diff((sa or '').splitlines(),(sj or '').splitlines(),'apr','jun',lineterm=''): print(l[:300])
