import json,glob,collections,sys
out=collections.defaultdict(lambda:[0,0,set(),set()])
for path in glob.glob('results/**/*.jsonl',recursive=True):
    if 'qwen32b_reconciliation' in path: continue
    try:
        with open(path) as f:
            for l in f:
                l=l.strip()
                if not l or 'qwen3:32b' not in l: continue
                try: r=json.loads(l)
                except: continue
                c=r.get('condition') or {}
                m=c.get('model') or r.get('model') or {}
                if isinstance(m,str): name=m; think=None
                else: name=m.get('model_name') or m.get('name'); think=m.get('think')
                if name!='qwen3:32b': continue
                d=c.get('defense') or {}
                dn=d.get('name') if isinstance(d,dict) else str(d)
                at=(c.get('attack') or {}).get('type') if isinstance(c.get('attack'),dict) else None
                if dn and ('sandbox' in dn or dn=='no_defense') and at in ('delayed_trigger',None):
                    k=(path.split('results/')[1],dn,think,r.get('experiment_label'))
                    o=out[k]; o[0]+=1; o[1]+=bool(r.get('attack_success'))
                    ts=r.get('run_timestamp') or r.get('timestamp') or ''
                    o[2].add(ts[:10]); 
                    ol=r.get('ollama_version') or (r.get('ollama') or {}).get('version') if isinstance(r.get('ollama'),dict) else r.get('ollama_version')
                    if ol: o[3].add(str(ol))
    except Exception as e:
        print('ERR',path,e,file=sys.stderr)
for k,v in sorted(out.items(), key=lambda x:str(x[0])):
    print(k,'N',v[0],'ASR',v[1],'dates',sorted(v[2])[:3],'..',sorted(v[2])[-1:],'ollama',sorted(v[3]))
