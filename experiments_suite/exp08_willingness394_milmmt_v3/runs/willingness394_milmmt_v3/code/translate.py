import argparse,json,os,hashlib,threading,importlib.metadata as md
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R.parents[1]
MODEL='xiaomi-research/MiLMMT-46-12B-v1.0';REV='a27dbbb37142ff076990820a1c9f0827beb5d6ea'
p=argparse.ArgumentParser();p.add_argument('--limit',type=int);a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and md.version('vllm')=='0.8.5'
items=json.loads((R/'inputs/items.json').read_text());assert len(items)==394
jobs=[{'key':x['id']+'|'+l,'id':x['id'],'language':l,'original':x['prompt']} for x in items for l in ['Norwegian','Finnish','Arabic']]
if a.limit:jobs=jobs[:a.limit]
def read(p):return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []
meta={'model':MODEL,'revision':REV,'gpu':1,'vllm':md.version('vllm'),'temperature':0,'max_tokens':1024,'seed':20260916,'protocol_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'items_sha256':hashlib.sha256((R/'inputs/items.json').read_bytes()).hexdigest(),'prompt_template':'Translate this from {source} to {target}:\n{source}: {text}\n{target}:'}
mp=R/'outputs/translation.metadata.json'
if mp.exists():assert json.loads(mp.read_text())==meta,'incompatible resume'
mp.write_text(json.dumps(meta,indent=2))
from transformers import AutoTokenizer
from vllm import LLM,SamplingParams
modelpath=B/'model_cache'/('models--'+MODEL.replace('/','--'))/'snapshots'/REV
tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True);llm=None
def config_compat(cfg):
 # vLLM 0.8.5 reads the former public name; preserve the model's exact pattern.
 tc=cfg.text_config
 pattern=tc._sliding_window_pattern
 assert pattern==6
 assert tc.layer_types==['full_attention' if (i+1)%pattern==0 else 'sliding_attention' for i in range(tc.num_hidden_layers)]
 tc.sliding_window_pattern=pattern
 return cfg

def generate(rows,path,forward):
 global llm
 old=read(path);known={x['key']:x for x in old};assert len(known)==len(old)
 for x in rows:
  if x['key'] in known:
   for k in ['original','language','id']:assert known[x['key']][k]==x[k]
   if not forward:assert known[x['key']]['translated']==x['translated']
 todo=[x for x in rows if x['key'] not in known];print(path.name,'pending',len(todo),flush=True)
 if todo and llm is None:
  llm=LLM(model=str(modelpath),dtype='bfloat16',max_model_len=4096,gpu_memory_utilization=.75,tensor_parallel_size=1,enforce_eager=True,max_num_seqs=32,disable_log_stats=True,limit_mm_per_prompt={'image':0},hf_overrides=config_compat)
 with path.open('a') as f:
  for start in range(0,len(todo),32):
   batch=todo[start:start+32];prompts=[]
   for x in batch:
    source,target,text=('English',x['language'],x['original']) if forward else (x['language'],'English',x['translated'])
    prompts.append(meta['prompt_template'].format(source=source,target=target,text=text))
   ids=[tok.encode(t,add_special_tokens=False) for t in prompts];assert max(map(len,ids))+1024<=4096,'context overflow'
   outputs=llm.generate([{'prompt_token_ids':v} for v in ids],SamplingParams(temperature=0,top_k=1,max_tokens=1024,seed=20260916),use_tqdm=False)
   for x,prompt,ids_,y in zip(batch,prompts,ids,outputs):
    g=y.outputs[0];prefix='forward' if forward else 'back';field='translated' if forward else 'backtranslation'
    row=x|{field:g.text.strip(),prefix+'_terminated':g.finish_reason=='stop' and bool(g.text.strip()),prefix+'_finish_reason':g.finish_reason,prefix+'_prompt':prompt,prefix+'_input_tokens':len(ids_),prefix+'_output_tokens':len(g.token_ids),'translation_model':MODEL,'model_revision':REV}
    f.write(json.dumps(row,ensure_ascii=False)+'\n')
   f.flush();print(path.name,min(start+32,len(todo)),'/',len(todo),flush=True)
forwardpath=R/'outputs/forward.jsonl';generate(jobs,forwardpath,True)
keys={x['key'] for x in jobs};generate([x for x in read(forwardpath) if x['key'] in keys],R/'outputs/translations.jsonl',False)
if llm is not None:
 t=threading.Thread(target=llm.llm_engine.engine_core.shutdown,daemon=True);t.start();t.join(15)
print('TRANSLATION COMPLETE',flush=True);os._exit(0)
