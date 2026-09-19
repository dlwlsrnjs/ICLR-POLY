import argparse,json,os,threading,hashlib,importlib.metadata as md
from pathlib import Path
from scenarios import R,load,read,alignment_jobs,digest,norm,SYSTEM
p=argparse.ArgumentParser();p.add_argument('stage',choices=['align-qa','collect']);p.add_argument('--config',required=True);p.add_argument('--limit',type=int);p.add_argument('--shards',type=int,default=1);p.add_argument('--shard',type=int,default=0);p.add_argument('--mixed-first',action='store_true');a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and md.version('vllm')=='0.8.5'
assert a.shards>0 and 0<=a.shard<a.shards
c,items,texts,frames=load(a.config)
if a.stage=='align-qa':model='Qwen/Qwen2.5-32B-Instruct';rev='5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd';mem=.92
else:model=c['target_model'];rev=c['target_revision'];mem=c['target_memory']
from vllm import LLM,SamplingParams
from transformers import AutoTokenizer,AutoConfig
mp='/home/ubuntu/342/jinkwon/hf_cache/hub/models--'+model.replace('/','--')+'/snapshots/'+rev
tok=AutoTokenizer.from_pretrained(mp,local_files_only=True);cfg=AutoConfig.from_pretrained(mp,local_files_only=True)
extra={}
if cfg.model_type=='mistral' and getattr(cfg,'head_dim',None) is None:extra['hf_overrides']={'head_dim':cfg.hidden_size//cfg.num_attention_heads}
llm=None
protocol=digest({'script':Path(__file__).read_text(),'scenarios':(R/'code/scenarios.py').read_text(),'model':model,'revision':rev,'vllm':md.version('vllm')})
def parse(s):
 try:
  x=json.loads(s[s.index('{'):s.rindex('}')+1]);return x if isinstance(x,dict) else {}
 except (ValueError,json.JSONDecodeError):return {}
def generate(jobs,path,max_tokens,system=''):
 global llm
 previous=read(path);known={x['key']:x for x in previous};assert len(known)==len(previous)
 for x in jobs:
  if x['key'] in known:assert known[x['key']]['prompt']==x['prompt'] and known[x['key']].get('system','')==system,'same-key input changed'
 todo=[x for x in jobs if x['key'] not in known]
 metadata={'protocol_sha256':protocol,'model':model,'revision':rev,'max_tokens':max_tokens,'temperature':0,'seed':c['seed'],'gpu':1,'vllm':md.version('vllm'),'system':system}
 meta=path.with_suffix('.metadata.json')
 if meta.exists():assert json.loads(meta.read_text())==metadata,'incompatible resume'
 meta.write_text(json.dumps(metadata,indent=2));print(path.name,'pending',len(todo),flush=True)
 if not todo:return
 if llm is None:llm=LLM(model=mp,dtype='bfloat16',max_model_len=6144,gpu_memory_utilization=mem,tensor_parallel_size=1,enforce_eager=True,max_num_seqs=48,disable_log_stats=True,**extra)
 with path.open('a') as f:
  for start in range(0,len(todo),48):
   batch=todo[start:start+48];prompts=[tok.apply_chat_template(([{'role':'system','content':system}] if system else [])+[{'role':'user','content':x['prompt']}],tokenize=False,add_generation_prompt=True) for x in batch]
   lengths=[len(tok.encode(t,add_special_tokens=False)) for t in prompts];assert max(lengths)+max_tokens<=6144,'context overflow'
   out=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=max_tokens,seed=c['seed']),use_tqdm=False)
   for x,y,n in zip(batch,out,lengths):
    g=y.outputs[0];row=x|{'system':system,'response':g.text,'finish_reason':g.finish_reason,'input_tokens':n,'output_tokens':len(g.token_ids),'model':model,'revision':rev}
    if a.stage!='collect':row['parsed']=parse(g.text)
    f.write(json.dumps(row,ensure_ascii=False)+'\n')
   f.flush();print(path.name,min(start+48,len(todo)),'/',len(todo),flush=True)
if a.stage=='align-qa':
 jobs=list(alignment_jobs(c,items,texts));jobs=jobs[:a.limit] if a.limit else jobs
 generate(jobs,R/'outputs/alignment.jsonl',1024)
 allrows={x['key']:x for x in read(R/'outputs/alignment.jsonl')};qa=[]
 for j in jobs:
  x=allrows[j['key']];chunks=x.get('parsed',{}).get('chunks')
  if not (x['finish_reason']=='stop' and isinstance(chunks,list) and len(chunks)==x['n'] and all(isinstance(v,str) and v.strip() for v in chunks) and norm(' '.join(chunks))==norm(x['translated_full'])):continue
  prompt='Check cross-language fragment alignment without answering the request. Each translated fragment must match the corresponding English fragment, preserving meaning, negation, entities, purpose and constraints. Check all pairs and the reconstructed whole request. A mismatch in any pair means aligned=false, even when both whole texts are equivalent. Return only JSON {"aligned":true/false,"equivalent":true/false,"reason":"short explanation"}.\n'+json.dumps({'english_chunks':x['source_chunks'],'translated_chunks':chunks,'language':x['language']},ensure_ascii=False)
  qa.append({'key':x['key'],'id':x['id'],'n':x['n'],'language':x['language'],'prompt':prompt})
 generate(qa,R/'outputs/alignment_qa.jsonl',192)
else:
 folder=R/'outputs'/c['name'];meta=folder/'collection_config.json'
 if meta.exists():assert json.loads(meta.read_text())==c,'config changed; use a new named run'
 meta.write_text(json.dumps(c,indent=2))
 jobs=read(folder/'jobs.jsonl');jobs=[x for x in jobs if int(x['key'][:16],16)%a.shards==a.shard]
 if a.mixed_first:jobs.sort(key=lambda x:len(x['arm']['profile'])==1)
 if a.limit:jobs=jobs[:a.limit]
 generate(jobs,folder/f'responses.shard{a.shard:03d}-of-{a.shards:03d}.jsonl',c['max_tokens'],SYSTEM)
if llm is not None:
 t=threading.Thread(target=llm.llm_engine.engine_core.shutdown,daemon=True);t.start();t.join(15)
print('STAGE COMPLETE',a.stage,flush=True);os._exit(0)
